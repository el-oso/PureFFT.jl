# Batched-strided pow2 FFT along a strided dim WITHOUT transpose: vectorize across the contiguous
# `inner` batch (W=4 complex per Vec{8,T}), running a mixed-radix-8 DIT on row-vectors with
# scalar-broadcast twiddles, out of a cache-resident pack-tile. This is the validated Task-6e kernel
# (scratchpad/batched8_proto.jl) cleaned for src/: hot path is pointer-based + GC.@preserve (no
# `reshape(x,:)`, which allocates an array header and would break the zero-alloc gate) and zero-alloc.
# Re-uses AvxRadix's tuned, width-generic column butterflies (avx_column_butterfly8/4) verbatim.

using SIMD: Vec, vload, vstore

# Batch width = full 512-bit per element type: W complex per Vec{L,T}, L = 2W lanes. F64 → 4 complex
# (Vec{8,Float64}); F32 → 8 complex (Vec{16,Float32}, needs AVX-512 — else fall back to 4-complex
# Vec{8,Float32} so non-AVX512 builds stay byte-identical to the old path). The batched loads are
# CONTIGUOUS across the inner batch (plain vload, not a gather) ⇒ the wider F32 vector is a genuine ~2×.
@inline _batch_lanes(::Type{Float64}) = 8
@inline _batch_lanes(::Type{Float32}) = AvxRadix._HAS_AVX512 ? 16 : 8

# Pointer-based W-complex load/store (i = 0-based complex index; complex = 2 T ⇒ byte stride 2·sizeof(T)).
@inline _ldv(::Type{Vec{L, T}}, pt::Ptr{T}, i::Int) where {L, T} = vload(Vec{L, T}, pt + i * 2 * sizeof(T))
@inline _stv!(pt::Ptr{T}, i::Int, v::Vec{L, T}) where {L, T} = vstore(v, pt + i * 2 * sizeof(T))
# Broadcast a complex (re,im) to all W lanes of a Vec{L,T}: lanes [re,im,re,im,…]. Compile-time ntuple.
@inline _bcast_c(::Type{Vec{L, T}}, re, im) where {L, T} = Vec{L, T}(ntuple(k -> isodd(k) ? T(re) : T(im), Val(L)))
# rot90 direction mask at width L: forward [-0,0,…], inverse [0,-0,…] (matches AvxRadix._rot90_fwd8/inv8).
@inline _rot_mask(::Type{Vec{L, T}}, forward::Bool) where {L, T} =
    forward ? _bcast_c(Vec{L, T}, -zero(T), zero(T)) : _bcast_c(Vec{L, T}, zero(T), -zero(T))

# ----------------------------------------------------------------------------------------------
# Plan: mixed-radix stage list [r0?,8,8,...], generalized digit-reversal, broadcast twiddles, tile.
# ----------------------------------------------------------------------------------------------
struct BatchPlan8{T, L}
    n::Int
    log2n::Int
    forward::Bool
    radices::Vector{Int}         # stage order (smallest len first); e.g. n=256 -> [4,8,8]
    rot::Vec{L, T}               # direction mask for the butterflies
    tw::Vector{Vec{L, T}}        # tw[k+1] = W_n^k broadcast to W=L÷2 lanes, full length n
    rev::Vector{Int}             # generalized digit reversal, 0-based
    GB::Int                      # groups (of W=L÷2 cols) per cache block; tile holds n*GB Vec{L}
    scratch::Vector{Vec{L, T}}   # tile: n*GB, reused per block
    sscratch::Vector{Complex{T}} # length n, reused per scalar-tail transform
end

function _choose_radices(log2n::Int)
    rad = Int[]
    m = log2n
    r = m % 3
    if r == 1
        push!(rad, 2); m -= 1
    elseif r == 2
        push!(rad, 4); m -= 2
    end
    for _ in 1:(m ÷ 3); push!(rad, 8); end
    return rad
end

# generalized mixed-radix digit reversal: first-applied radix -> most-significant reversed digit.
function _digitrev(i::Int, radices::Vector{Int})
    r = 0
    @inbounds for f in radices
        r = r * f + (i % f)
        i ÷= f
    end
    return r
end

# tile target ~4096 complex (≈ L1/2 for F64); GB = BLK/4 groups, tile = n·GB·4 complex. Clamp GB∈[1,8]
# (BLK∈[4,32]): matches the swept best (256²→16, 512²→8, 64³→32, 128²→16) without re-sweeping per plan.
_default_blk(n::Int) = clamp((4096 ÷ max(n, 1)) & ~3, 4, 32)

# Mixed-radix (BatchPlanMR) wants a BIGGER tile than the pow2 path. Profiling the length-240 strided pass
# (Task 6z, scratchpad/p240*.jl) showed its cost is ~2/3 arithmetic (radix-3/5 butterflies + twiddles) and
# only ~1/3 strided pack/scatter; the per-group FFT working set is L1-resident, so a larger GROUP COUNT per
# block (an ~L2-resident tile, not L1/2) amortizes the strided gather/scatter over more columns — a strict
# win on every measured 2^a·3^b·5^c·7^d shape (240²/224²/160³/112³, F64+F32) with no regression (the smaller
# default GB=4..8 was leaving 5-12% on the table). GB = BLK÷4 lands in the swept sweet spot (8..16) here:
# n=240→10, 224→10, 160→15, 112→16. Pow2 (BatchPlan8) keeps the L1/2 _default_blk above (its tuning differs).
_default_blk_mr(n::Int) = clamp((9600 ÷ max(n, 1)) & ~3, 32, 64)

function BatchPlan8(::Type{T}, n::Int; forward::Bool, BLK::Int = _default_blk(n)) where {T}
    @assert ispow2(n)
    L = _batch_lanes(T)
    log2n = trailing_zeros(n)
    radices = _choose_radices(log2n)
    tw = Vector{Vec{L, T}}(undef, n)
    for k in 0:(n - 1)
        cr, ci = AvxRadix.compute_twiddle(k, n, forward)
        tw[k + 1] = _bcast_c(Vec{L, T}, cr, ci)
    end
    # digit reversal must extract digits in REVERSE stage order: stage 1 (lowest len) processes the
    # least-significant reversed digit, so the radix-4/2 cleanup digit must be least significant.
    revrad = reverse(radices)
    rev = Int[_digitrev(r, revrad) for r in 0:(n - 1)]
    # GB groups of W=L÷2 cols. tile = n·GB Vec{L} = n·GB·64 bytes for BOTH F64 (Vec8) and F32 (Vec16),
    # so BLK÷4 (tuned at W=4) keeps the tile byte-size identical across element types.
    GB = max(1, BLK ÷ 4)
    rot = _rot_mask(Vec{L, T}, forward)
    BatchPlan8{T, L}(n, log2n, forward, radices, rot, tw, rev, GB,
                     Vector{Vec{L, T}}(undef, n * GB), Vector{Complex{T}}(undef, n))
end

# ----------------------------------------------------------------------------------------------
# Radix stages, operating on the n-length group sc[b0+1 .. b0+n] (b0 = group offset in the tile).
# Input already digit-reversed; output natural order. len = cumulative product up to & incl this stage.
# avx_column_butterfly8/4 return outputs in NATURAL bin order ⇒ the DIT stage stores straight back.
# ----------------------------------------------------------------------------------------------
@inline function _radix8_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}}, rot::Vec{L, T},
                                n::Int, len::Int) where {T, L}
    q = len >> 3
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < q
                js = j * step
                base = b0 + start + j
                v0 = sc[base + 1]
                v1 = AvxRadix.avx_mul_complex(sc[base + q + 1],  tw[js + 1])
                v2 = AvxRadix.avx_mul_complex(sc[base + 2q + 1], tw[2js + 1])
                v3 = AvxRadix.avx_mul_complex(sc[base + 3q + 1], tw[3js + 1])
                v4 = AvxRadix.avx_mul_complex(sc[base + 4q + 1], tw[4js + 1])
                v5 = AvxRadix.avx_mul_complex(sc[base + 5q + 1], tw[5js + 1])
                v6 = AvxRadix.avx_mul_complex(sc[base + 6q + 1], tw[6js + 1])
                v7 = AvxRadix.avx_mul_complex(sc[base + 7q + 1], tw[7js + 1])
                o = AvxRadix.avx_column_butterfly8(v0, v1, v2, v3, v4, v5, v6, v7, rot)
                sc[base + 1]    = o[1]; sc[base + q + 1]  = o[2]
                sc[base + 2q + 1] = o[3]; sc[base + 3q + 1] = o[4]
                sc[base + 4q + 1] = o[5]; sc[base + 5q + 1] = o[6]
                sc[base + 6q + 1] = o[7]; sc[base + 7q + 1] = o[8]
                j += 1
            end
            start += len
        end
    end
    return nothing
end

@inline function _radix4_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}}, rot::Vec{L, T},
                                n::Int, len::Int) where {T, L}
    q = len >> 2
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < q
                js = j * step
                base = b0 + start + j
                v0 = sc[base + 1]
                v1 = AvxRadix.avx_mul_complex(sc[base + q + 1],  tw[js + 1])
                v2 = AvxRadix.avx_mul_complex(sc[base + 2q + 1], tw[2js + 1])
                v3 = AvxRadix.avx_mul_complex(sc[base + 3q + 1], tw[3js + 1])
                o = AvxRadix.avx_column_butterfly4(v0, v1, v2, v3, rot)
                sc[base + 1] = o[1]; sc[base + q + 1] = o[2]
                sc[base + 2q + 1] = o[3]; sc[base + 3q + 1] = o[4]
                j += 1
            end
            start += len
        end
    end
    return nothing
end

@inline function _radix2_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}},
                                n::Int, len::Int) where {T, L}
    half = len >> 1
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < half
                base = b0 + start + j
                a = sc[base + 1]
                b = AvxRadix.avx_mul_complex(sc[base + half + 1], tw[j * step + 1])
                sc[base + 1]        = a + b
                sc[base + half + 1] = a - b
                j += 1
            end
            start += len
        end
    end
    return nothing
end

@inline function _fft_group!(sc::Vector{Vec{L, T}}, b0::Int, radices::Vector{Int},
                             tw::Vector{Vec{L, T}}, rot::Vec{L, T}, n::Int) where {T, L}
    len = 1
    @inbounds for f in radices
        len *= f
        if f == 8
            _radix8_stage!(sc, b0, tw, rot, n, len)
        elseif f == 4
            _radix4_stage!(sc, b0, tw, rot, n, len)
        else
            _radix2_stage!(sc, b0, tw, n, len)
        end
    end
    return nothing
end

# scalar radix-2 DIT for the inner%W remainder tail (one Complex transform).
@inline function _bitrev2(r::Int, log2n::Int)
    b = 0; x = r
    for _ in 1:log2n
        b = (b << 1) | (x & 1); x >>= 1
    end
    return b
end
@inline function _radix2_scalar!(ss::Vector{Complex{T}}, n::Int, forward::Bool) where {T}
    len = 2
    @inbounds while len <= n
        half = len >> 1
        ang = (forward ? -2 : 2) * T(pi) / len
        wlen = Complex{T}(cos(ang), sin(ang))
        start = 0
        while start < n
            w = one(Complex{T}); j = 0
            while j < half
                a = ss[start + j + 1]
                b = ss[start + j + half + 1] * w
                ss[start + j + 1]        = a + b
                ss[start + j + half + 1] = a - b
                w *= wlen; j += 1
            end
            start += len
        end
        len <<= 1
    end
    return nothing
end

# ==============================================================================================
# DEDICATED length-128 batched codelet (the one L2-resident/compute-bound small-square: 128²).
# n=128=2^7 is an ODD power of two ⇒ the generic radix-8 path needs a radix-2 cleanup pass (7≡1 mod 3)
# → 3 in-tile FFT passes + a separate strided pack and scatter = ~5 tile passes. FFTW's n1fv_128 loads
# once, computes in registers, stores once. We close most of that gap with a FUSED two-step (four-step
# CT) codelet: 128 = 8 × 16, computed as 8 length-16 DFTs (strided GATHER fused in) → twiddle →
# 16 length-8 DFTs (strided SCATTER fused in). The separate pack+scatter PASSES are eliminated — the
# tile is written ONCE (step1) and read ONCE (step2) instead of 6× — and only 2 butterfly layers run.
#   n = n1 + 8·n2  (n1∈0:7, n2∈0:15);  k = 16·k1 + k2  (k1∈0:7, k2∈0:15)
#   X[16k1+k2] = Σ_{n1} W_8^{n1 k1} · W_128^{n1 k2} · [ Σ_{n2} x[n1+8n2] W_16^{n2 k2} ]
# All twiddles are already in p.tw: W_128^{n1 k2}=tw[n1 k2+1] (n1 k2≤105<128), W_16^1=W_128^8=tw[9],
# W_16^3=W_128^24=tw[25]. Reuses the across-batch broadcast-twiddle butterflies (avx_column_butterfly8;
# _rbf16 = register radix-16, natural-order, verified). No digit reversal (the two-step indexing is direct).
# ----------------------------------------------------------------------------------------------

# Register-form radix-16 (across-batch: each lane = a different column, twiddles broadcast). Same 4×4
# algebra as AvxRadix.avx_column_butterfly16 but takes 16 row-VECTORS (not a strided load) and returns
# the 16 outputs in NATURAL bin order (verified bit-exact vs the size-16 DFT). tw1=W_16^1, tw3=W_16^3.
@inline function _rbf16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
                        tw1::Vec{L, T}, tw3::Vec{L, T}, rot::Vec{L, T}) where {L, T}
    bf4 = AvxRadix.avx_column_butterfly4; mul = AvxRadix.avx_mul_complex
    m1 = bf4(r1, r5, r9, r13, rot)
    m1 = (m1[1], mul(m1[2], tw1), AvxRadix.avx_bf8_tw1(m1[3], rot), mul(m1[4], tw3))
    m2 = bf4(r2, r6, r10, r14, rot)
    m2 = (m2[1], AvxRadix.avx_bf8_tw1(m2[2], rot), AvxRadix.avx_rotate90(m2[3], rot), AvxRadix.avx_bf8_tw3(m2[4], rot))
    m3 = bf4(r3, r7, r11, r15, rot)
    m3 = (m3[1], mul(m3[2], tw3), AvxRadix.avx_bf8_tw3(m3[3], rot), mul(m3[4], AvxRadix.avx_neg(tw1)))
    m0 = bf4(r0, r4, r8, r12, rot)
    c1 = bf4(m0[1], m1[1], m2[1], m3[1], rot)
    c2 = bf4(m0[2], m1[2], m2[2], m3[2], rot)
    c3 = bf4(m0[3], m1[3], m2[3], m3[3], rot)
    c4 = bf4(m0[4], m1[4], m2[4], m3[4], rot)
    (c1[1], c2[1], c3[1], c4[1], c1[2], c2[2], c3[2], c4[2],
     c1[3], c2[3], c3[3], c4[3], c1[4], c2[4], c3[4], c4[4])
end

# Register-form radix-15 (15 = 3×5 Cooley–Tukey, N1=3 inner / N2=5 outer): 15 row-VECTORS in, 15 outputs
# in NATURAL bin order. Reuses the across-batch column butterflies (avx_column_butterfly3/5). Verified
# bit-exact vs the size-15 DFT (scratchpad/p240f.jl; F64 ~2e-15, F32 ~6e-7, fwd+inv). w_j = W_15^j broadcast.
#   X[k1+3k2] = Σ_{n1} W_3^{n1 k1} W_15^{n2 k1} [Σ_{n1} x[5n1+n2] W_3^{n1 k1}]  — inner bf3 over n1, twiddle
#   W_15^{n2 k1}, outer bf5 over n2. (k1=0 column untwiddled; W_15^0=1.)
@inline function _rbf15(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14,
                        bf3::Vec{L, T}, bf5_0::Vec{L, T}, bf5_1::Vec{L, T},
                        w1::Vec{L, T}, w2::Vec{L, T}, w3::Vec{L, T}, w4::Vec{L, T},
                        w6::Vec{L, T}, w8::Vec{L, T}) where {L, T}
    bf3f = AvxRadix.avx_column_butterfly3; bf5f = AvxRadix.avx_column_butterfly5
    mul = AvxRadix.avx_mul_complex
    # step1: inner bf3 over n1 (one per n2): M_n2 = bf3(x[n2], x[5+n2], x[10+n2])
    M0 = bf3f(r0, r5, r10, bf3)
    M1 = bf3f(r1, r6, r11, bf3)
    M2 = bf3f(r2, r7, r12, bf3)
    M3 = bf3f(r3, r8, r13, bf3)
    M4 = bf3f(r4, r9, r14, bf3)
    # twiddle M_n2[k1] *= W_15^{n2 k1}, then outer bf5 over n2 (one per k1). k1=0 untwiddled.
    P0 = bf5f(M0[1], M1[1], M2[1], M3[1], M4[1], bf5_0, bf5_1)
    P1 = bf5f(M0[2], mul(M1[2], w1), mul(M2[2], w2), mul(M3[2], w3), mul(M4[2], w4), bf5_0, bf5_1)
    P2 = bf5f(M0[3], mul(M1[3], w2), mul(M2[3], w4), mul(M3[3], w6), mul(M4[3], w8), bf5_0, bf5_1)
    # X[k1+3k2] = P_{k1}[k2+1]
    (P0[1], P1[1], P2[1], P0[2], P1[2], P2[2], P0[3], P1[3], P2[3],
     P0[4], P1[4], P2[4], P0[5], P1[5], P2[5])
end

# Fused two-step length-128 transform of ONE W-column group. `cb` = 0-based complex index of the group's
# element 0 (the W columns are contiguous → _ldv loads all W at once); element e is at cb+e·inner.
# `sc` = the plan tile, used as the 128-entry intermediate B (laid out B[k2·8+n1], step2-contiguous).
# @generated ⇒ fully-unrolled straight-line code with LITERAL element/twiddle indices (CLAUDE rule #1);
# only cb and inner are runtime. Unit twiddles (n1==0 or k2==0) are skipped at codegen.
@generated function _bf128_group!(::Type{Vec{L, T}}, pt::Ptr{T}, cb::Int, inner::Int,
                                  sc::Vector{Vec{L, T}}, tw::Vector{Vec{L, T}}, rot::Vec{L, T}) where {L, T}
    VT = Vec{L, T}
    body = Expr(:block)
    push!(body.args, :(tw1 = tw[9]; tw3 = tw[25]))      # W_16^1, W_16^3
    # step1: 8 length-16 DFTs over n2, strided gather fused in, twiddle, store to tile B[k2·8+n1]
    for n1 in 0:7
        rs = Symbol[]
        for m in 0:15
            s = Symbol("r_", n1, "_", m); push!(rs, s)
            push!(body.args, :($s = _ldv($VT, pt, cb + $(n1 + 8m) * inner)))
        end
        osym = Symbol("o_", n1)
        push!(body.args, :($osym = _rbf16($(rs...), tw1, tw3, rot)))
        for k2 in 0:15
            val = :($osym[$(k2 + 1)])
            rhs = (n1 == 0 || k2 == 0) ? val : :(AvxRadix.avx_mul_complex($val, tw[$(n1 * k2 + 1)]))
            push!(body.args, :(sc[$(k2 * 8 + n1 + 1)] = $rhs))
        end
    end
    # step2: 16 length-8 DFTs over n1, read tile, scatter fused in to X[16k1+k2]
    for k2 in 0:15
        b = k2 * 8
        psym = Symbol("p_", k2)
        push!(body.args, :($psym = AvxRadix.avx_column_butterfly8(
            sc[$(b + 1)], sc[$(b + 2)], sc[$(b + 3)], sc[$(b + 4)],
            sc[$(b + 5)], sc[$(b + 6)], sc[$(b + 7)], sc[$(b + 8)], rot)))
        for k1 in 0:7
            push!(body.args, :(_stv!(pt, cb + $(16k1 + k2) * inner, $psym[$(k1 + 1)])))
        end
    end
    return Expr(:block, Expr(:macrocall, Symbol("@inbounds"), nothing, body), :(return nothing))
end

# Driver for the dedicated length-128 codelet: vector groups via the fused two-step, scalar tail unchanged.
function batched_fft128!(p::BatchPlan8{T, L}, pc::Ptr{Complex{T}}, off::Int,
                         inner::Int, outer::Int) where {T, L}
    sc = p.scratch; ss = p.sscratch; tw = p.tw; rot = p.rot; W = L >> 1
    pt = reinterpret(Ptr{T}, pc)
    nvtot = inner ÷ W
    @inbounds for o in 0:(outer - 1)
        base_o = off + o * inner * 128
        g = 0
        while g < nvtot
            _bf128_group!(Vec{L, T}, pt, base_o + g * W, inner, sc, tw, rot)
            g += 1
        end
        ic = nvtot * W                          # scalar tail: leftover inner % W columns, one transform each
        while ic < inner
            cb = base_o + ic
            for r in 0:127
                ss[_bitrev2(r, 7) + 1] = unsafe_load(pc, cb + r * inner + 1)
            end
            _radix2_scalar!(ss, 128, p.forward)
            for r in 0:127
                unsafe_store!(pc, ss[r + 1], cb + r * inner + 1)
            end
            ic += 1
        end
    end
    return nothing
end

# ----------------------------------------------------------------------------------------------
# Cache-blocked driver: FFT all inner*outer length-n transforms along the strided dim, NO transpose.
# Per outer slice, process inner W-groups in blocks of GB. Pack a block (row-major, into digit-rev
# positions) into the cache-resident tile, FFT each group out of the tile, scatter back. The strided
# gather/scatter is thus amortized over the whole block (TLB-friendly bursts).
# `pc` = pointer to the flat Complex{T} buffer; caller holds GC.@preserve. `off` = 0-based complex offset.
# n==128 routes to the dedicated fused codelet above — but ONLY at the wide F32 width (L=16, AVX-512
# Vec{16}=8 complex), where it's a decisive win (128² F32 0.95×→1.09×, measured +15% throughput). At
# L=8 (F64, or non-AVX512 F32) the fused single-group two-step is ~3% SLOWER than the GB-blocked generic
# radix path (less cross-group ILP for the narrow 4-complex vector) — so F64 keeps the proven generic
# path (128² F64 held at 0.76×, no regression). Only 128² has a strided length-128 dim, so this gate
# changes exactly the one F32 measurement and nothing else.
# ----------------------------------------------------------------------------------------------
function batched_fft8!(p::BatchPlan8{T, L}, pc::Ptr{Complex{T}}, off::Int,
                       inner::Int, outer::Int) where {T, L}
    (L == 16 && p.n == 128) && return batched_fft128!(p, pc, off, inner, outer)
    n = p.n; sc = p.scratch; ss = p.sscratch; rev = p.rev; tw = p.tw; rot = p.rot
    radices = p.radices; GB = p.GB; W = L >> 1
    pt = reinterpret(Ptr{T}, pc)
    nvtot = inner ÷ W                 # whole W-groups
    @inbounds for o in 0:(outer - 1)
        base_o = off + o * inner * n
        gstart = 0
        while gstart < nvtot
            gb = min(GB, nvtot - gstart)
            bcol = base_o + gstart * W
            # pack: row-major burst → tile, into digit-reversed row slots
            for r in 0:(n - 1)
                rr = rev[r + 1]
                rin = r * inner
                g = 0
                while g < gb
                    sc[g * n + rr + 1] = _ldv(Vec{L, T}, pt, bcol + g * W + rin)
                    g += 1
                end
            end
            # FFT each group out of the cache-resident tile
            g = 0
            while g < gb
                _fft_group!(sc, g * n, radices, tw, rot, n)
                g += 1
            end
            # scatter natural order back to the array
            for r in 0:(n - 1)
                rin = r * inner
                g = 0
                while g < gb
                    _stv!(pt, bcol + g * W + rin, sc[g * n + r + 1])
                    g += 1
                end
            end
            gstart += gb
        end
        # scalar tail: leftover inner % W columns, one transform each
        ic = nvtot * W
        while ic < inner
            cb = base_o + ic
            for r in 0:(n - 1)
                ss[_bitrev2(r, p.log2n) + 1] = unsafe_load(pc, cb + r * inner + 1)
            end
            _radix2_scalar!(ss, n, p.forward)
            for r in 0:(n - 1)
                unsafe_store!(pc, ss[r + 1], cb + r * inner + 1)
            end
            ic += 1
        end
    end
    return nothing
end

# ==============================================================================================
# MIXED-RADIX (2^a·3^b·5^c·7^d) batched-strided kernel. Generalizes BatchPlan8 above by adding batched
# RADIX-3/5/7 DIT stages that reuse AvxRadix.avx_column_butterfly3/5/7's tuned size-r algebra on r row-
# VECTORS (each W=4 complex across the contiguous inner batch) with scalar-broadcast twiddles.
# Radix-3 validated bit-exact in scratchpad/batched_nonpow2_proto.jl (Task 6h); radix-5/7 added Task 6w
# (5/7-smooth N-D shapes: 240²/224²/160³/112³). The radix-8/4/2 stages and the cache-blocked driver
# structure are shared verbatim with the pow2 kernel above; only the radix-3/5/7 stages + factorization
# (7s, 5s, 3s, then the pow2 cleanup) and the generic scalar tail are new.
# ==============================================================================================

# n = 2^a·3^b·5^c·7^d is "smooth" for this kernel ⇔ dividing out all factors of 7,5,3 leaves a power of
# two. (Pure pow2 is caught earlier by the BatchedDim route, so reaching this means c+b+d ≥ 1.)
function _is_smooth_2a3(n::Int)
    n < 1 && return false
    for p in (7, 5, 3); while n % p == 0; n ÷= p; end; end
    return ispow2(n)
end

# Factor n into stages: all 7s, then 5s, then 3s, then the pow2 cleanup (2 or 4 so the remaining log2 ≡ 0
# mod 3), then radix-8s. Any factor order is valid for self-sorting DIT as long as digitrev uses it reversed.
function _choose_radices_smooth(n::Int)
    rad = Int[]
    m = n
    while m % 7 == 0; push!(rad, 7); m ÷= 7; end
    while m % 5 == 0; push!(rad, 5); m ÷= 5; end
    while m % 3 == 0; push!(rad, 3); m ÷= 3; end
    @assert ispow2(m) "non-(2^a·3^b·5^c·7^d) size $n unsupported by the smooth batched kernel"
    log2m = trailing_zeros(m); r = log2m % 3
    if r == 1
        push!(rad, 2); log2m -= 1
    elseif r == 2
        push!(rad, 4); log2m -= 2
    end
    for _ in 1:(log2m ÷ 3); push!(rad, 8); end
    return rad
end

struct BatchPlanMR{T, L}
    n::Int
    forward::Bool
    radices::Vector{Int}         # stage order (smallest len first); e.g. n=384 -> [3,2,8,8]
    rot::Vec{L, T}               # direction mask for the radix-8/4 butterflies
    bf3::Vec{L, T}               # W_3 twiddle for avx_column_butterfly3 (carries the direction)
    bf5_0::Vec{L, T}             # W_5^1 / W_5^2 twiddles for avx_column_butterfly5
    bf5_1::Vec{L, T}
    bf7_0::Vec{L, T}             # W_7^1 / W_7^2 / W_7^3 twiddles for avx_column_butterfly7
    bf7_1::Vec{L, T}
    bf7_2::Vec{L, T}
    tw::Vector{Vec{L, T}}        # tw[k+1] = W_n^k broadcast to W=L÷2 lanes, full length n
    rev::Vector{Int}             # generalized digit reversal, 0-based
    GB::Int                      # groups (of W=L÷2 cols) per cache block; tile holds n*GB Vec{L}
    scratch::Vector{Vec{L, T}}   # tile: n*GB, reused per block
    sin::Vector{Complex{T}}      # length n, scalar-tail input gather
    sout::Vector{Complex{T}}     # length n, scalar-tail output
    stw::Vector{Complex{T}}      # length n, scalar twiddles W_n^k for the naive-DFT tail
end

function BatchPlanMR(::Type{T}, n::Int; forward::Bool, BLK::Int = _default_blk_mr(n)) where {T}
    @assert _is_smooth_2a3(n)
    L = _batch_lanes(T)
    radices = _choose_radices_smooth(n)
    tw = Vector{Vec{L, T}}(undef, n)
    stw = Vector{Complex{T}}(undef, n)
    for k in 0:(n - 1)
        cr, ci = AvxRadix.compute_twiddle(k, n, forward)
        tw[k + 1] = _bcast_c(Vec{L, T}, cr, ci)
        stw[k + 1] = Complex{T}(cr, ci)
    end
    revrad = reverse(radices)
    rev = Int[_digitrev(r, revrad) for r in 0:(n - 1)]
    GB = max(1, BLK ÷ 4)
    rot = _rot_mask(Vec{L, T}, forward)
    b3r, b3i = AvxRadix.compute_twiddle(1, 3, forward)
    bf3 = _bcast_c(Vec{L, T}, b3r, b3i)
    _bc(idx, len) = _bcast_c(Vec{L, T}, AvxRadix.compute_twiddle(idx, len, forward)...)
    BatchPlanMR{T, L}(n, forward, radices, rot, bf3,
                      _bc(1, 5), _bc(2, 5), _bc(1, 7), _bc(2, 7), _bc(3, 7),
                      tw, rev, GB,
                      Vector{Vec{L, T}}(undef, n * GB),
                      Vector{Complex{T}}(undef, n), Vector{Complex{T}}(undef, n), stw)
end

# Batched radix-3 DIT stage: size-3 DFT of 3 row-vectors with scalar-broadcast twiddles, reusing
# avx_column_butterfly3 (width-generic). Same shape as _radix8_stage! with factor 3.
@inline function _radix3_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}},
                                bf3::Vec{L, T}, n::Int, len::Int) where {T, L}
    q = len ÷ 3
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < q
                js = j * step
                base = b0 + start + j
                v0 = sc[base + 1]
                v1 = AvxRadix.avx_mul_complex(sc[base + q + 1],  tw[js + 1])
                v2 = AvxRadix.avx_mul_complex(sc[base + 2q + 1], tw[2js + 1])
                o = AvxRadix.avx_column_butterfly3(v0, v1, v2, bf3)
                sc[base + 1] = o[1]; sc[base + q + 1] = o[2]; sc[base + 2q + 1] = o[3]
                j += 1
            end
            start += len
        end
    end
    return nothing
end

# Batched radix-5 DIT stage: size-5 DFT of 5 row-vectors with scalar-broadcast twiddles, reusing
# avx_column_butterfly5 (width-generic). bf5_0/bf5_1 = W_5^1 / W_5^2 twiddles.
@inline function _radix5_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}},
                                bf5_0::Vec{L, T}, bf5_1::Vec{L, T}, n::Int, len::Int) where {T, L}
    q = len ÷ 5
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < q
                js = j * step
                base = b0 + start + j
                v0 = sc[base + 1]
                v1 = AvxRadix.avx_mul_complex(sc[base + q + 1],  tw[js + 1])
                v2 = AvxRadix.avx_mul_complex(sc[base + 2q + 1], tw[2js + 1])
                v3 = AvxRadix.avx_mul_complex(sc[base + 3q + 1], tw[3js + 1])
                v4 = AvxRadix.avx_mul_complex(sc[base + 4q + 1], tw[4js + 1])
                o = AvxRadix.avx_column_butterfly5(v0, v1, v2, v3, v4, bf5_0, bf5_1)
                sc[base + 1] = o[1]; sc[base + q + 1] = o[2]; sc[base + 2q + 1] = o[3]
                sc[base + 3q + 1] = o[4]; sc[base + 4q + 1] = o[5]
                j += 1
            end
            start += len
        end
    end
    return nothing
end

# Batched radix-7 DIT stage: size-7 DFT of 7 row-vectors, reusing avx_column_butterfly7 (width-generic).
@inline function _radix7_stage!(sc::Vector{Vec{L, T}}, b0::Int, tw::Vector{Vec{L, T}},
                                bf7_0::Vec{L, T}, bf7_1::Vec{L, T}, bf7_2::Vec{L, T},
                                n::Int, len::Int) where {T, L}
    q = len ÷ 7
    step = n ÷ len
    @inbounds begin
        start = 0
        while start < n
            j = 0
            while j < q
                js = j * step
                base = b0 + start + j
                v0 = sc[base + 1]
                v1 = AvxRadix.avx_mul_complex(sc[base + q + 1],  tw[js + 1])
                v2 = AvxRadix.avx_mul_complex(sc[base + 2q + 1], tw[2js + 1])
                v3 = AvxRadix.avx_mul_complex(sc[base + 3q + 1], tw[3js + 1])
                v4 = AvxRadix.avx_mul_complex(sc[base + 4q + 1], tw[4js + 1])
                v5 = AvxRadix.avx_mul_complex(sc[base + 5q + 1], tw[5js + 1])
                v6 = AvxRadix.avx_mul_complex(sc[base + 6q + 1], tw[6js + 1])
                o = AvxRadix.avx_column_butterfly7(v0, v1, v2, v3, v4, v5, v6, bf7_0, bf7_1, bf7_2)
                sc[base + 1] = o[1]; sc[base + q + 1] = o[2]; sc[base + 2q + 1] = o[3]; sc[base + 3q + 1] = o[4]
                sc[base + 4q + 1] = o[5]; sc[base + 5q + 1] = o[6]; sc[base + 6q + 1] = o[7]
                j += 1
            end
            start += len
        end
    end
    return nothing
end

@inline function _fft_group_mr!(sc::Vector{Vec{L, T}}, b0::Int, radices::Vector{Int},
                                tw::Vector{Vec{L, T}}, rot::Vec{L, T}, bf3::Vec{L, T},
                                bf5_0::Vec{L, T}, bf5_1::Vec{L, T},
                                bf7_0::Vec{L, T}, bf7_1::Vec{L, T}, bf7_2::Vec{L, T}, n::Int) where {T, L}
    len = 1
    @inbounds for f in radices
        len *= f
        if f == 8
            _radix8_stage!(sc, b0, tw, rot, n, len)
        elseif f == 4
            _radix4_stage!(sc, b0, tw, rot, n, len)
        elseif f == 3
            _radix3_stage!(sc, b0, tw, bf3, n, len)
        elseif f == 5
            _radix5_stage!(sc, b0, tw, bf5_0, bf5_1, n, len)
        elseif f == 7
            _radix7_stage!(sc, b0, tw, bf7_0, bf7_1, bf7_2, n, len)
        else
            _radix2_stage!(sc, b0, tw, n, len)
        end
    end
    return nothing
end

# ponytail: naive O(n²) DFT for the inner%W remainder tail (≤3 columns, rare). n is small (≤ a few
# hundred) so this is negligible vs the vector body; upgrade to a staged scalar kernel only if a
# workload is dominated by inner%4≠0 slabs.
@inline function _dft_scalar!(sout::Vector{Complex{T}}, sin::Vector{Complex{T}},
                              stw::Vector{Complex{T}}, n::Int) where {T}
    @inbounds for k in 0:(n - 1)
        acc = sin[1]
        for jj in 1:(n - 1)
            acc += sin[jj + 1] * stw[(jj * k) % n + 1]
        end
        sout[k + 1] = acc
    end
    return nothing
end

# ==============================================================================================
# DEDICATED length-240 batched codelet (240 = 2^4·3·5 — the only N-D shape whose [5,3,2,8] generic
# composition lands below the parity gate). The generic path makes ~6 tile passes (pack + 4 radix stages
# + scatter); FFTW computes n=240 as a fused TWO codelet steps (12×20). We do the same: a FUSED two-step
# Cooley–Tukey codelet, 240 = 16 × 15, computed as 16 length-15 DFTs (strided GATHER fused in) → twiddle
# → 15 length-16 DFTs (strided SCATTER fused in). The 4 intermediate tile passes are eliminated — the
# tile is written ONCE and read ONCE — and all radix-3/5/16 arithmetic happens in registers. Measured
# +74% (F64) / +85% (F32) on the strided pass vs [5,3,2,8] (scratchpad/p240f.jl), bit-exact.
#   n = n1 + 16·n2  (n1∈0:15, n2∈0:14);  k = 15·k1 + k2  (k1∈0:15, k2∈0:14)
#   X[15k1+k2] = Σ_{n1} W_16^{n1 k1} · W_240^{n1 k2} · [ Σ_{n2} x[n1+16n2] W_15^{n2 k2} ]
# All twiddles live in p.tw: W_240^{n1 k2}=tw[n1 k2+1] (≤210<240); W_15^j=W_240^{16j}=tw[16j+1];
# W_16^1=tw[16], W_16^3=tw[46]. Reuses _rbf15 / _rbf16 (across-batch register codelets) + p.bf3/bf5.
# @generated ⇒ fully-unrolled straight-line code with LITERAL element/twiddle indices (CLAUDE rule #1);
# only cb and inner are runtime. Unit twiddles (n1==0 or k2==0) are skipped at codegen.
@generated function _bf240_group!(::Type{Vec{L, T}}, pt::Ptr{T}, cb::Int, inner::Int,
                                  sc::Vector{Vec{L, T}}, tw::Vector{Vec{L, T}},
                                  bf3::Vec{L, T}, bf5_0::Vec{L, T}, bf5_1::Vec{L, T},
                                  rot::Vec{L, T}) where {L, T}
    VT = Vec{L, T}
    body = Expr(:block)
    # W_15^{1,2,3,4,6,8} and W_16^{1,3} as compile-time-fixed tw lookups
    push!(body.args, :(w1 = tw[17]; w2 = tw[33]; w3 = tw[49]; w4 = tw[65]; w6 = tw[97]; w8 = tw[129]))
    push!(body.args, :(tw1_16 = tw[16]; tw3_16 = tw[46]))
    # step1: 16 length-15 DFTs over n2 (strided gather), twiddle W_240^{n1 k2}, store to tile B[k2·16+n1]
    for n1 in 0:15
        rs = Symbol[]
        for m in 0:14
            s = Symbol("r_", n1, "_", m); push!(rs, s)
            push!(body.args, :($s = _ldv($VT, pt, cb + $(n1 + 16m) * inner)))
        end
        osym = Symbol("o_", n1)
        push!(body.args, :($osym = _rbf15($(rs...), bf3, bf5_0, bf5_1, w1, w2, w3, w4, w6, w8)))
        for k2 in 0:14
            val = :($osym[$(k2 + 1)])
            rhs = (n1 == 0 || k2 == 0) ? val : :(AvxRadix.avx_mul_complex($val, tw[$(n1 * k2 + 1)]))
            push!(body.args, :(sc[$(k2 * 16 + n1 + 1)] = $rhs))
        end
    end
    # step2: 15 length-16 DFTs over n1, read tile, scatter fused in to X[15k1+k2]
    for k2 in 0:14
        b = k2 * 16
        psym = Symbol("p_", k2)
        push!(body.args, :($psym = _rbf16(
            sc[$(b + 1)], sc[$(b + 2)], sc[$(b + 3)], sc[$(b + 4)], sc[$(b + 5)], sc[$(b + 6)],
            sc[$(b + 7)], sc[$(b + 8)], sc[$(b + 9)], sc[$(b + 10)], sc[$(b + 11)], sc[$(b + 12)],
            sc[$(b + 13)], sc[$(b + 14)], sc[$(b + 15)], sc[$(b + 16)], tw1_16, tw3_16, rot)))
        for k1 in 0:15
            push!(body.args, :(_stv!(pt, cb + $(15k1 + k2) * inner, $psym[$(k1 + 1)])))
        end
    end
    return Expr(:block, Expr(:macrocall, Symbol("@inbounds"), nothing, body), :(return nothing))
end

# Driver for the dedicated length-240 codelet: vector groups via the fused two-step, generic scalar tail.
function batched_fft240!(p::BatchPlanMR{T, L}, pc::Ptr{Complex{T}}, off::Int,
                         inner::Int, outer::Int) where {T, L}
    sc = p.scratch; tw = p.tw; rot = p.rot; bf3 = p.bf3; bf5_0 = p.bf5_0; bf5_1 = p.bf5_1
    sin = p.sin; sout = p.sout; stw = p.stw; W = L >> 1
    pt = reinterpret(Ptr{T}, pc)
    nvtot = inner ÷ W
    @inbounds for o in 0:(outer - 1)
        base_o = off + o * inner * 240
        g = 0
        while g < nvtot
            _bf240_group!(Vec{L, T}, pt, base_o + g * W, inner, sc, tw, bf3, bf5_0, bf5_1, rot)
            g += 1
        end
        ic = nvtot * W                          # scalar tail: leftover inner % W columns, one transform each
        while ic < inner
            cb = base_o + ic
            for r in 0:239
                sin[r + 1] = unsafe_load(pc, cb + r * inner + 1)
            end
            _dft_scalar!(sout, sin, stw, 240)
            for r in 0:239
                unsafe_store!(pc, sout[r + 1], cb + r * inner + 1)
            end
            ic += 1
        end
    end
    return nothing
end

# Cache-blocked driver — identical structure to batched_fft8! (generic radices + generic scalar tail).
function batched_fft_mr!(p::BatchPlanMR{T, L}, pc::Ptr{Complex{T}}, off::Int,
                         inner::Int, outer::Int) where {T, L}
    p.n == 240 && return batched_fft240!(p, pc, off, inner, outer)
    n = p.n; sc = p.scratch; rev = p.rev; tw = p.tw; rot = p.rot; bf3 = p.bf3
    bf5_0 = p.bf5_0; bf5_1 = p.bf5_1; bf7_0 = p.bf7_0; bf7_1 = p.bf7_1; bf7_2 = p.bf7_2
    radices = p.radices; GB = p.GB; W = L >> 1
    sin = p.sin; sout = p.sout; stw = p.stw
    pt = reinterpret(Ptr{T}, pc)
    nvtot = inner ÷ W
    @inbounds for o in 0:(outer - 1)
        base_o = off + o * inner * n
        gstart = 0
        while gstart < nvtot
            gb = min(GB, nvtot - gstart)
            bcol = base_o + gstart * W
            for r in 0:(n - 1)                  # pack: burst → tile, into digit-reversed row slots
                rr = rev[r + 1]; rin = r * inner
                g = 0
                while g < gb
                    sc[g * n + rr + 1] = _ldv(Vec{L, T}, pt, bcol + g * W + rin)
                    g += 1
                end
            end
            g = 0
            while g < gb
                _fft_group_mr!(sc, g * n, radices, tw, rot, bf3, bf5_0, bf5_1, bf7_0, bf7_1, bf7_2, n)
                g += 1
            end
            for r in 0:(n - 1)                  # scatter natural order back
                rin = r * inner
                g = 0
                while g < gb
                    _stv!(pt, bcol + g * W + rin, sc[g * n + r + 1])
                    g += 1
                end
            end
            gstart += gb
        end
        ic = nvtot * W                          # scalar tail: leftover inner % W columns
        while ic < inner
            cb = base_o + ic
            for r in 0:(n - 1)
                sin[r + 1] = unsafe_load(pc, cb + r * inner + 1)
            end
            _dft_scalar!(sout, sin, stw, n)
            for r in 0:(n - 1)
                unsafe_store!(pc, sout[r + 1], cb + r * inner + 1)
            end
            ic += 1
        end
    end
    return nothing
end
