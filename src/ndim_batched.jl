# Batched-strided pow2 FFT along a strided dim WITHOUT transpose: vectorize across the contiguous
# `inner` batch (W=4 complex per Vec{8,T}), running a mixed-radix-8 DIT on row-vectors with
# scalar-broadcast twiddles, out of a cache-resident pack-tile. This is the validated Task-6e kernel
# (scratchpad/batched8_proto.jl) cleaned for src/: hot path is pointer-based + GC.@preserve (no
# `reshape(x,:)`, which allocates an array header and would break the zero-alloc gate) and zero-alloc.
# Re-uses AvxRadix's tuned, width-generic column butterflies (avx_column_butterfly8/4) verbatim.

using SIMD: Vec, vload, vstore

# Pointer-based 4-complex load/store (i = 0-based complex index; complex = 2 T ⇒ byte stride 2·sizeof(T)).
@inline _ld8(pt::Ptr{T}, i::Int) where {T} = vload(Vec{8, T}, pt + i * 2 * sizeof(T))
@inline _st8!(pt::Ptr{T}, i::Int, v::Vec{8, T}) where {T} = vstore(v, pt + i * 2 * sizeof(T))

# ----------------------------------------------------------------------------------------------
# Plan: mixed-radix stage list [r0?,8,8,...], generalized digit-reversal, broadcast twiddles, tile.
# ----------------------------------------------------------------------------------------------
struct BatchPlan8{T}
    n::Int
    log2n::Int
    forward::Bool
    radices::Vector{Int}         # stage order (smallest len first); e.g. n=256 -> [4,8,8]
    rot::Vec{8, T}               # _rot90_fwd8/inv8(T) — direction mask for the butterflies
    tw::Vector{Vec{8, T}}        # tw[k+1] = W_n^k broadcast to W lanes, full length n
    rev::Vector{Int}             # generalized digit reversal, 0-based
    GB::Int                      # groups (of W=4 cols) per cache block; tile holds n*GB Vec{8}
    scratch::Vector{Vec{8, T}}   # tile: n*GB, reused per block
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

function BatchPlan8(::Type{T}, n::Int; forward::Bool, BLK::Int = _default_blk(n)) where {T}
    @assert ispow2(n)
    log2n = trailing_zeros(n)
    radices = _choose_radices(log2n)
    tw = Vector{Vec{8, T}}(undef, n)
    for k in 0:(n - 1)
        cr, ci = AvxRadix.compute_twiddle(k, n, forward)
        tw[k + 1] = AvxRadix.avx_broadcast_complex8(T, cr, ci)
    end
    # digit reversal must extract digits in REVERSE stage order: stage 1 (lowest len) processes the
    # least-significant reversed digit, so the radix-4/2 cleanup digit must be least significant.
    revrad = reverse(radices)
    rev = Int[_digitrev(r, revrad) for r in 0:(n - 1)]
    GB = max(1, BLK ÷ 4)
    rot = forward ? AvxRadix._rot90_fwd8(T) : AvxRadix._rot90_inv8(T)
    BatchPlan8{T}(n, log2n, forward, radices, rot, tw, rev, GB,
                  Vector{Vec{8, T}}(undef, n * GB), Vector{Complex{T}}(undef, n))
end

# ----------------------------------------------------------------------------------------------
# Radix stages, operating on the n-length group sc[b0+1 .. b0+n] (b0 = group offset in the tile).
# Input already digit-reversed; output natural order. len = cumulative product up to & incl this stage.
# avx_column_butterfly8/4 return outputs in NATURAL bin order ⇒ the DIT stage stores straight back.
# ----------------------------------------------------------------------------------------------
@inline function _radix8_stage!(sc::Vector{Vec{8, T}}, b0::Int, tw::Vector{Vec{8, T}}, rot::Vec{8, T},
                                n::Int, len::Int) where {T}
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

@inline function _radix4_stage!(sc::Vector{Vec{8, T}}, b0::Int, tw::Vector{Vec{8, T}}, rot::Vec{8, T},
                                n::Int, len::Int) where {T}
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

@inline function _radix2_stage!(sc::Vector{Vec{8, T}}, b0::Int, tw::Vector{Vec{8, T}},
                                n::Int, len::Int) where {T}
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

@inline function _fft_group!(sc::Vector{Vec{8, T}}, b0::Int, radices::Vector{Int},
                             tw::Vector{Vec{8, T}}, rot::Vec{8, T}, n::Int) where {T}
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

# ----------------------------------------------------------------------------------------------
# Cache-blocked driver: FFT all inner*outer length-n transforms along the strided dim, NO transpose.
# Per outer slice, process inner W-groups in blocks of GB. Pack a block (row-major, into digit-rev
# positions) into the cache-resident tile, FFT each group out of the tile, scatter back. The strided
# gather/scatter is thus amortized over the whole block (TLB-friendly bursts).
# `pc` = pointer to the flat Complex{T} buffer; caller holds GC.@preserve. `off` = 0-based complex offset.
# ----------------------------------------------------------------------------------------------
function batched_fft8!(p::BatchPlan8{T}, pc::Ptr{Complex{T}}, off::Int,
                       inner::Int, outer::Int) where {T}
    n = p.n; sc = p.scratch; ss = p.sscratch; rev = p.rev; tw = p.tw; rot = p.rot
    radices = p.radices; GB = p.GB; W = 4
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
                    sc[g * n + rr + 1] = _ld8(pt, bcol + g * W + rin)
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
                    _st8!(pt, bcol + g * W + rin, sc[g * n + r + 1])
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
# MIXED-RADIX (2^a·3^b) batched-strided kernel. Generalizes BatchPlan8 above by adding a batched
# RADIX-3 DIT stage that reuses AvxRadix.avx_column_butterfly3's tuned size-3 algebra on 3 row-
# VECTORS (each W=4 complex across the contiguous inner batch) with scalar-broadcast twiddles.
# Validated bit-exact in scratchpad/batched_nonpow2_proto.jl (Task 6h). The radix-8/4/2 stages and
# the cache-blocked driver structure are shared verbatim with the pow2 kernel above; only the
# radix-3 stage + factorization (3s first, then the pow2 cleanup) and the generic scalar tail are new.
# Structured so radix-5/7 stages can be slotted into _choose_radices_smooth / _fft_group_mr! later
# (YAGNI — only radix-3 built now, the class the planner routes here).
# ==============================================================================================

# n = 2^a·3^b is "smooth" for this kernel ⇔ dividing out all factors of 3 leaves a power of two.
# (Pure pow2 is caught earlier by the BatchedDim route, so reaching this means b ≥ 1.)
function _is_smooth_2a3(n::Int)
    n < 1 && return false
    while n % 3 == 0; n ÷= 3; end
    return ispow2(n)
end

# Factor n into stages: all 3s first, then the pow2 cleanup (2 or 4 so the remaining log2 ≡ 0 mod 3),
# then radix-8s. Any factor order is valid for self-sorting DIT as long as digitrev uses it reversed.
function _choose_radices_smooth(n::Int)
    rad = Int[]
    m = n
    while m % 3 == 0; push!(rad, 3); m ÷= 3; end
    @assert ispow2(m) "non-(2^a·3^b) size $n unsupported by the smooth batched kernel"
    log2m = trailing_zeros(m); r = log2m % 3
    if r == 1
        push!(rad, 2); log2m -= 1
    elseif r == 2
        push!(rad, 4); log2m -= 2
    end
    for _ in 1:(log2m ÷ 3); push!(rad, 8); end
    return rad
end

struct BatchPlanMR{T}
    n::Int
    forward::Bool
    radices::Vector{Int}         # stage order (smallest len first); e.g. n=384 -> [3,2,8,8]
    rot::Vec{8, T}               # direction mask for the radix-8/4 butterflies
    bf3::Vec{8, T}               # W_3 twiddle for avx_column_butterfly3 (carries the direction)
    tw::Vector{Vec{8, T}}        # tw[k+1] = W_n^k broadcast to W lanes, full length n
    rev::Vector{Int}             # generalized digit reversal, 0-based
    GB::Int                      # groups (of W=4 cols) per cache block; tile holds n*GB Vec{8}
    scratch::Vector{Vec{8, T}}   # tile: n*GB, reused per block
    sin::Vector{Complex{T}}      # length n, scalar-tail input gather
    sout::Vector{Complex{T}}     # length n, scalar-tail output
    stw::Vector{Complex{T}}      # length n, scalar twiddles W_n^k for the naive-DFT tail
end

function BatchPlanMR(::Type{T}, n::Int; forward::Bool, BLK::Int = _default_blk(n)) where {T}
    @assert _is_smooth_2a3(n)
    radices = _choose_radices_smooth(n)
    tw = Vector{Vec{8, T}}(undef, n)
    stw = Vector{Complex{T}}(undef, n)
    for k in 0:(n - 1)
        cr, ci = AvxRadix.compute_twiddle(k, n, forward)
        tw[k + 1] = AvxRadix.avx_broadcast_complex8(T, cr, ci)
        stw[k + 1] = Complex{T}(cr, ci)
    end
    revrad = reverse(radices)
    rev = Int[_digitrev(r, revrad) for r in 0:(n - 1)]
    GB = max(1, BLK ÷ 4)
    rot = forward ? AvxRadix._rot90_fwd8(T) : AvxRadix._rot90_inv8(T)
    bf3 = AvxRadix.avx_broadcast_twiddle8(T, 1, 3, forward)
    BatchPlanMR{T}(n, forward, radices, rot, bf3, tw, rev, GB,
                   Vector{Vec{8, T}}(undef, n * GB),
                   Vector{Complex{T}}(undef, n), Vector{Complex{T}}(undef, n), stw)
end

# Batched radix-3 DIT stage: size-3 DFT of 3 row-vectors with scalar-broadcast twiddles, reusing
# avx_column_butterfly3 (width-generic). Same shape as _radix8_stage! with factor 3.
@inline function _radix3_stage!(sc::Vector{Vec{8, T}}, b0::Int, tw::Vector{Vec{8, T}},
                                bf3::Vec{8, T}, n::Int, len::Int) where {T}
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

@inline function _fft_group_mr!(sc::Vector{Vec{8, T}}, b0::Int, radices::Vector{Int},
                                tw::Vector{Vec{8, T}}, rot::Vec{8, T}, bf3::Vec{8, T}, n::Int) where {T}
    len = 1
    @inbounds for f in radices
        len *= f
        if f == 8
            _radix8_stage!(sc, b0, tw, rot, n, len)
        elseif f == 4
            _radix4_stage!(sc, b0, tw, rot, n, len)
        elseif f == 3
            _radix3_stage!(sc, b0, tw, bf3, n, len)
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

# Cache-blocked driver — identical structure to batched_fft8! (generic radices + generic scalar tail).
function batched_fft_mr!(p::BatchPlanMR{T}, pc::Ptr{Complex{T}}, off::Int,
                         inner::Int, outer::Int) where {T}
    n = p.n; sc = p.scratch; rev = p.rev; tw = p.tw; rot = p.rot; bf3 = p.bf3
    radices = p.radices; GB = p.GB; W = 4
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
                    sc[g * n + rr + 1] = _ld8(pt, bcol + g * W + rin)
                    g += 1
                end
            end
            g = 0
            while g < gb
                _fft_group_mr!(sc, g * n, radices, tw, rot, bf3, n)
                g += 1
            end
            for r in 0:(n - 1)                  # scatter natural order back
                rin = r * inner
                g = 0
                while g < gb
                    _st8!(pt, bcol + g * W + rin, sc[g * n + r + 1])
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
