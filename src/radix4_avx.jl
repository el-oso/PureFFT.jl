# Explicit-AVX Radix4 (variant :radix4avx) — the rustfft-AVX-matching path.
#
# Same Radix4 structure (cache-blocked transpose → digit-reversed base butterflies → Butterfly4
# cross-passes), but the cross-pass is hand-vectorized with SIMD.jl: data stays AoS and the
# complex multiply uses the interleaved trick (duplicate twiddle re/im, swap data re/im, FMA +
# sign) — exactly rustfft's AVX butterfly. Measured ~1.21× over the `@simd ivdep` autovec cross.
#
# SIMD.jl is a thin VecElement/llvmcall wrapper; re-added (it was dropped when it LOST to autovec
# on the memory-bound radix-2, but it WINS on the compute-dense radix-4 cross).

using SIMD: Vec, vload, vstore, shufflevector

# W complex per vector → a 64-byte (AVX-512) register on this Zen 5 host. M = 2W interleaved lanes.
@inline _avx_width(::Type{T}) where {T} = 32 ÷ sizeof(T)   # F64→4 (M=8), F32→8 (M=16)

# d * w for interleaved [re,im,…] vectors (the AVX complex multiply): duplicate w's re/im,
# swap d's re/im, FMA with a sign vector. Explicit @inline per width (a `@generated` version
# was textually identical but did NOT inline → ~2.8× slower).
@inline function _vcmul(d::Vec{8, Float64}, w::Vec{8, Float64})
    wr = shufflevector(w, Val((0, 0, 2, 2, 4, 4, 6, 6)))
    wi = shufflevector(w, Val((1, 1, 3, 3, 5, 5, 7, 7)))
    dd = shufflevector(d, Val((1, 0, 3, 2, 5, 4, 7, 6)))
    sgn = Vec{8, Float64}((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    return muladd(d, wr, sgn * (dd * wi))
end
@inline function _vcmul(d::Vec{16, Float32}, w::Vec{16, Float32})
    wr = shufflevector(w, Val((0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14)))
    wi = shufflevector(w, Val((1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15)))
    dd = shufflevector(d, Val((1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)))
    sgn = Vec{16, Float32}(ntuple(k -> isodd(k) ? -1.0f0 : 1.0f0, Val(16)))
    return muladd(d, wr, sgn * (dd * wi))
end

# multiply interleaved vector by ∓i (radix-4 twist): forward S=-1 → ·(-i); inverse S=+1 → ·(+i)
@inline function _vtwist(z::Vec{8, Float64}, ::Val{S}) where {S}
    zz = shufflevector(z, Val((1, 0, 3, 2, 5, 4, 7, 6)))
    sgn = S == -1 ? Vec{8, Float64}((1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)) :
        Vec{8, Float64}((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    return zz * sgn
end
@inline function _vtwist(z::Vec{16, Float32}, ::Val{S}) where {S}
    zz = shufflevector(z, Val((1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14)))
    sgn = Vec{16, Float32}(ntuple(k -> (S == -1 ? (isodd(k) ? 1.0f0 : -1.0f0) : (isodd(k) ? -1.0f0 : 1.0f0)), Val(16)))
    return zz * sgn
end

# ---- within-butterfly AVX base codelet (size-16 = 4×4, rustfft's Butterfly16) --------------
# 16 complex → 4 AVX-512 registers (4 complex each). DFT-4 down columns (across registers,
# shuffle-free) → twiddle → 4×4 register transpose (the only shuffles) → DFT-4 down columns
# again (shuffle-free) → contiguous store. Measured ~2.1× over the scalar codelet.

@inline function _dft4reg(V0::Vec{8, Float64}, V1, V2, V3, ::Val{S}) where {S}
    t0 = V0 + V2; t1 = V0 - V2; t2 = V1 + V3
    e = shufflevector(V1 - V3, Val((1, 0, 3, 2, 5, 4, 7, 6)))   # swap re/im → ·(∓i) next
    t3 = S == -1 ? e * Vec{8, Float64}((1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0)) :
        e * Vec{8, Float64}((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
    return (t0 + t2, t1 + t3, t0 - t2, t1 - t3)
end

@inline function _transpose4(C0::Vec{8, Float64}, C1, C2, C3)
    P0 = shufflevector(C0, C1, Val((0, 1, 8, 9, 2, 3, 10, 11)))
    P1 = shufflevector(C0, C1, Val((4, 5, 12, 13, 6, 7, 14, 15)))
    P2 = shufflevector(C2, C3, Val((0, 1, 8, 9, 2, 3, 10, 11)))
    P3 = shufflevector(C2, C3, Val((4, 5, 12, 13, 6, 7, 14, 15)))
    return (
        shufflevector(P0, P2, Val((0, 1, 2, 3, 8, 9, 10, 11))),
        shufflevector(P0, P2, Val((4, 5, 6, 7, 12, 13, 14, 15))),
        shufflevector(P1, P3, Val((0, 1, 2, 3, 8, 9, 10, 11))),
        shufflevector(P1, P3, Val((4, 5, 6, 7, 12, 13, 14, 15))),
    )
end

# size-16 W16^{n1·k1} twiddle register for column k1 ∈ {1,2,3}
@inline _tw16(W, p) = Vec{8, Float64}((1.0, 0.0, real(W^p), imag(W^p), real(W^(2p)), imag(W^(2p)), real(W^(3p)), imag(W^(3p))))

# DFT-16 register core: 4 input registers (V_r = elem[4r:4r+3]) → 4 output registers in natural
# order. Two shuffle-free DFT-4s with a 4×4 register transpose between.
@inline function _dft16_regs(V0, V1, V2, V3, tw1, tw2, tw3, ::Val{S}) where {S}
    B0, B1, B2, B3 = _dft4reg(V0, V1, V2, V3, Val(S))
    C0 = B0; C1 = _vcmul(B1, tw1); C2 = _vcmul(B2, tw2); C3 = _vcmul(B3, tw3)
    D0, D1, D2, D3 = _transpose4(C0, C1, C2, C3)
    return _dft4reg(D0, D1, D2, D3, Val(S))
end

function _base16_avx!(dst, src, width::Int, k::Int, ::Val{S}) where {S}
    W = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(W, 1); tw2 = _tw16(W, 2); tw3 = _tw16(W, 3)
    GC.@preserve dst src begin
        po = reinterpret(Ptr{Float64}, pointer(dst))
        pin = reinterpret(Ptr{Float64}, pointer(src))
        @inbounds for xi in 0:(width - 1)
            b = pin + _digitrev4(xi, k) * 256
            V0 = vload(Vec{8, Float64}, b); V1 = vload(Vec{8, Float64}, b + 64)
            V2 = vload(Vec{8, Float64}, b + 128); V3 = vload(Vec{8, Float64}, b + 192)
            Z0, Z1, Z2, Z3 = _dft16_regs(V0, V1, V2, V3, tw1, tw2, tw3, Val(S))
            o = po + xi * 256
            vstore(Z0, o); vstore(Z1, o + 64); vstore(Z2, o + 128); vstore(Z3, o + 192)
        end
    end
    return
end

# size-32 AVX codelet (rustfft's Butterfly32) = two DFT-16 (even/odd decimation) + radix-2 combine.
function _base32_avx!(dst, src, width::Int, k::Int, ::Val{S}) where {S}
    W16 = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(W16, 1); tw2 = _tw16(W16, 2); tw3 = _tw16(W16, 3)
    W = S == -1 ? cispi(-2.0 / 32) : cispi(2.0 / 32)   # W32^k combine twiddles, k = 0:15
    u(o) = Vec{8, Float64}((real(W^o), imag(W^o), real(W^(o + 1)), imag(W^(o + 1)), real(W^(o + 2)), imag(W^(o + 2)), real(W^(o + 3)), imag(W^(o + 3))))
    u0 = u(0); u1 = u(4); u2 = u(8); u3 = u(12)
    de(a, b) = shufflevector(a, b, Val((0, 1, 4, 5, 8, 9, 12, 13)))   # evens of [a;b]
    od(a, b) = shufflevector(a, b, Val((2, 3, 6, 7, 10, 11, 14, 15)))  # odds of [a;b]
    GC.@preserve dst src begin
        po = reinterpret(Ptr{Float64}, pointer(dst))
        pin = reinterpret(Ptr{Float64}, pointer(src))
        @inbounds for xi in 0:(width - 1)
            b = pin + _digitrev4(xi, k) * 512
            V0 = vload(Vec{8, Float64}, b); V1 = vload(Vec{8, Float64}, b + 64)
            V2 = vload(Vec{8, Float64}, b + 128); V3 = vload(Vec{8, Float64}, b + 192)
            V4 = vload(Vec{8, Float64}, b + 256); V5 = vload(Vec{8, Float64}, b + 320)
            V6 = vload(Vec{8, Float64}, b + 384); V7 = vload(Vec{8, Float64}, b + 448)
            E0 = de(V0, V1); E1 = de(V2, V3); E2 = de(V4, V5); E3 = de(V6, V7)
            O0 = od(V0, V1); O1 = od(V2, V3); O2 = od(V4, V5); O3 = od(V6, V7)
            Z0, Z1, Z2, Z3 = _dft16_regs(E0, E1, E2, E3, tw1, tw2, tw3, Val(S))
            Y0, Y1, Y2, Y3 = _dft16_regs(O0, O1, O2, O3, tw1, tw2, tw3, Val(S))
            T0 = _vcmul(Y0, u0); T1 = _vcmul(Y1, u1); T2 = _vcmul(Y2, u2); T3 = _vcmul(Y3, u3)
            o = po + xi * 512
            vstore(Z0 + T0, o); vstore(Z1 + T1, o + 64); vstore(Z2 + T2, o + 128); vstore(Z3 + T3, o + 192)
            vstore(Z0 - T0, o + 256); vstore(Z1 - T1, o + 320); vstore(Z2 - T2, o + 384); vstore(Z3 - T3, o + 448)
        end
    end
    return
end

# NOTE (negative result, kept for the record): a size-64 within-register codelet (base 64 =
# radix-4 of 16: 4-way stride-4 deinterleave → 4×DFT-16 → Butterfly4 combine) was implemented to
# trade one cross pass for a bigger base. It REGRESSED ~40→33 GFLOP/s: the base pass roughly
# doubled (16 live registers spill + 32 deinterleave shuffles per leaf) while the cross pass it
# removed is the *cheapest* small-stride one. Reverted. The size-32 codelet is likewise slower
# than base-16, so base-16 (even n) / base-32 (odd n) remain the sweet spot.

# AVX base butterflies where supported (Float64, base 16/32); scalar codelets otherwise.
@inline function _base_butterflies_avx!(dst::AbstractVector{Complex{T}}, src, base, width, k, ::Val{S}) where {T, S}
    if T === Float64 && base == 16
        _base16_avx!(dst, src, width, k, Val(S))
    elseif T === Float64 && base == 32
        _base32_avx!(dst, src, width, k, Val(S))
    else
        _base_butterflies_dr!(dst, src, base, width, k, Val(S))
    end
    return
end

# one radix-4 cross pass (stride L), vectorized over j with a scalar remainder.
@inline function _radix4_pass_avx!(po::Ptr{T}, tw, L::Int, n::Int, out, ::Val{S}) where {T, S}
    W = _avx_width(T)
    VT = Vec{2W, T}
    es = 2 * sizeof(T)
    cur = 4L
    GC.@preserve tw begin
        pt = reinterpret(Ptr{T}, pointer(tw))
        @inbounds for blk in 0:cur:(n - 1)
            j = 0
            while j + W <= L
                o = blk + j
                a = vload(VT, po + o * es)
                b = _vcmul(vload(VT, po + (o + L) * es), vload(VT, pt + j * es))
                c = _vcmul(vload(VT, po + (o + 2L) * es), vload(VT, pt + (L + j) * es))
                d = _vcmul(vload(VT, po + (o + 3L) * es), vload(VT, pt + (2L + j) * es))
                t0 = a + c; t1 = a - c; t2 = b + d
                t3 = _vtwist(b - d, Val(S))
                vstore(t0 + t2, po + o * es)
                vstore(t1 + t3, po + (o + L) * es)
                vstore(t0 - t2, po + (o + 2L) * es)
                vstore(t1 - t3, po + (o + 3L) * es)
                j += W
            end
            while j < L
                o = blk + j
                a = out[o + 1]
                b = _cmul(out[o + L + 1], tw[j + 1])
                c = _cmul(out[o + 2L + 1], tw[L + j + 1])
                d = _cmul(out[o + 3L + 1], tw[2L + j + 1])
                t0 = a + c; t1 = a - c; t2 = b + d
                t3 = _twist(b - d, Val(S))
                out[o + 1] = t0 + t2; out[o + L + 1] = t1 + t3
                out[o + 2L + 1] = t0 - t2; out[o + 3L + 1] = t1 - t3
                j += 1
            end
        end
    end
    return
end

# fused radix-16 cross pass: does the stride-L AND stride-4L radix-4 passes in ONE read-modify-
# write sweep (halves the bandwidth-bound full-array passes). Gathers the 16 elements at
# blk + j + m·L (m=0:15), runs the two radix-4 levels keeping intermediates in registers, scatters
# back. twA = layer for stride L (idx j @ cur=4L); twB = layer for stride 4L (idx j+qpos·L @ 16L).
# Requires L ≥ W (vectorizes cleanly over j). Verified vs sequential two-pass.
@inline function _radix16_pass_avx!(po::Ptr{T}, twA, twB, L::Int, n::Int, ::Val{S}) where {T, S}
    W = _avx_width(T)
    VT = Vec{2W, T}
    es = 2 * sizeof(T)
    blkstep = 16L
    GC.@preserve twA twB begin
        pa = reinterpret(Ptr{T}, pointer(twA))
        pb = reinterpret(Ptr{T}, pointer(twB))
        # pass-1 twiddles depend only on j (idx j, L+j, 2L+j); load per j.
        @inbounds for blk in 0:blkstep:(n - 1)
            j = 0
            while j + W <= L
                base = blk + j
                wa1 = vload(VT, pa + j * es); wa2 = vload(VT, pa + (L + j) * es); wa3 = vload(VT, pa + (2L + j) * es)
                # pass1 on the 4 quartets (m mod 4 = pos within quartet), gq = m÷4
                # quartet gq elements at base + (4gq + s)·L, s=0:3
                r0a = vload(VT, po + (base + 0L) * es)
                r1a = _vcmul(vload(VT, po + (base + 1L) * es), wa1)
                r2a = _vcmul(vload(VT, po + (base + 2L) * es), wa2)
                r3a = _vcmul(vload(VT, po + (base + 3L) * es), wa3)
                q00, q01, q02, q03 = _bf4(r0a, r1a, r2a, r3a, Val(S))
                r0b = vload(VT, po + (base + 4L) * es)
                r1b = _vcmul(vload(VT, po + (base + 5L) * es), wa1)
                r2b = _vcmul(vload(VT, po + (base + 6L) * es), wa2)
                r3b = _vcmul(vload(VT, po + (base + 7L) * es), wa3)
                q10, q11, q12, q13 = _bf4(r0b, r1b, r2b, r3b, Val(S))
                r0c = vload(VT, po + (base + 8L) * es)
                r1c = _vcmul(vload(VT, po + (base + 9L) * es), wa1)
                r2c = _vcmul(vload(VT, po + (base + 10L) * es), wa2)
                r3c = _vcmul(vload(VT, po + (base + 11L) * es), wa3)
                q20, q21, q22, q23 = _bf4(r0c, r1c, r2c, r3c, Val(S))
                r0d = vload(VT, po + (base + 12L) * es)
                r1d = _vcmul(vload(VT, po + (base + 13L) * es), wa1)
                r2d = _vcmul(vload(VT, po + (base + 14L) * es), wa2)
                r3d = _vcmul(vload(VT, po + (base + 15L) * es), wa3)
                q30, q31, q32, q33 = _bf4(r0d, r1d, r2d, r3d, Val(S))
                # pass2 across gq (0:3) for each qpos s, twiddle idx j2 = j + s·L
                _r16p2!(po, pb, base, 0, j, L, es, q00, q10, q20, q30, VT, Val(S))
                _r16p2!(po, pb, base, 1, j, L, es, q01, q11, q21, q31, VT, Val(S))
                _r16p2!(po, pb, base, 2, j, L, es, q02, q12, q22, q32, VT, Val(S))
                _r16p2!(po, pb, base, 3, j, L, es, q03, q13, q23, q33, VT, Val(S))
                j += W
            end
        end
    end
    return
end

# pass-2 of the fused radix-16: butterfly across the 4 quartet-results for qpos s, scatter to
# global offsets base + (s + 4·gq)·L for gq=0:3. j2 = j + s·L → twiddle idx into twB.
@inline function _r16p2!(
        po::Ptr{T}, pb::Ptr{T}, base::Int, s::Int, j::Int, L::Int, es::Int,
        a, b0, c0, d0, ::Type{VT}, ::Val{S}
    ) where {T, VT, S}
    j2 = j + s * L
    @inbounds begin
        b = _vcmul(b0, vload(VT, pb + j2 * es))
        c = _vcmul(c0, vload(VT, pb + (4L + j2) * es))
        d = _vcmul(d0, vload(VT, pb + (8L + j2) * es))
        y0, y1, y2, y3 = _bf4(a, b, c, d, Val(S))
        vstore(y0, po + (base + (s + 0) * L) * es)
        vstore(y1, po + (base + (s + 4) * L) * es)
        vstore(y2, po + (base + (s + 8) * L) * es)
        vstore(y3, po + (base + (s + 12) * L) * es)
    end
    return
end

# radix-4 butterfly on 4 vectors → 4 vectors (forward/inverse via twist sign).
@inline function _bf4(a::VT, b::VT, c::VT, d::VT, ::Val{S}) where {VT, S}
    t0 = a + c; t1 = a - c; t2 = b + d
    t3 = _vtwist(b - d, Val(S))
    return (t0 + t2, t1 + t3, t0 - t2, t1 - t3)
end

# fuse two passes (radix-16) only while the fused 16L-element block still has good locality
# (16 strided streams within this many complex elements). Above it, single radix-4 wins on cache.
const _R16_FUSE_MAX = 1 << 13   # complex elements; tuned on Zen5 (8192 best n≤16384 balance)

function _radix4_cross_avx!(out::AbstractVector{Complex{T}}, base::Int, n::Int, layers, ::Val{S}) where {T, S}
    W = _avx_width(T)
    npass = length(layers)
    GC.@preserve out begin
        po = reinterpret(Ptr{T}, pointer(out))
        L = base
        p = 1
        # fuse pairs of passes (radix-16) when L ≥ W, two passes remain, and the fused block is
        # cache-local; else single radix-4. Halves the bandwidth-bound full-array sweeps.
        while p <= npass
            if p + 1 <= npass && L >= W && 16L <= _R16_FUSE_MAX
                _radix16_pass_avx!(po, layers[p], layers[p + 1], L, n, Val(S))
                L *= 16
                p += 2
            else
                _radix4_pass_avx!(po, layers[p], L, n, out, Val(S))
                L *= 4
                p += 1
            end
        end
    end
    return
end

"""
    Radix4AvxPlan{T} <: AbstractFFTPlan{T}

Radix4 with the hand-vectorized (SIMD.jl) Butterfly4 cross-pass — the rustfft-AVX-matching
variant (`:radix4avx`). Power-of-two `n`; allocation-free.
"""
struct Radix4AvxPlan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    base::Int
    k::Int
    layers::Vector{Vector{Complex{T}}}
    scratch::Vector{Complex{T}}
end

function Radix4AvxPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ispow2(n) || throw(ArgumentError(":radix4avx supports power-of-two sizes only; got n=$n"))
    nl = trailing_zeros(Int(n))
    base = _radix4_base(nl)
    k = (nl - trailing_zeros(base)) ÷ 2
    layers = _radix4_layers(Complex{T}, Int(n), base; inverse)
    return Radix4AvxPlan{T}(Int(n), inverse, base, k, layers, Vector{Complex{T}}(undef, Int(n)))
end

plan_length(p::Radix4AvxPlan)::Int = p.n
plan_inverse(p::Radix4AvxPlan)::Bool = p.inverse

function apply_unnormalized!(p::Radix4AvxPlan{T}, x::AbstractVector) where {T}
    n = p.n
    n <= 1 && return x
    scr = p.scratch
    width = n ÷ p.base
    _radix4_transpose!(scr, x, p.base, width)
    if p.inverse
        _base_butterflies_avx!(x, scr, p.base, width, p.k, Val(1))
        _radix4_cross_avx!(x, p.base, n, p.layers, Val(1))
    else
        _base_butterflies_avx!(x, scr, p.base, width, p.k, Val(-1))
        _radix4_cross_avx!(x, p.base, n, p.layers, Val(-1))
    end
    return x
end
