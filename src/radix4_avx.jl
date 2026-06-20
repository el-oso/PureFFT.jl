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

# AVX base butterflies where supported (Float64, base 16); scalar codelets otherwise.
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

function _radix4_cross_avx!(out::AbstractVector{Complex{T}}, base::Int, n::Int, layers, ::Val{S}) where {T, S}
    W = _avx_width(T)
    M = 2W
    VT = Vec{M, T}
    es = 2 * sizeof(T)        # bytes per complex
    L = base
    p = 1
    @inbounds while L < n
        cur = 4L
        tw = layers[p]
        GC.@preserve out tw begin
            po = reinterpret(Ptr{T}, pointer(out))
            pt = reinterpret(Ptr{T}, pointer(tw))
            for blk in 0:cur:(n - 1)
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
                while j < L          # scalar remainder (small early passes, L < W)
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
        L = cur
        p += 1
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
