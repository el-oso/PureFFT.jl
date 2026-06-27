# Explicit-AVX Radix4 (variant :radix4avx) — the hand-vectorized AVX path.
#
# Same Radix4 structure (cache-blocked transpose → digit-reversed base butterflies → Butterfly4
# cross-passes), but the cross-pass is hand-vectorized with SIMD.jl: data stays AoS and the
# complex multiply uses the interleaved trick (duplicate twiddle re/im, swap data re/im, FMA +
# sign) — a standard AVX butterfly. Measured ~1.21× over the `@simd ivdep` autovec cross.
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
# 4-complex Float32 (256-bit ymm) — same 8-lane pattern as the Vec{8,Float64} method; used by the
# Float32 base-16/32 AVX codelet (the cross pass uses the wider Vec{16,Float32} method above).
@inline function _vcmul(d::Vec{8, Float32}, w::Vec{8, Float32})
    wr = shufflevector(w, Val((0, 0, 2, 2, 4, 4, 6, 6)))
    wi = shufflevector(w, Val((1, 1, 3, 3, 5, 5, 7, 7)))
    dd = shufflevector(d, Val((1, 0, 3, 2, 5, 4, 7, 6)))
    sgn = Vec{8, Float32}((-1.0f0, 1.0f0, -1.0f0, 1.0f0, -1.0f0, 1.0f0, -1.0f0, 1.0f0))
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

# ---- within-butterfly AVX base codelet (size-16 = 4×4, the Butterfly16) --------------
# 16 complex → 4 AVX-512 registers (4 complex each). DFT-4 down columns (across registers,
# shuffle-free) → twiddle → 4×4 register transpose (the only shuffles) → DFT-4 down columns
# again (shuffle-free) → contiguous store. Measured ~2.1× over the scalar codelet.

# Generic over the element type T (Float64 = 4-complex 512-bit zmm, Float32 = 4-complex 256-bit ymm);
# the 8-lane shuffle pattern is identical for both — same "genericize over Vec{8,T}" as avxport.
@inline function _dft4reg(V0::Vec{8, T}, V1, V2, V3, ::Val{S}) where {T, S}
    t0 = V0 + V2; t1 = V0 - V2; t2 = V1 + V3
    e = shufflevector(V1 - V3, Val((1, 0, 3, 2, 5, 4, 7, 6)))   # swap re/im → ·(∓i) next
    o, z = one(T), -one(T)
    t3 = S == -1 ? e * Vec{8, T}((o, z, o, z, o, z, o, z)) : e * Vec{8, T}((z, o, z, o, z, o, z, o))
    return (t0 + t2, t1 + t3, t0 - t2, t1 - t3)
end

# Pure 8-lane shuffles → element-type-agnostic (Vec{8,Float64} or Vec{8,Float32}).
@inline function _transpose4(C0::Vec{8}, C1, C2, C3)
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

# size-16 W16^{n1·k1} twiddle register for column k1 ∈ {1,2,3}; element type T (Float64 default).
@inline _tw16(::Type{T}, W, p) where {T} = Vec{8, T}((one(T), zero(T), T(real(W^p)), T(imag(W^p)), T(real(W^(2p))), T(imag(W^(2p))), T(real(W^(3p))), T(imag(W^(3p)))))
@inline _tw16(W, p) = _tw16(Float64, W, p)

# DFT-16 register core: 4 input registers (V_r = elem[4r:4r+3]) → 4 output registers in natural
# order. Two shuffle-free DFT-4s with a 4×4 register transpose between.
@inline function _dft16_regs(V0, V1, V2, V3, tw1, tw2, tw3, ::Val{S}) where {S}
    B0, B1, B2, B3 = _dft4reg(V0, V1, V2, V3, Val(S))
    C0 = B0; C1 = _vcmul(B1, tw1); C2 = _vcmul(B2, tw2); C3 = _vcmul(B3, tw3)
    D0, D1, D2, D3 = _transpose4(C0, C1, C2, C3)
    return _dft4reg(D0, D1, D2, D3, Val(S))
end

# size-32 W32^k combine twiddle register (k = o:o+3), used by the radix-2 step of DFT-32; element type T.
@inline _tw32(::Type{T}, W, o) where {T} = Vec{8, T}((T(real(W^o)), T(imag(W^o)), T(real(W^(o + 1))), T(imag(W^(o + 1))), T(real(W^(o + 2))), T(imag(W^(o + 2))), T(real(W^(o + 3))), T(imag(W^(o + 3)))))
@inline _tw32(W, o) = _tw32(Float64, W, o)

# DFT-32 register core (the Butterfly32): 8 input registers (V_r = elem[4r:4r+3]) → 8 output
# registers in natural order. Even/odd decimation → two DFT-16 → radix-2 combine with W32 twiddles.
@inline function _dft32_regs(V0, V1, V2, V3, V4, V5, V6, V7, tw1, tw2, tw3, u0, u1, u2, u3, ::Val{S}) where {S}
    de(a, b) = shufflevector(a, b, Val((0, 1, 4, 5, 8, 9, 12, 13)))   # evens of [a;b]
    od(a, b) = shufflevector(a, b, Val((2, 3, 6, 7, 10, 11, 14, 15)))  # odds of [a;b]
    Z0, Z1, Z2, Z3 = _dft16_regs(de(V0, V1), de(V2, V3), de(V4, V5), de(V6, V7), tw1, tw2, tw3, Val(S))
    Y0, Y1, Y2, Y3 = _dft16_regs(od(V0, V1), od(V2, V3), od(V4, V5), od(V6, V7), tw1, tw2, tw3, Val(S))
    T0 = _vcmul(Y0, u0); T1 = _vcmul(Y1, u1); T2 = _vcmul(Y2, u2); T3 = _vcmul(Y3, u3)
    return (Z0 + T0, Z1 + T1, Z2 + T2, Z3 + T3, Z0 - T0, Z1 - T1, Z2 - T2, Z3 - T3)
end

# Generic over T: 4-complex Vec{8,T} registers (F64 = 512-bit, F32 = 256-bit). Byte strides scale with
# sizeof(T): a Vec{8,T} = 4 complex = 8·sizeof(T) bytes; a size-16 column = 32·sizeof(T) bytes.
function _base16_avx!(dst::AbstractVector{Complex{T}}, src, width::Int, k::Int, ::Val{S}) where {T, S}
    W = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(T, W, 1); tw2 = _tw16(T, W, 2); tw3 = _tw16(T, W, 3)
    rb = 8 * sizeof(T); colb = 32 * sizeof(T)
    GC.@preserve dst src begin
        po = reinterpret(Ptr{T}, pointer(dst))
        pin = reinterpret(Ptr{T}, pointer(src))
        @inbounds for xi in 0:(width - 1)
            b = pin + _digitrev4(xi, k) * colb
            V0 = vload(Vec{8, T}, b); V1 = vload(Vec{8, T}, b + rb)
            V2 = vload(Vec{8, T}, b + 2rb); V3 = vload(Vec{8, T}, b + 3rb)
            Z0, Z1, Z2, Z3 = _dft16_regs(V0, V1, V2, V3, tw1, tw2, tw3, Val(S))
            o = po + xi * colb
            vstore(Z0, o); vstore(Z1, o + rb); vstore(Z2, o + 2rb); vstore(Z3, o + 3rb)
        end
    end
    return
end

# size-32 AVX codelet (the Butterfly32) = two DFT-16 (even/odd decimation) + radix-2 combine. Generic over T.
function _base32_avx!(dst::AbstractVector{Complex{T}}, src, width::Int, k::Int, ::Val{S}) where {T, S}
    W16 = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(T, W16, 1); tw2 = _tw16(T, W16, 2); tw3 = _tw16(T, W16, 3)
    W = S == -1 ? cispi(-2.0 / 32) : cispi(2.0 / 32)   # W32^k combine twiddles, k = 0:15
    u0 = _tw32(T, W, 0); u1 = _tw32(T, W, 4); u2 = _tw32(T, W, 8); u3 = _tw32(T, W, 12)
    rb = 8 * sizeof(T); colb = 64 * sizeof(T)
    GC.@preserve dst src begin
        po = reinterpret(Ptr{T}, pointer(dst))
        pin = reinterpret(Ptr{T}, pointer(src))
        @inbounds for xi in 0:(width - 1)
            b = pin + _digitrev4(xi, k) * colb
            V0 = vload(Vec{8, T}, b); V1 = vload(Vec{8, T}, b + rb)
            V2 = vload(Vec{8, T}, b + 2rb); V3 = vload(Vec{8, T}, b + 3rb)
            V4 = vload(Vec{8, T}, b + 4rb); V5 = vload(Vec{8, T}, b + 5rb)
            V6 = vload(Vec{8, T}, b + 6rb); V7 = vload(Vec{8, T}, b + 7rb)
            G0, G1, G2, G3, G4, G5, G6, G7 = _dft32_regs(V0, V1, V2, V3, V4, V5, V6, V7, tw1, tw2, tw3, u0, u1, u2, u3, Val(S))
            o = po + xi * colb
            vstore(G0, o); vstore(G1, o + rb); vstore(G2, o + 2rb); vstore(G3, o + 3rb)
            vstore(G4, o + 4rb); vstore(G5, o + 5rb); vstore(G6, o + 6rb); vstore(G7, o + 7rb)
        end
    end
    return
end

# NOTE: a size-64 within-register decomposition LOSES as a *base codelet* for large n (tried,
# regressed ~40→33 GF/s: it doubles the base pass — register spill + extra deinterleave shuffles —
# to remove the cheapest small-stride cross pass). But the *same* decomposition WINS as a
# *standalone whole-transform* kernel for n == 64 (`_fft64_avx!` below), because there it removes
# the entire scratch transpose — a ~70 ns fixed cost that dominates at small n (≈70 % of runtime
# at n=64). Different trade in each context; base-16/32 stay the sweet spot for the staged path.

# size-64 twiddle register: lanes hold c = 4g+l (l=0:3), value W64^{w·c} (a compile-time literal).
@inline _tw64(W, w, g) = Vec{8, Float64}((
    real(W^(w * (4g))), imag(W^(w * (4g))),
    real(W^(w * (4g + 1))), imag(W^(w * (4g + 1))),
    real(W^(w * (4g + 2))), imag(W^(w * (4g + 2))),
    real(W^(w * (4g + 3))), imag(W^(w * (4g + 3))),
))

# Standalone in-register size-64 FFT (Float64): n = 4×16 four-step done entirely in AVX registers.
# Load 64 complex contiguously → 4× register-transpose to gather the four stride-4 subsequences →
# 4× DFT-16 (+ W64 twiddles) → radix-4 (DFT-4) combine across the four → contiguous store. No
# scratch, no strided memory: ~38 GF/s vs ~19 for the general transpose path (beats FFTW's ~32).
function _fft64_avx!(x::AbstractVector{Complex{Float64}}, ::Val{S}) where {S}
    W16 = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(W16, 1); tw2 = _tw16(W16, 2); tw3 = _tw16(W16, 3)
    W64 = S == -1 ? cispi(-2.0 / 64) : cispi(2.0 / 64)
    GC.@preserve x begin
        p = reinterpret(Ptr{Float64}, pointer(x))
        @inbounds begin
            ld(i) = vload(Vec{8, Float64}, p + i * 64)
            r0 = ld(0); r1 = ld(1); r2 = ld(2); r3 = ld(3)
            r4 = ld(4); r5 = ld(5); r6 = ld(6); r7 = ld(7)
            r8 = ld(8); r9 = ld(9); r10 = ld(10); r11 = ld(11)
            r12 = ld(12); r13 = ld(13); r14 = ld(14); r15 = ld(15)
            # gather stride-4 subsequences: block w spans registers (u_w, v_w, w_w, q_w)
            u0, u1, u2, u3 = _transpose4(r0, r1, r2, r3)
            v0, v1, v2, v3 = _transpose4(r4, r5, r6, r7)
            x0, x1, x2, x3 = _transpose4(r8, r9, r10, r11)
            q0, q1, q2, q3 = _transpose4(r12, r13, r14, r15)
            # DFT-16 of each block, then twiddle output group g by W64^{w·c}
            s00, s01, s02, s03 = _dft16_regs(u0, v0, x0, q0, tw1, tw2, tw3, Val(S))
            s10, s11, s12, s13 = _dft16_regs(u1, v1, x1, q1, tw1, tw2, tw3, Val(S))
            s20, s21, s22, s23 = _dft16_regs(u2, v2, x2, q2, tw1, tw2, tw3, Val(S))
            s30, s31, s32, s33 = _dft16_regs(u3, v3, x3, q3, tw1, tw2, tw3, Val(S))
            s10 = _vcmul(s10, _tw64(W64, 1, 0)); s11 = _vcmul(s11, _tw64(W64, 1, 1)); s12 = _vcmul(s12, _tw64(W64, 1, 2)); s13 = _vcmul(s13, _tw64(W64, 1, 3))
            s20 = _vcmul(s20, _tw64(W64, 2, 0)); s21 = _vcmul(s21, _tw64(W64, 2, 1)); s22 = _vcmul(s22, _tw64(W64, 2, 2)); s23 = _vcmul(s23, _tw64(W64, 2, 3))
            s30 = _vcmul(s30, _tw64(W64, 3, 0)); s31 = _vcmul(s31, _tw64(W64, 3, 1)); s32 = _vcmul(s32, _tw64(W64, 3, 2)); s33 = _vcmul(s33, _tw64(W64, 3, 3))
            # radix-4 (DFT-4) combine across the four blocks, per group g
            h00, h10, h20, h30 = _dft4reg(s00, s10, s20, s30, Val(S))
            h01, h11, h21, h31 = _dft4reg(s01, s11, s21, s31, Val(S))
            h02, h12, h22, h32 = _dft4reg(s02, s12, s22, s32, Val(S))
            h03, h13, h23, h33 = _dft4reg(s03, s13, s23, s33, Val(S))
            # block m (16 complex = 256 bytes) stores contiguously, group g at +64·g
            st(m, a, b, c, d) = (o = p + m * 256; vstore(a, o); vstore(b, o + 64); vstore(c, o + 128); vstore(d, o + 192))
            st(0, h00, h01, h02, h03); st(1, h10, h11, h12, h13)
            st(2, h20, h21, h22, h23); st(3, h30, h31, h32, h33)
        end
    end
    return x
end

# Standalone in-register size-128 FFT (Float64): n = 4×32 four-step, same shape as `_fft64_avx!`
# with DFT-32 blocks. Uses all 32 AVX registers for the data, but explicit straight-line code
# schedules cleanly (closures/tuples boxed the vectors and were ~9× slower). ~44 GF/s vs ~25 for
# the transpose path (beats FFTW's ~26). Contiguous load → 8× register-transpose to gather the
# four stride-4 subsequences (8 registers each) → 4× DFT-32 (+ W128 twiddles) → radix-4 combine.
function _fft128_avx!(x::AbstractVector{Complex{Float64}}, ::Val{S}) where {S}
    W16 = S == -1 ? cispi(-2.0 / 16) : cispi(2.0 / 16)
    tw1 = _tw16(W16, 1); tw2 = _tw16(W16, 2); tw3 = _tw16(W16, 3)
    W32 = S == -1 ? cispi(-2.0 / 32) : cispi(2.0 / 32)
    u0 = _tw32(W32, 0); u1 = _tw32(W32, 4); u2 = _tw32(W32, 8); u3 = _tw32(W32, 12)
    W = S == -1 ? cispi(-2.0 / 128) : cispi(2.0 / 128)
    GC.@preserve x begin
        p = reinterpret(Ptr{Float64}, pointer(x))
        @inbounds begin
            ld(i) = vload(Vec{8, Float64}, p + i * 64)
            # gather stride-4 subsequences: 8× register-transpose, one per y-group of 4.
            # c{t}{w} = block w, elements y = 4t .. 4t+3.
            c00, c01, c02, c03 = _transpose4(ld(0), ld(1), ld(2), ld(3))
            c10, c11, c12, c13 = _transpose4(ld(4), ld(5), ld(6), ld(7))
            c20, c21, c22, c23 = _transpose4(ld(8), ld(9), ld(10), ld(11))
            c30, c31, c32, c33 = _transpose4(ld(12), ld(13), ld(14), ld(15))
            c40, c41, c42, c43 = _transpose4(ld(16), ld(17), ld(18), ld(19))
            c50, c51, c52, c53 = _transpose4(ld(20), ld(21), ld(22), ld(23))
            c60, c61, c62, c63 = _transpose4(ld(24), ld(25), ld(26), ld(27))
            c70, c71, c72, c73 = _transpose4(ld(28), ld(29), ld(30), ld(31))
            # DFT-32 of each block w = (c0w..c7w); twiddle output group g by W128^{w·c}
            a0, a1, a2, a3, a4, a5, a6, a7 = _dft32_regs(c00, c10, c20, c30, c40, c50, c60, c70, tw1, tw2, tw3, u0, u1, u2, u3, Val(S))
            b0, b1, b2, b3, b4, b5, b6, b7 = _dft32_regs(c01, c11, c21, c31, c41, c51, c61, c71, tw1, tw2, tw3, u0, u1, u2, u3, Val(S))
            d0, d1, d2, d3, d4, d5, d6, d7 = _dft32_regs(c02, c12, c22, c32, c42, c52, c62, c72, tw1, tw2, tw3, u0, u1, u2, u3, Val(S))
            e0, e1, e2, e3, e4, e5, e6, e7 = _dft32_regs(c03, c13, c23, c33, c43, c53, c63, c73, tw1, tw2, tw3, u0, u1, u2, u3, Val(S))
            b0 = _vcmul(b0, _tw64(W, 1, 0)); b1 = _vcmul(b1, _tw64(W, 1, 1)); b2 = _vcmul(b2, _tw64(W, 1, 2)); b3 = _vcmul(b3, _tw64(W, 1, 3))
            b4 = _vcmul(b4, _tw64(W, 1, 4)); b5 = _vcmul(b5, _tw64(W, 1, 5)); b6 = _vcmul(b6, _tw64(W, 1, 6)); b7 = _vcmul(b7, _tw64(W, 1, 7))
            d0 = _vcmul(d0, _tw64(W, 2, 0)); d1 = _vcmul(d1, _tw64(W, 2, 1)); d2 = _vcmul(d2, _tw64(W, 2, 2)); d3 = _vcmul(d3, _tw64(W, 2, 3))
            d4 = _vcmul(d4, _tw64(W, 2, 4)); d5 = _vcmul(d5, _tw64(W, 2, 5)); d6 = _vcmul(d6, _tw64(W, 2, 6)); d7 = _vcmul(d7, _tw64(W, 2, 7))
            e0 = _vcmul(e0, _tw64(W, 3, 0)); e1 = _vcmul(e1, _tw64(W, 3, 1)); e2 = _vcmul(e2, _tw64(W, 3, 2)); e3 = _vcmul(e3, _tw64(W, 3, 3))
            e4 = _vcmul(e4, _tw64(W, 3, 4)); e5 = _vcmul(e5, _tw64(W, 3, 5)); e6 = _vcmul(e6, _tw64(W, 3, 6)); e7 = _vcmul(e7, _tw64(W, 3, 7))
            # radix-4 (DFT-4) combine across the four blocks per group g; store block m contiguous
            # at x[32m + 4g ..] (32 complex = 512 bytes per block, group g at +64·g)
            st4(g, A, B, C, D) = begin
                h0, h1, h2, h3 = _dft4reg(A, B, C, D, Val(S))
                o = p + g * 64
                vstore(h0, o); vstore(h1, o + 512); vstore(h2, o + 1024); vstore(h3, o + 1536)
            end
            st4(0, a0, b0, d0, e0); st4(1, a1, b1, d1, e1); st4(2, a2, b2, d2, e2); st4(3, a3, b3, d3, e3)
            st4(4, a4, b4, d4, e4); st4(5, a5, b5, d5, e5); st4(6, a6, b6, d6, e6); st4(7, a7, b7, d7, e7)
        end
    end
    return x
end

# AVX base butterflies where supported (Float64/Float32, base 16/32); scalar codelets otherwise.
@inline function _base_butterflies_avx!(dst::AbstractVector{Complex{T}}, src, base, width, k, ::Val{S}) where {T, S}
    if (T === Float64 || T === Float32) && base == 16
        _base16_avx!(dst, src, width, k, Val(S))
    elseif (T === Float64 || T === Float32) && base == 32
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

# Vectorized transpose: base×width → width×base via a grid of 4×4 register transposes (`_transpose4`,
# generic over Vec{8,T}), replacing the scalar strided-store transpose. ~1.8× faster while the array
# fits L1; above that the scattered stores thrash and the scalar cache-blocked transpose wins, so the
# caller gates on size. Requires base and width both multiples of 4 (true for base 16/32, n≥256). The
# byte stride is 2·sizeof(T) per complex (16 for Float64, 8 for Float32).
function _radix4_transpose_avx!(dst::AbstractVector{Complex{T}}, x, base::Int, width::Int) where {T}
    es = 2 * sizeof(T)
    GC.@preserve dst x begin
        px = reinterpret(Ptr{T}, pointer(x))
        pd = reinterpret(Ptr{T}, pointer(dst))
        @inbounds for I in 0:(base ÷ 4 - 1), J in 0:(width ÷ 4 - 1)
            a, b, c, d = _transpose4(
                vload(Vec{8, T}, px + ((4I + 0) * width + 4J) * es),
                vload(Vec{8, T}, px + ((4I + 1) * width + 4J) * es),
                vload(Vec{8, T}, px + ((4I + 2) * width + 4J) * es),
                vload(Vec{8, T}, px + ((4I + 3) * width + 4J) * es),
            )
            o = pd + ((4J) * base + 4I) * es
            vstore(a, o); vstore(b, o + base * es); vstore(c, o + 2base * es); vstore(d, o + 3base * es)
        end
    end
    return dst
end

# Largest transform (complex elements) for which the vectorized transpose beats the scalar
# cache-blocked one — i.e. while the array still fits comfortably in L1. Derived from the real
# L1 size (see src/cpuinfo.jl); 1024 on Zen5.
const _VTRANSPOSE_MAX = _L1_TILE

# fuse two passes (radix-16) only while the fused 16L-element block still has good locality
# (16 strided streams within this many complex elements). Above it, single radix-4 wins on cache.
# Derived from L2 (see src/cpuinfo.jl); 8192 on Zen5 (best n≤16384 balance).
const _R16_FUSE_MAX = _L2_FUSE

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

Radix4 with the hand-vectorized (SIMD.jl) Butterfly4 cross-pass — the explicit-AVX
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
    # Small-n fast paths: skip the ~70 ns scratch transpose that otherwise dominates. n=16/32 have
    # width==1 (the transpose is an identity copy), so the base codelet runs straight on x in place
    # (it loads all lanes before storing) — works for any T. n=64/128 use the fused in-register
    # kernels, which are Float64-only for now.
    if n == 16 || n == 32
        _base_butterflies_avx!(x, x, p.base, 1, 0, p.inverse ? Val(1) : Val(-1))
        return x
    end
    if T === Float64
        n == 64 && return _fft64_avx!(x, p.inverse ? Val(1) : Val(-1))
        n == 128 && return _fft128_avx!(x, p.inverse ? Val(1) : Val(-1))
    end
    scr = p.scratch
    width = n ÷ p.base
    # Vectorized transpose while the array fits L1 (`_VTRANSPOSE_MAX` is in complex elements for
    # Float64; Float32 elements are half the size, so 2× as many fit — scale by 8÷sizeof(T)).
    if (T === Float64 || T === Float32) && n <= _VTRANSPOSE_MAX * (8 ÷ sizeof(T)) && p.base % 4 == 0 && width % 4 == 0
        _radix4_transpose_avx!(scr, x, p.base, width)
    else
        _radix4_transpose!(scr, x, p.base, width)
    end
    if p.inverse
        _base_butterflies_avx!(x, scr, p.base, width, p.k, Val(1))
        _radix4_cross_avx!(x, p.base, n, p.layers, Val(1))
    else
        _base_butterflies_avx!(x, scr, p.base, width, p.k, Val(-1))
        _radix4_cross_avx!(x, p.base, n, p.layers, Val(-1))
    end
    return x
end
