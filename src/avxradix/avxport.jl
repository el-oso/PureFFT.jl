# AVX2 complex-vector primitives. Vec{4,Float64} = one 256-bit register, lanes [re0, im0, re1, im1]
# (2 interleaved complex). Each function maps to the exact x86 AVX intrinsic, op-for-op (named inline).
# Width-generic later: these are written for the 4-lane (AVX2) vector; an 8-lane (AVX-512)
# instantiation slots in behind the same names in a later phase.

using SIMD: Vec, vload, vstore, shufflevector

const V4f = Vec{4, Float64}
const _NT4 = NTuple{4, VecElement{Float64}}

# ---- fmaddsub / fmsubadd: exact x86 FMA-addsub intrinsics via llvmcall (_mm256_fmaddsub_pd) ----
const _IR_MADDSUB = """
declare <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>)
define <4 x double> @entry(<4 x double> %a, <4 x double> %b, <4 x double> %c) #0 {
  %r = call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c)
  ret <4 x double> %r
}
attributes #0 = { alwaysinline }
"""
const _IR_SUBADD = replace(_IR_MADDSUB, "vfmaddsub" => "vfmsubadd")
@inline avx_fmaddsub(a::V4f, b::V4f, c::V4f) =
    Vec(Base.llvmcall((_IR_MADDSUB, "entry"), _NT4, Tuple{_NT4, _NT4, _NT4}, a.data, b.data, c.data))
@inline avx_fmsubadd(a::V4f, b::V4f, c::V4f) =
    Vec(Base.llvmcall((_IR_SUBADD, "entry"), _NT4, Tuple{_NT4, _NT4, _NT4}, a.data, b.data, c.data))

# ---- basic arithmetic (_mm256_{add,sub,mul,xor}_pd, _mm256_{fmadd,fnmadd}_pd) ----
@inline avx_add(a, b) = a + b                                      # generic over vector width
@inline avx_sub(a, b) = a - b
@inline avx_mul(a, b) = a * b
@inline avx_neg(a) = -a
@inline avx_fmadd(a, b, c) = muladd(a, b, c)                       # fmadd:  a*b + c
@inline avx_fnmadd(a, b, c) = muladd(-a, b, c)                     # fnmadd: -(a*b) + c
@inline function avx_xor(a::V4f, b::V4f)
    reinterpret(V4f, reinterpret(Vec{4, UInt64}, a) ⊻ reinterpret(Vec{4, UInt64}, b))
end

# ---- complex-layout shuffles (_mm256_permute_pd / _mm256_permute2f128_pd / _mm256_movedup_pd) ----
@inline avx_swap_complex(s::V4f) = shufflevector(s, Val((1, 0, 3, 2)))                   # permute_pd 0x05
@inline avx_dup_re(s::V4f) = shufflevector(s, Val((0, 0, 2, 2)))                         # movedup_pd
@inline avx_dup_im(s::V4f) = shufflevector(s, Val((1, 1, 3, 3)))                         # permute_pd 0x0F
@inline avx_duplicate_complex(s::V4f) = (avx_dup_re(s), avx_dup_im(s))
@inline avx_reverse_complex(s::V4f) = shufflevector(s, Val((2, 3, 0, 1)))                # permute2f128 0x01
@inline avx_unpacklo_complex(a::V4f, b::V4f) = shufflevector(a, b, Val((0, 1, 4, 5)))    # permute2f128 0x20
@inline avx_unpackhi_complex(a::V4f, b::V4f) = shufflevector(a, b, Val((2, 3, 6, 7)))    # permute2f128 0x31

# ---- complex multiply (mul_complex) ----
@inline function avx_mul_complex(left::V4f, right::V4f)
    lre = avx_dup_re(left)
    lim = avx_dup_im(left)
    rsh = avx_swap_complex(right)
    avx_fmaddsub(lre, right, avx_mul(lim, rsh))
end

# ---- rotation by 90° (make_rotation90 + rotate90) ----
# Forward  → broadcast Complex(-0.0, 0.0) → mask lanes [-0,0,-0,0]
# Inverse  → broadcast Complex(0.0, -0.0) → mask lanes [0,-0,0,-0]
const _ROT90_FWD = V4f((-0.0, 0.0, -0.0, 0.0))
const _ROT90_INV = V4f((0.0, -0.0, 0.0, -0.0))
@inline avx_rotate90(s, mask) = avx_swap_complex(avx_xor(s, mask))

# ---- dual-width: __m128d ≅ Vec{2,Float64} = 1 complex (AvxVector for __m128d) ----
const V2f = Vec{2, Float64}
@inline avx_swap_complex(s::V2f) = shufflevector(s, Val((1, 0)))                # _mm_permute_pd 0x1
@inline function avx_xor(a::V2f, b::V2f)
    reinterpret(V2f, reinterpret(Vec{2, UInt64}, a) ⊻ reinterpret(Vec{2, UInt64}, b))
end
@inline avx_broadcast_complex2(re::Float64, im::Float64) = V2f((re, im))
const _ROT90_FWD2 = V2f((-0.0, 0.0))
const _ROT90_INV2 = V2f((0.0, -0.0))
# column_butterfly2 (default impl): [a+b, a-b] — generic over width
@inline avx_butterfly2(a, b) = (a + b, a - b)

# lo/hi/merge (AvxVector256 lo/hi + AvxVector128 merge)
@inline avx_lo(v::V4f) = shufflevector(v, Val((0, 1)))     # low 128 bits  = complex 0
@inline avx_hi(v::V4f) = shufflevector(v, Val((2, 3)))     # high 128 bits = complex 1
@inline avx_merge(lo::V2f, hi::V2f) = shufflevector(lo, hi, Val((0, 1, 2, 3)))

# transpose_2x2_f64 (avx64_utils): [unpacklo, unpackhi]
@inline avx_transpose_2x2(a::V4f, b::V4f) = (avx_unpacklo_complex(a, b), avx_unpackhi_complex(a, b))
# transpose_3x3_f64 (avx64_utils): dual-width — col 0 packed as V2f, cols 1-2 as V4f
@inline function avx_transpose_3x3(a1::V2f, a2::V2f, a3::V2f, b1::V4f, b2::V4f, b3::V4f)
    out0 = (a1, avx_lo(b1), avx_hi(b1))
    lt = avx_transpose_2x2(b2, b3)
    out1 = (avx_merge(a2, a3), lt[1], lt[2])
    (out0, out1)
end

# partial (1-complex) load/store (load_partial1_complex / store_partial1_complex = _mm_loadu/storeu_pd)
@inline function avx_load_partial1(x::AbstractVector{Complex{Float64}}, i::Int)
    GC.@preserve x vload(V2f, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
@inline function avx_store_partial1!(x::AbstractVector{Complex{Float64}}, i::Int, v::V2f)
    GC.@preserve x vstore(v, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end

# ---- column butterflies (column_butterfly3 / column_butterfly6), width-generic ----
@inline function avx_column_butterfly3(r0, r1, r2, tw)
    mid1, mid2 = avx_butterfly2(r1, r2)
    output0 = avx_add(r0, mid1)
    twr, twi = avx_duplicate_complex(tw)
    mid1 = avx_fmadd(mid1, twr, r0)
    mid2_rot = avx_rotate90(mid2, _rot90_inv(mid2))      # width-generic (V4f or V2f)
    output1 = avx_fmadd(mid2_rot, twi, mid1)
    output2 = avx_fnmadd(mid2_rot, twi, mid1)
    (output0, output1, output2)
end
@inline function avx_column_butterfly6(r::NTuple{6}, tw)   # 3x2 good-thomas
    mid0 = avx_column_butterfly3(r[1], r[3], r[5], tw)     # rows 0,2,4
    mid1 = avx_column_butterfly3(r[4], r[6], r[2], tw)     # rows 3,5,1
    o0, o1 = avx_butterfly2(mid0[1], mid1[1])
    o2, o3 = avx_butterfly2(mid0[2], mid1[2])
    o4, o5 = avx_butterfly2(mid0[3], mid1[3])
    (o0, o3, o4, o1, o2, o5)
end

# ---- mixed-radix twiddle chunk (make_mixedradix_twiddle_chunk: 2 complex per Vec{4}) ----
@inline function avx_mixedradix_twiddle_chunk(x::Int, y::Int, len::Int, forward::Bool)
    t0r, t0i = compute_twiddle(y * x, len, forward)
    t1r, t1i = compute_twiddle(y * (x + 1), len, forward)
    V4f((t0r, t0i, t1r, t1i))
end
@inline function avx_broadcast_twiddle(index::Int, len::Int, forward::Bool)
    r, i = compute_twiddle(index, len, forward)
    avx_broadcast_complex(r, i)
end

# ---- Vec{2} (__m128d) duplicate/mul_complex/fmaddsub for the partial-column path ----
const _IR_MADDSUB2 = """
declare <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double>, <2 x double>, <2 x double>)
define <2 x double> @entry(<2 x double> %a, <2 x double> %b, <2 x double> %c) #0 {
  %r = call <2 x double> @llvm.x86.fma.vfmaddsub.pd(<2 x double> %a, <2 x double> %b, <2 x double> %c)
  ret <2 x double> %r
}
attributes #0 = { alwaysinline }
"""
const _NT2 = NTuple{2, VecElement{Float64}}
@inline avx_fmaddsub(a::V2f, b::V2f, c::V2f) =
    Vec(Base.llvmcall((_IR_MADDSUB2, "entry"), _NT2, Tuple{_NT2, _NT2, _NT2}, a.data, b.data, c.data))
@inline avx_dup_re(s::V2f) = shufflevector(s, Val((0, 0)))
@inline avx_dup_im(s::V2f) = shufflevector(s, Val((1, 1)))
@inline avx_duplicate_complex(s::V2f) = (avx_dup_re(s), avx_dup_im(s))
@inline function avx_mul_complex(left::V2f, right::V2f)
    avx_fmaddsub(avx_dup_re(left), right, avx_mul(avx_dup_im(left), avx_swap_complex(right)))
end

# width-generic inverse-rotation mask (make_rotation90(Inverse) dispatches on vector type)
@inline _rot90_inv(::V4f) = _ROT90_INV
@inline _rot90_inv(::V2f) = _ROT90_INV2

# ---- column_butterfly5, width-generic ----
@inline function avx_column_butterfly5(r1, r2, r3, r4, r5, tw0, tw1)
    sum1, diff4 = avx_butterfly2(r2, r5)
    sum2, diff3 = avx_butterfly2(r3, r4)
    rot = _rot90_inv(r1)
    rotated4 = avx_rotate90(diff4, rot)
    rotated3 = avx_rotate90(diff3, rot)
    output0 = avx_add(r1, avx_add(sum1, sum2))
    t0r, t0i = avx_duplicate_complex(tw0)
    t1r, t1i = avx_duplicate_complex(tw1)
    twiddled1_mid = avx_fmadd(t0r, sum1, r1)
    twiddled2_mid = avx_fmadd(t1r, sum1, r1)
    twiddled3_mid = avx_mul(t1i, rotated4)
    twiddled4_mid = avx_mul(t0i, rotated4)
    twiddled1 = avx_fmadd(t1r, sum2, twiddled1_mid)
    twiddled2 = avx_fmadd(t0r, sum2, twiddled2_mid)
    twiddled3 = avx_fnmadd(t0i, rotated3, twiddled3_mid)
    twiddled4 = avx_fmadd(t1i, rotated3, twiddled4_mid)
    output1, output4 = avx_butterfly2(twiddled1, twiddled4)
    output2, output3 = avx_butterfly2(twiddled2, twiddled3)
    (output0, output1, output2, output3, output4)
end

# column_butterfly4: rotation = make_rotation90(FFT direction) — _ROT90_FWD for forward.
@inline function avx_column_butterfly4(r1, r2, r3, r4, rotation)
    mid0, mid2 = avx_butterfly2(r1, r3)
    mid1, mid3 = avx_butterfly2(r2, r4)
    mid3_rot = avx_rotate90(mid3, rotation)
    o0, o1 = avx_butterfly2(mid0, mid1)
    o2, o3 = avx_butterfly2(mid2, mid3_rot)
    (o0, o2, o1, o3)
end
# transpose4_packed (__m256d): unpack_complex pairs → [lo01, lo23, hi01, hi23]
@inline function avx_transpose4_packed(r1::V4f, r2::V4f, r3::V4f, r4::V4f)
    (avx_unpacklo_complex(r1, r2), avx_unpacklo_complex(r3, r4),
     avx_unpackhi_complex(r1, r2), avx_unpackhi_complex(r3, r4))
end

# ---- column_butterfly8 (4x2 mixed radix) + transpose8 + bf8 twiddle helpers ----
const _HALF_ROOT2 = V4f((sqrt(0.5), sqrt(0.5), sqrt(0.5), sqrt(0.5)))
@inline _half_root2(::V4f) = _HALF_ROOT2                                                # width-dispatched (V8f added below)
@inline avx_bf8_tw1(x, rot) = avx_mul(_half_root2(x), avx_add(avx_rotate90(x, rot), x))   # apply_butterfly8_twiddle1
@inline avx_bf8_tw3(x, rot) = avx_mul(_half_root2(x), avx_sub(avx_rotate90(x, rot), x))   # apply_butterfly8_twiddle3
@inline function avx_column_butterfly8(r1, r2, r3, r4, r5, r6, r7, r8, rot)
    m0 = avx_column_butterfly4(r1, r3, r5, r7, rot)
    m1 = avx_column_butterfly4(r2, r4, r6, r8, rot)
    m1_2 = avx_bf8_tw1(m1[2], rot)
    m1_3 = avx_rotate90(m1[3], rot)
    m1_4 = avx_bf8_tw3(m1[4], rot)
    o0, o1 = avx_butterfly2(m0[1], m1[1])
    o2, o3 = avx_butterfly2(m0[2], m1_2)
    o4, o5 = avx_butterfly2(m0[3], m1_3)
    o6, o7 = avx_butterfly2(m0[4], m1_4)
    (o0, o2, o4, o6, o1, o3, o5, o7)
end
# transpose8_packed (__m256d): two transpose4 + reorder
@inline function avx_transpose8_packed(r1, r2, r3, r4, r5, r6, r7, r8)
    a = avx_transpose4_packed(r1, r2, r3, r4)
    b = avx_transpose4_packed(r5, r6, r7, r8)
    (a[1], a[2], b[1], b[2], a[3], a[4], b[3], b[4])
end
# transpose6_packed (__m256d): unpack_complex pairs → (u0,u2,u4,u1,u3,u5)
@inline avx_transpose6_packed(r0, r1, r2, r3, r4, r5) =
    (avx_unpacklo_complex(r0, r1), avx_unpacklo_complex(r2, r3), avx_unpacklo_complex(r4, r5),
     avx_unpackhi_complex(r0, r1), avx_unpackhi_complex(r2, r3), avx_unpackhi_complex(r4, r5))
# transpose9_packed (__m256d): permute2f128 pattern (0x30 = (r8_lo, r0_hi))
@inline avx_transpose9_packed(r0, r1, r2, r3, r4, r5, r6, r7, r8) =
    (avx_unpacklo_complex(r0, r1), avx_unpacklo_complex(r2, r3), avx_unpacklo_complex(r4, r5), avx_unpacklo_complex(r6, r7),
     shufflevector(r8, r0, Val((0, 1, 6, 7))),
     avx_unpackhi_complex(r1, r2), avx_unpackhi_complex(r3, r4), avx_unpackhi_complex(r5, r6), avx_unpackhi_complex(r7, r8))
# transpose12_packed (__m256d): three transpose4 + reorder
@inline function avx_transpose12_packed(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11)
    a = avx_transpose4_packed(r0, r1, r2, r3); b = avx_transpose4_packed(r4, r5, r6, r7); c = avx_transpose4_packed(r8, r9, r10, r11)
    (a[1], a[2], b[1], b[2], c[1], c[2], a[3], a[4], b[3], b[4], c[3], c[4])
end
# column_butterfly12 (4x3 good-thomas — NO twiddles between); bf3 = bf3 twiddle, rot = rotation
@inline function avx_column_butterfly12(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, bf3, rot)
    mid0 = avx_column_butterfly4(r0, r3, r6, r9, rot)
    mid1 = avx_column_butterfly4(r4, r7, r10, r1, rot)
    mid2 = avx_column_butterfly4(r8, r11, r2, r5, rot)
    o0 = avx_column_butterfly3(mid0[1], mid1[1], mid2[1], bf3)
    o1 = avx_column_butterfly3(mid0[2], mid1[2], mid2[2], bf3)
    o2 = avx_column_butterfly3(mid0[3], mid1[3], mid2[3], bf3)
    o3 = avx_column_butterfly3(mid0[4], mid1[4], mid2[4], bf3)
    (o0[1], o1[2], o2[3], o3[1], o0[2], o1[3], o2[1], o3[2], o0[3], o1[1], o2[2], o3[3])
end
# column_butterfly9 (3x3 mixed radix); tw1=W9^1, tw2=W9^2, tw3=W9^4 (broadcast complex); bf3=bf3 twiddle
@inline function avx_column_butterfly9(r0, r1, r2, r3, r4, r5, r6, r7, r8, tw1, tw2, tw3, bf3)
    mid0 = avx_column_butterfly3(r0, r3, r6, bf3)
    mid1 = avx_column_butterfly3(r1, r4, r7, bf3)
    mid2 = avx_column_butterfly3(r2, r5, r8, bf3)
    m1_2 = avx_mul_complex(tw1, mid1[2]); m1_3 = avx_mul_complex(tw2, mid1[3])
    m2_2 = avx_mul_complex(tw2, mid2[2]); m2_3 = avx_mul_complex(tw3, mid2[3])
    o0 = avx_column_butterfly3(mid0[1], mid1[1], mid2[1], bf3)
    o1 = avx_column_butterfly3(mid0[2], m1_2, m2_2, bf3)
    o2 = avx_column_butterfly3(mid0[3], m1_3, m2_3, bf3)
    (o0[1], o1[1], o2[1], o0[2], o1[2], o2[2], o0[3], o1[3], o2[3])
end

# transpose5_packed (__m256d): note _mm256_blend_pd(a,b,0x03) = lanes 0,1 from b, 2,3 from a
@inline _blend03(a::V4f, b::V4f) = shufflevector(a, b, Val((4, 5, 2, 3)))
# transpose3_packed (__m256d): unpacklo(r0,r1), blend_pd(r0,r2,0x03), unpackhi(r1,r2)
@inline avx_transpose3_packed(r0::V4f, r1::V4f, r2::V4f) =
    (avx_unpacklo_complex(r0, r1), _blend03(r0, r2), avx_unpackhi_complex(r1, r2))
@inline function avx_transpose5_packed(r1::V4f, r2::V4f, r3::V4f, r4::V4f, r5::V4f)
    (avx_unpacklo_complex(r1, r2), avx_unpacklo_complex(r3, r4), _blend03(r1, r5),
     avx_unpackhi_complex(r2, r3), avx_unpackhi_complex(r4, r5))
end

# ---- broadcast / twiddles (broadcast_complex_elements, twiddles::compute_twiddle) ----
@inline avx_broadcast_complex(re::Float64, im::Float64) = V4f((re, im, re, im))
# compute_twiddle: angle = -2π·index/len (Forward); Inverse negates the angle (conjugate).
@inline function compute_twiddle(index::Int, len::Int, forward::Bool)
    angle = (forward ? -2.0 : 2.0) * pi * index / len
    (cos(angle), sin(angle))
end

# ---- load / store 2 complex (load_complex / store_complex = loadu_pd / storeu_pd) ----
@inline function avx_load_complex(x::AbstractVector{Complex{Float64}}, i::Int)  # i: 0-based complex index
    GC.@preserve x vload(V4f, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
@inline function avx_store_complex!(x::AbstractVector{Complex{Float64}}, i::Int, v::V4f)
    GC.@preserve x vstore(v, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end

# ============================================================================================
# AVX-512 width: Vec{8,Float64} = one 512-bit register = 4 interleaved complex (CPV=4). The
# column butterflies (cb3/4/5/6/8/9/12) are width-generic — they only need these width-dispatched
# primitives + constants (no rewrite). The differentiator vs RustFFT (AVX2-only). complex multiply
# uses muladd+sign (cf. radix4_avx _vcmul), not a 512-bit fmaddsub intrinsic.
# ============================================================================================
const V8f = Vec{8, Float64}
@inline avx_swap_complex(s::V8f) = shufflevector(s, Val((1, 0, 3, 2, 5, 4, 7, 6)))
@inline avx_dup_re(s::V8f) = shufflevector(s, Val((0, 0, 2, 2, 4, 4, 6, 6)))
@inline avx_dup_im(s::V8f) = shufflevector(s, Val((1, 1, 3, 3, 5, 5, 7, 7)))
@inline avx_duplicate_complex(s::V8f) = (avx_dup_re(s), avx_dup_im(s))
@inline avx_xor(a::V8f, b::V8f) = reinterpret(V8f, reinterpret(Vec{8, UInt64}, a) ⊻ reinterpret(Vec{8, UInt64}, b))
const _ROT90_FWD8 = V8f((-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0))
const _ROT90_INV8 = V8f((0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0))
@inline _rot90_inv(::V8f) = _ROT90_INV8
const _HALF_ROOT2_8 = V8f(ntuple(_ -> sqrt(0.5), Val(8)))
@inline _half_root2(::V8f) = _HALF_ROOT2_8
const _SGN8 = V8f((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
@inline avx_mul_complex(left::V8f, right::V8f) =
    muladd(avx_dup_re(left), right, _SGN8 * (avx_dup_im(left) * avx_swap_complex(right)))
@inline avx_broadcast_complex8(re::Float64, im::Float64) = V8f((re, im, re, im, re, im, re, im))
@inline function avx_broadcast_twiddle8(index::Int, len::Int, forward::Bool)
    r, i = compute_twiddle(index, len, forward); avx_broadcast_complex8(r, i)
end
# mixedradix twiddle chunk, 4 complex per V8f (vs 2 per V4f): columns x..x+3
@inline function avx_mixedradix_twiddle_chunk8(x::Int, y::Int, len::Int, forward::Bool)
    t0r, t0i = compute_twiddle(y * x, len, forward); t1r, t1i = compute_twiddle(y * (x + 1), len, forward)
    t2r, t2i = compute_twiddle(y * (x + 2), len, forward); t3r, t3i = compute_twiddle(y * (x + 3), len, forward)
    V8f((t0r, t0i, t1r, t1i, t2r, t2i, t3r, t3i))
end
@inline function avx_load_complex8(x::AbstractVector{Complex{Float64}}, i::Int)
    GC.@preserve x vload(V8f, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
@inline function avx_store_complex8!(x::AbstractVector{Complex{Float64}}, i::Int, v::V8f)
    GC.@preserve x vstore(v, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
