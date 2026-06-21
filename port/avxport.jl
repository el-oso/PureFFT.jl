# Faithful mechanical port of RustFFT 6.4.1 src/avx/avx_vector.rs `AvxVector for __m256d`.
#
# __m256d  ≅  Vec{4,Float64}, lanes laid out [re0, im0, re1, im1] (2 interleaved complex).
# Every function mirrors the exact Rust intrinsic, op-for-op (see the inline `// rust:` notes).
# Width-generic later: these are written for the 4-lane (AVX2) vector; an 8-lane (AVX-512)
# instantiation slots in behind the same names in a later phase.
#
# Standalone for now (developed/verified outside src/; moved into PureFFT after the parity gate).

using SIMD: Vec, vload, vstore, shufflevector

const V4f = Vec{4, Float64}
const _NT4 = NTuple{4, VecElement{Float64}}

# ---- fmaddsub / fmsubadd: exact x86 FMA-addsub intrinsics via llvmcall (rust: _mm256_fmaddsub_pd) ----
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

# ---- basic arithmetic (rust: _mm256_{add,sub,mul,xor}_pd, _mm256_{fmadd,fnmadd}_pd) ----
@inline avx_add(a::V4f, b::V4f) = a + b
@inline avx_sub(a::V4f, b::V4f) = a - b
@inline avx_mul(a::V4f, b::V4f) = a * b
@inline avx_neg(a::V4f) = -a
@inline avx_fmadd(a::V4f, b::V4f, c::V4f) = muladd(a, b, c)        # rust fmadd:  a*b + c
@inline avx_fnmadd(a::V4f, b::V4f, c::V4f) = muladd(-a, b, c)      # rust fnmadd: -(a*b) + c
@inline function avx_xor(a::V4f, b::V4f)
    reinterpret(V4f, reinterpret(Vec{4, UInt64}, a) ⊻ reinterpret(Vec{4, UInt64}, b))
end

# ---- complex-layout shuffles (rust: _mm256_permute_pd / _mm256_permute2f128_pd / _mm256_movedup_pd) ----
@inline avx_swap_complex(s::V4f) = shufflevector(s, Val((1, 0, 3, 2)))                   # permute_pd 0x05
@inline avx_dup_re(s::V4f) = shufflevector(s, Val((0, 0, 2, 2)))                         # movedup_pd
@inline avx_dup_im(s::V4f) = shufflevector(s, Val((1, 1, 3, 3)))                         # permute_pd 0x0F
@inline avx_duplicate_complex(s::V4f) = (avx_dup_re(s), avx_dup_im(s))
@inline avx_reverse_complex(s::V4f) = shufflevector(s, Val((2, 3, 0, 1)))                # permute2f128 0x01
@inline avx_unpacklo_complex(a::V4f, b::V4f) = shufflevector(a, b, Val((0, 1, 4, 5)))    # permute2f128 0x20
@inline avx_unpackhi_complex(a::V4f, b::V4f) = shufflevector(a, b, Val((2, 3, 6, 7)))    # permute2f128 0x31

# ---- complex multiply (rust mul_complex) ----
@inline function avx_mul_complex(left::V4f, right::V4f)
    lre = avx_dup_re(left)
    lim = avx_dup_im(left)
    rsh = avx_swap_complex(right)
    avx_fmaddsub(lre, right, avx_mul(lim, rsh))
end

# ---- rotation by 90° (rust make_rotation90 + rotate90) ----
# Forward  → broadcast Complex(-0.0, 0.0) → mask lanes [-0,0,-0,0]
# Inverse  → broadcast Complex(0.0, -0.0) → mask lanes [0,-0,0,-0]
const _ROT90_FWD = V4f((-0.0, 0.0, -0.0, 0.0))
const _ROT90_INV = V4f((0.0, -0.0, 0.0, -0.0))
@inline avx_rotate90(s::V4f, mask::V4f) = avx_swap_complex(avx_xor(s, mask))

# ---- broadcast / twiddles (rust broadcast_complex_elements, twiddles::compute_twiddle) ----
@inline avx_broadcast_complex(re::Float64, im::Float64) = V4f((re, im, re, im))
# rust compute_twiddle: angle = -2π·index/len (Forward); Inverse negates the angle (conjugate).
@inline function compute_twiddle(index::Int, len::Int, forward::Bool)
    angle = (forward ? -2.0 : 2.0) * pi * index / len
    (cos(angle), sin(angle))
end

# ---- load / store 2 complex (rust load_complex / store_complex = loadu_pd / storeu_pd) ----
@inline function avx_load_complex(x::AbstractVector{Complex{Float64}}, i::Int)  # i: 0-based complex index
    GC.@preserve x vload(V4f, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
@inline function avx_store_complex!(x::AbstractVector{Complex{Float64}}, i::Int, v::V4f)
    GC.@preserve x vstore(v, reinterpret(Ptr{Float64}, pointer(x)) + i * 16)
end
