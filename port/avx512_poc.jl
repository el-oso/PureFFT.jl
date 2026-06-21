# Phase 8 proof-of-concept: AVX-512 (Vec{8,Float64} = 4 complex/vector) for the non-pow2 compute core.
# RustFFT is AVX2-only (2 complex/vector) — so a working V8f column butterfly that processes 2× the
# complex per vector op is where PureFFT can EXCEED rust on Zen5. Verify cb8 at 512-bit, then measure
# the per-complex speedup of the column-butterfly pass V8f vs V4f.
include(joinpath(@__DIR__, "..", "src", "avxradix", "avxport.jl"))   # V4f primitives + cb4/cb8
using SIMD: Vec, shufflevector
using Printf, Statistics

const V8f = Vec{8, Float64}
# V8f primitives (512-bit lane patterns: 4 interleaved complex)
@inline avx_swap_complex(s::V8f) = shufflevector(s, Val((1, 0, 3, 2, 5, 4, 7, 6)))
@inline avx_dup_re(s::V8f) = shufflevector(s, Val((0, 0, 2, 2, 4, 4, 6, 6)))
@inline avx_dup_im(s::V8f) = shufflevector(s, Val((1, 1, 3, 3, 5, 5, 7, 7)))
@inline avx_xor(a::V8f, b::V8f) = reinterpret(V8f, reinterpret(Vec{8, UInt64}, a) ⊻ reinterpret(Vec{8, UInt64}, b))
const _ROT90_FWD8 = V8f((-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0))
const _ROT90_INV8 = V8f((0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0))
const _HALF_ROOT2_8 = V8f(ntuple(_ -> sqrt(0.5), Val(8)))
const _SGN8 = V8f((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0))
# complex multiply (dup left, swap right) — muladd+sign, no risky .512 fmaddsub intrinsic (cf. _vcmul)
@inline avx_mul_complex(left::V8f, right::V8f) =
    muladd(avx_dup_re(left), right, _SGN8 * (avx_dup_im(left) * avx_swap_complex(right)))
@inline _half_root2(::V4f) = _HALF_ROOT2
@inline _half_root2(::V8f) = _HALF_ROOT2_8

# width-generic cb4/cb8 (mirror avxport, but bf8 twiddle uses the width-dispatched _half_root2)
@inline function cb4w(a, b, c, d, rot)
    mid0, mid2 = avx_butterfly2(a, c)
    mid1, mid3 = avx_butterfly2(b, d)
    mid3r = avx_rotate90(mid3, rot)
    o0, o1 = avx_butterfly2(mid0, mid1)
    o2, o3 = avx_butterfly2(mid2, mid3r)
    (o0, o2, o1, o3)
end
@inline bf8tw1w(x, rot) = avx_mul(_half_root2(x), avx_add(avx_rotate90(x, rot), x))
@inline bf8tw3w(x, rot) = avx_mul(_half_root2(x), avx_sub(avx_rotate90(x, rot), x))
@inline function cb8w(r1, r2, r3, r4, r5, r6, r7, r8, rot)
    m0 = cb4w(r1, r3, r5, r7, rot); m1 = cb4w(r2, r4, r6, r8, rot)
    m1_2 = bf8tw1w(m1[2], rot); m1_3 = avx_rotate90(m1[3], rot); m1_4 = bf8tw3w(m1[4], rot)
    o0, o1 = avx_butterfly2(m0[1], m1[1]); o2, o3 = avx_butterfly2(m0[2], m1_2)
    o4, o5 = avx_butterfly2(m0[3], m1_3); o6, o7 = avx_butterfly2(m0[4], m1_4)
    (o0, o2, o4, o6, o1, o3, o5, o7)
end

# ---- verify cb8w at V8f vs DFT-8 (4 independent transforms across the 4 complex lanes) ----
getc(v::V8f, k) = Complex(v[2k + 1], v[2k + 2])           # complex k (0..3) of a V8f
rows = ntuple(_ -> V8f(ntuple(_ -> randn(), Val(8))), 8)
o = cb8w(rows..., _ROT90_FWD8)
maxerr = 0.0
for lane in 0:3
    x = [getc(rows[r + 1], lane) for r in 0:7]
    ref = [sum(x[j + 1] * cispi(-2.0 * j * k / 8) for j in 0:7) for k in 0:7]
    got = [getc(o[k + 1], lane) for k in 0:7]
    global maxerr = max(maxerr, maximum(abs.(got .- ref)))
end
@printf("cb8 V8f (4 transforms/vec) vs DFT-8: max-err %.2e  %s\n", maxerr, maxerr < 1e-12 ? "✓" : "WRONG")

# ---- benchmark: column-butterfly pass throughput, V8f (4 complex/chunk) vs V4f (2 complex/chunk) ----
# Same #chunks C; V8f moves 2× the complex per chunk → expect ~2× per-complex throughput.
const C = 32
b4 = [V4f(ntuple(_ -> randn(), Val(4))) for _ in 1:(8C)]
b8 = [V8f(ntuple(_ -> randn(), Val(8))) for _ in 1:(8C)]
tw4 = [V4f(ntuple(_ -> randn(), Val(4))) for _ in 1:(7C)]
tw8 = [V8f(ntuple(_ -> randn(), Val(8))) for _ in 1:(7C)]
@noinline function pass4!(b, tw)
    @inbounds for c in 0:(C - 1)
        i = 8c; r = cb8w(b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], _ROT90_FWD)
        b[i+1] = r[1]; t = 7c
        b[i+2] = avx_mul_complex(tw[t+1], r[2]); b[i+3] = avx_mul_complex(tw[t+2], r[3]); b[i+4] = avx_mul_complex(tw[t+3], r[4])
        b[i+5] = avx_mul_complex(tw[t+4], r[5]); b[i+6] = avx_mul_complex(tw[t+5], r[6]); b[i+7] = avx_mul_complex(tw[t+6], r[7]); b[i+8] = avx_mul_complex(tw[t+7], r[8])
    end
end
@noinline function pass8!(b, tw)
    @inbounds for c in 0:(C - 1)
        i = 8c; r = cb8w(b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7], b[i+8], _ROT90_FWD8)
        b[i+1] = r[1]; t = 7c
        b[i+2] = avx_mul_complex(tw[t+1], r[2]); b[i+3] = avx_mul_complex(tw[t+2], r[3]); b[i+4] = avx_mul_complex(tw[t+3], r[4])
        b[i+5] = avx_mul_complex(tw[t+4], r[5]); b[i+6] = avx_mul_complex(tw[t+5], r[6]); b[i+7] = avx_mul_complex(tw[t+6], r[7]); b[i+8] = avx_mul_complex(tw[t+7], r[8])
    end
end
med(f, b, tw) = (for _ in 1:200; f(b, tw); end; ts = Float64[]; for _ in 1:101; t = time_ns(); for _ in 1:2000; f(b, tw); end; push!(ts, (time_ns() - t) / 2000); end; median(ts))
m4 = med(pass4!, b4, tw4); m8 = med(pass8!, b8, tw8)
# complex processed per pass: V4f = 2·8·C, V8f = 4·8·C → throughput = complex/ns
@printf("colbf pass: V4f %.1f ns (%.2f Gc/s), V8f %.1f ns (%.2f Gc/s) → %.2f× per-complex throughput\n",
        m4, 2 * 8C / m4, m8, 4 * 8C / m8, (4 * 8C / m8) / (2 * 8C / m4))
