# Canonical faithful-port FFT kernels (no verify/bench blocks). Built on avxport.jl.
# These are the reusable pieces; verify scripts and (eventually) PureFFT integration include this.
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec

# ===== Butterfly7Avx64 =====
function bf7_twiddles(forward::Bool)
    t1r, t1i = compute_twiddle(1, 7, forward); t2r, t2i = compute_twiddle(2, 7, forward); t3r, t3i = compute_twiddle(3, 7, forward)
    (V4f((t1r, t1r, t1i, t1i)), V4f((t2r, t2r, t2i, t2i)), V4f((t3r, t3r, t3i, t3i)), V4f((t3r, t3r, -t3i, -t3i)), V4f((t1r, t1r, -t1i, -t1i)))
end
function butterfly7!(buf, tw, invm::V4f, invm_lo::V2f)
    c0 = avx_load_partial1(buf, 0); input0 = avx_merge(c0, c0)
    input12 = avx_load_complex(buf, 1); input3 = avx_load_partial1(buf, 3); input4 = avx_load_partial1(buf, 4); input56 = avx_load_complex(buf, 5)
    input65 = avx_reverse_complex(input56)
    sum12, diff65 = avx_butterfly2(input12, input65); sum3, diff4 = avx_butterfly2(input3, input4)
    rotated65 = avx_rotate90(diff65, invm); rotated4 = avx_rotate90(diff4, invm_lo)
    mid16, mid25 = avx_transpose_2x2(sum12, rotated65); mid34 = avx_merge(sum3, rotated4)
    o0 = avx_add(avx_add(avx_lo(mid16), avx_lo(mid25)), avx_add(avx_lo(input0), avx_lo(mid34)))
    a = avx_mul(mid16, tw[1]); b = avx_mul(mid16, tw[2]); c = avx_mul(mid16, tw[3])
    a = avx_fmadd(mid25, tw[2], a); b = avx_fmadd(mid25, tw[4], b); c = avx_fmadd(mid25, tw[5], c)
    tw16 = avx_fmadd(mid34, tw[3], a); tw25 = avx_fmadd(mid34, tw[5], b); tw34 = avx_fmadd(mid34, tw[2], c)
    tw12, tw65 = avx_transpose_2x2(tw16, tw25); tw03 = avx_add(avx_lo(tw34), avx_lo(input0))
    out12, out65 = avx_butterfly2(tw12, tw65); final12 = avx_add(out12, input0)
    out56 = avx_reverse_complex(out65); final56 = avx_add(out56, input0); final3, final4 = avx_butterfly2(tw03, avx_hi(tw34))
    avx_store_partial1!(buf, 0, o0); avx_store_complex!(buf, 1, final12); avx_store_partial1!(buf, 3, final3); avx_store_partial1!(buf, 4, final4); avx_store_complex!(buf, 5, final56)
end

# ===== Butterfly36Avx64 (6x6) =====
function bf36_twiddles(forward::Bool)
    ntuple(15) do idx0
        idx = idx0 - 1; y = (idx % 5) + 1; x = (idx ÷ 5) * 2
        avx_mixedradix_twiddle_chunk(x, y, 36, forward)
    end
end
@inline function avx_transpose_6x6(r0::NTuple{6}, r1::NTuple{6}, r2::NTuple{6})
    t(a, b) = avx_transpose_2x2(a, b)
    o00 = t(r0[1], r0[2]); o01 = t(r1[1], r1[2]); o02 = t(r2[1], r2[2])
    o10 = t(r0[3], r0[4]); o11 = t(r1[3], r1[4]); o12 = t(r2[3], r2[4])
    o20 = t(r0[5], r0[6]); o21 = t(r1[5], r1[6]); o22 = t(r2[5], r2[6])
    ((o00[1], o00[2], o01[1], o01[2], o02[1], o02[2]),
     (o10[1], o10[2], o11[1], o11[2], o12[1], o12[2]),
     (o20[1], o20[2], o21[1], o21[2], o22[1], o22[2]))
end
@inline _bf36_ld(buf, off) = (avx_load_complex(buf, off), avx_load_complex(buf, off + 6), avx_load_complex(buf, off + 12),
                              avx_load_complex(buf, off + 18), avx_load_complex(buf, off + 24), avx_load_complex(buf, off + 30))
@inline function _bf36_twmul(m, t1, t2, t3, t4, t5)
    (m[1], avx_mul_complex(m[2], t1), avx_mul_complex(m[3], t2), avx_mul_complex(m[4], t3), avx_mul_complex(m[5], t4), avx_mul_complex(m[6], t5))
end
function butterfly36!(buf, tw::NTuple{15, V4f}, tw3::V4f)
    mid0 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(buf, 0), tw3), tw[1], tw[2], tw[3], tw[4], tw[5])
    mid1 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(buf, 2), tw3), tw[6], tw[7], tw[8], tw[9], tw[10])
    mid2 = _bf36_twmul(avx_column_butterfly6(_bf36_ld(buf, 4), tw3), tw[11], tw[12], tw[13], tw[14], tw[15])
    t0, t1, t2 = avx_transpose_6x6(mid0, mid1, mid2)
    o0 = avx_column_butterfly6(t0, tw3)
    avx_store_complex!(buf, 0, o0[1]); avx_store_complex!(buf, 6, o0[2]); avx_store_complex!(buf, 12, o0[3]); avx_store_complex!(buf, 18, o0[4]); avx_store_complex!(buf, 24, o0[5]); avx_store_complex!(buf, 30, o0[6])
    o1 = avx_column_butterfly6(t1, tw3)
    avx_store_complex!(buf, 2, o1[1]); avx_store_complex!(buf, 8, o1[2]); avx_store_complex!(buf, 14, o1[3]); avx_store_complex!(buf, 20, o1[4]); avx_store_complex!(buf, 26, o1[5]); avx_store_complex!(buf, 32, o1[6])
    o2 = avx_column_butterfly6(t2, tw3)
    avx_store_complex!(buf, 4, o2[1]); avx_store_complex!(buf, 10, o2[2]); avx_store_complex!(buf, 16, o2[3]); avx_store_complex!(buf, 22, o2[4]); avx_store_complex!(buf, 28, o2[5]); avx_store_complex!(buf, 34, o2[6])
end

# ===== shared helpers =====
seeded(n) = [Complex(((k * 2 + 1) % 17) / 17 - 0.5, ((k * 3 + 2) % 19) / 19 - 0.5) for k in 0:(n - 1)]
function golden_fft(n)
    for ln in eachline(joinpath(@__DIR__, "..", "bench", "rustfft_compare", "golden.txt"))
        if startswith(ln, "F $n out")
            bs = parse.(UInt64, split(ln)[4:end]; base = 16)
            return [Complex(reinterpret(Float64, bs[2i - 1]), reinterpret(Float64, bs[2i])) for i in 1:n]
        end
    end
    error("no golden for n=$n")
end
bitsof(v) = [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in v]
