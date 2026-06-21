# Faithful port of RustFFT 6.4.1 Butterfly36Avx64 (6x6). Compute-bound codegen-parity check:
# hammers mul_complex (the fmaddsub/llvmcall path) under load. Run:
#   julia -O3 --project=bench port/butterfly36.jl
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec
using Chairmarks, Printf

# twiddles: gen_butterfly_twiddles_separated_columns!(6,6,0): index 0..14, y=(idx%5)+1, x=(idx÷5)*2
function butterfly36_twiddles(forward::Bool)
    ntuple(15) do idx0
        idx = idx0 - 1
        y = (idx % 5) + 1
        x = (idx ÷ 5) * 2
        avx_mixedradix_twiddle_chunk(x, y, 36, forward)
    end
end

# avx64_utils::transpose_6x6_f64
@inline function avx_transpose_6x6(r0::NTuple{6}, r1::NTuple{6}, r2::NTuple{6})
    t(a, b) = avx_transpose_2x2(a, b)
    o00 = t(r0[1], r0[2]); o01 = t(r1[1], r1[2]); o02 = t(r2[1], r2[2])
    o10 = t(r0[3], r0[4]); o11 = t(r1[3], r1[4]); o12 = t(r2[3], r2[4])
    o20 = t(r0[5], r0[6]); o21 = t(r1[5], r1[6]); o22 = t(r2[5], r2[6])
    ((o00[1], o00[2], o01[1], o01[2], o02[1], o02[2]),
     (o10[1], o10[2], o11[1], o11[2], o12[1], o12[2]),
     (o20[1], o20[2], o21[1], o21[2], o22[1], o22[2]))
end

# fully unrolled — NO runtime tuple indexing (which boxes in Julia); mirrors Rust's const-range loops.
@inline _ld(buf, off) = (avx_load_complex(buf, off), avx_load_complex(buf, off + 6), avx_load_complex(buf, off + 12),
                         avx_load_complex(buf, off + 18), avx_load_complex(buf, off + 24), avx_load_complex(buf, off + 30))
@inline function _twmul(m, t1, t2, t3, t4, t5)   # m = column_butterfly6 result; twiddle rows 2..6 (row1 untouched)
    (m[1], avx_mul_complex(m[2], t1), avx_mul_complex(m[3], t2), avx_mul_complex(m[4], t3),
     avx_mul_complex(m[5], t4), avx_mul_complex(m[6], t5))
end
function butterfly36!(buf::AbstractVector{Complex{Float64}}, tw::NTuple{15, V4f}, tw3::V4f)
    mid0 = _twmul(avx_column_butterfly6(_ld(buf, 0), tw3), tw[1], tw[2], tw[3], tw[4], tw[5])
    mid1 = _twmul(avx_column_butterfly6(_ld(buf, 2), tw3), tw[6], tw[7], tw[8], tw[9], tw[10])
    mid2 = _twmul(avx_column_butterfly6(_ld(buf, 4), tw3), tw[11], tw[12], tw[13], tw[14], tw[15])

    t0, t1, t2 = avx_transpose_6x6(mid0, mid1, mid2)

    o0 = avx_column_butterfly6(t0, tw3)
    avx_store_complex!(buf, 0, o0[1]); avx_store_complex!(buf, 6, o0[2]); avx_store_complex!(buf, 12, o0[3])
    avx_store_complex!(buf, 18, o0[4]); avx_store_complex!(buf, 24, o0[5]); avx_store_complex!(buf, 30, o0[6])
    o1 = avx_column_butterfly6(t1, tw3)
    avx_store_complex!(buf, 2, o1[1]); avx_store_complex!(buf, 8, o1[2]); avx_store_complex!(buf, 14, o1[3])
    avx_store_complex!(buf, 20, o1[4]); avx_store_complex!(buf, 26, o1[5]); avx_store_complex!(buf, 32, o1[6])
    o2 = avx_column_butterfly6(t2, tw3)
    avx_store_complex!(buf, 4, o2[1]); avx_store_complex!(buf, 10, o2[2]); avx_store_complex!(buf, 16, o2[3])
    avx_store_complex!(buf, 22, o2[4]); avx_store_complex!(buf, 28, o2[5]); avx_store_complex!(buf, 34, o2[6])
    return buf
end

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

let
    tw = butterfly36_twiddles(true)
    tw3 = avx_broadcast_twiddle(1, 3, true)
    buf = seeded(36); butterfly36!(buf, tw, tw3)
    want = golden_fft(36)
    exact = [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in buf] ==
            [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in want]
    rel = maximum(abs.(buf .- want)) / maximum(abs.(want))
    println("Butterfly36 vs rustfft plan_fft(36): ", exact ? "BIT-EXACT ✓" : "rel-error $rel")
    src = seeded(36)
    t = (@b copy(src) (w -> butterfly36!(w, tw, tw3)) seconds = 0.5).time
    @printf("Butterfly36 Julia: %.2f ns/call   rustfft golden: 16.94 ns   ratio %.2f×\n", t * 1e9, 16.94 / (t * 1e9))
end
