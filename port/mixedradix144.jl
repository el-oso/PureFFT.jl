# Faithful port of RustFFT MixedRadix4xnAvx(inner=Butterfly36) for n=144 (= rustfft's plan_fft(144)).
# ROW_COUNT=4, len_per_row=36 (even → no partial column). Validates a 2nd MixedRadix variant + a
# larger composite reusing Butterfly36. Run: julia -O3 --project=bench port/mixedradix144.jl
include(joinpath(@__DIR__, "kernels.jl"))
using SIMD: Vec
using Chairmarks, Printf

# MixedRadix4xn twiddles: make_mixedradix_twiddle_chunk(x*2, y, 144), x=0..17, y=1..3 → flat [x*3+y]
mr4_twiddles(forward) = ntuple(54) do k
    x = (k - 1) ÷ 3; y = (k - 1) % 3 + 1
    avx_mixedradix_twiddle_chunk(x * 2, y, 144, forward)
end

function mixedradix144!(out, buf, mrtw, bf36tw, tw3, fwd_rot)
    # 1) 4-pt column butterflies across 18 Vec{4} chunks (cols 0..35, no partial), with twiddles
    for c in 0:17
        ib = 2c
        o = avx_column_butterfly4(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 36),
                                  avx_load_complex(buf, ib + 72), avx_load_complex(buf, ib + 108), fwd_rot)
        avx_store_complex!(buf, ib, o[1])
        avx_store_complex!(buf, ib + 36,  avx_mul_complex(mrtw[c * 3 + 1], o[2]))
        avx_store_complex!(buf, ib + 72,  avx_mul_complex(mrtw[c * 3 + 2], o[3]))
        avx_store_complex!(buf, ib + 108, avx_mul_complex(mrtw[c * 3 + 3], o[4]))
    end
    # 2) inner Butterfly36 down each of the 4 rows (row i = buf[36i .. 36i+35])
    for i in 0:3
        butterfly36!(view(buf, 36i + 1:36i + 36), bf36tw, tw3)
    end
    # 3) transpose 4x36 -> 36x4 into `out`: 18 chunks, no partial
    for c in 0:17
        ib = 2c; ob = 8c
        t = avx_transpose4_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 36),
                                  avx_load_complex(buf, ib + 72), avx_load_complex(buf, ib + 108))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2])
        avx_store_complex!(out, ob + 4, t[3]); avx_store_complex!(out, ob + 6, t[4])
    end
    return out
end

let
    mrtw = mr4_twiddles(true); bf36tw = bf36_twiddles(true); tw3 = avx_broadcast_twiddle(1, 3, true)
    src = seeded(144); buf = copy(src); out = zeros(ComplexF64, 144)
    mixedradix144!(out, buf, mrtw, bf36tw, tw3, _ROT90_FWD)
    want = golden_fft(144)
    rel = maximum(abs.(out .- want)) / maximum(abs.(want))
    exact = bitsof(out) == bitsof(want)
    println("MixedRadix4xn(144) vs rustfft plan_fft(144): ", exact ? "BIT-EXACT ✓" : "rel-error $rel")
    if rel < 1e-12
        t = (@b (copyto!(buf, src)) (_ -> mixedradix144!(out, buf, mrtw, bf36tw, tw3, _ROT90_FWD)) seconds = 0.5).time
        @printf("MixedRadix4xn(144) Julia: %.2f ns   rustfft golden: 113.49 ns   ratio %.2f×\n", t * 1e9, 113.49 / (t * 1e9))
    else
        println("  out : ", out[1:3]); println("  want: ", want[1:3])
    end
end
