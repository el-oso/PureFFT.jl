# Faithful port of RustFFT MixedRadix5xnAvx(inner=Butterfly7) for n=35 (= rustfft's plan_fft(35)).
# ROW_COUNT=5, len_per_row=7. Orchestration: 5-pt column butterflies (3 full Vec{4} chunks + 1 partial
# Vec{2} column, with twiddles) → inner Butterfly7 ×5 → 5×7 transpose. The Phase 4-5 ORCHESTRATION gate.
# Run: julia -O3 --project=bench port/mixedradix35.jl
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec
using Chairmarks, Printf

# --- inner Butterfly7 (copied kernel; bit-exact, see butterfly7.jl) ---
function bf7_tw(forward)
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

# --- MixedRadix5xn twiddles: make_mixedradix_twiddle_chunk(x*2, y, 35), x=0..3, y=1..4 → flat [x*4+y] ---
function mr5_twiddles(forward)
    ntuple(16) do k
        x = (k - 1) ÷ 4; y = (k - 1) % 4 + 1
        avx_mixedradix_twiddle_chunk(x * 2, y, 35, forward)
    end
end

function mixedradix35!(out, buf, tw5_0, tw5_1, mrtw, b7tw, invm, invm_lo)
    # 1) column butterflies: 3 full Vec{4} chunks (cols 0-1,2-3,4-5) + 1 partial (col 6)
    for c in 0:2
        ib = 2c
        o = avx_column_butterfly5(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 7), avx_load_complex(buf, ib + 14),
                                  avx_load_complex(buf, ib + 21), avx_load_complex(buf, ib + 28), tw5_0, tw5_1)
        avx_store_complex!(buf, ib, o[1])
        avx_store_complex!(buf, ib + 7,  avx_mul_complex(mrtw[c * 4 + 1], o[2]))
        avx_store_complex!(buf, ib + 14, avx_mul_complex(mrtw[c * 4 + 2], o[3]))
        avx_store_complex!(buf, ib + 21, avx_mul_complex(mrtw[c * 4 + 3], o[4]))
        avx_store_complex!(buf, ib + 28, avx_mul_complex(mrtw[c * 4 + 4], o[5]))
    end
    # partial column 6 (Vec{2})
    m = avx_column_butterfly5(avx_load_partial1(buf, 6), avx_load_partial1(buf, 13), avx_load_partial1(buf, 20),
                              avx_load_partial1(buf, 27), avx_load_partial1(buf, 34), avx_lo(tw5_0), avx_lo(tw5_1))
    avx_store_partial1!(buf, 6, m[1])
    avx_store_partial1!(buf, 13, avx_mul_complex(avx_lo(mrtw[13]), m[2]))
    avx_store_partial1!(buf, 20, avx_mul_complex(avx_lo(mrtw[14]), m[3]))
    avx_store_partial1!(buf, 27, avx_mul_complex(avx_lo(mrtw[15]), m[4]))
    avx_store_partial1!(buf, 34, avx_mul_complex(avx_lo(mrtw[16]), m[5]))

    # 2) inner Butterfly7 down each of the 5 rows (contiguous: row i = buf[7i .. 7i+6])
    for i in 0:4
        butterfly7!(view(buf, 7i + 1:7i + 7), b7tw, invm, invm_lo)
    end

    # 3) transpose 5x7 -> 7x5 into `out`: 3 full chunks + partial gather
    for c in 0:2
        ib = 2c; ob = 10c
        t = avx_transpose5_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 7), avx_load_complex(buf, ib + 14),
                                  avx_load_complex(buf, ib + 21), avx_load_complex(buf, ib + 28))
        avx_store_complex!(out, ob, t[1]); avx_store_complex!(out, ob + 2, t[2]); avx_store_complex!(out, ob + 4, t[3])
        avx_store_complex!(out, ob + 6, t[4]); avx_store_complex!(out, ob + 8, t[5])
    end
    @inbounds for i in 0:4
        out[30 + i + 1] = buf[6 + 7i + 1]
    end
    return out
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
    tw5_0 = avx_broadcast_twiddle(1, 5, true); tw5_1 = avx_broadcast_twiddle(2, 5, true)
    mrtw = mr5_twiddles(true); b7tw = bf7_tw(true)
    src = seeded(35); buf = copy(src); out = zeros(ComplexF64, 35)
    mixedradix35!(out, buf, tw5_0, tw5_1, mrtw, b7tw, _ROT90_INV, _ROT90_INV2)
    want = golden_fft(35)
    rel = maximum(abs.(out .- want)) / maximum(abs.(want))
    exact = [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in out] ==
            [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in want]
    println("MixedRadix5xn(35) vs rustfft plan_fft(35): ", exact ? "BIT-EXACT ✓" : "rel-error $rel")
    if rel < 1e-12
        t = (@b (copyto!(buf, src)) (_ -> mixedradix35!(out, buf, tw5_0, tw5_1, mrtw, b7tw, _ROT90_INV, _ROT90_INV2)) seconds = 0.5).time
        @printf("MixedRadix5xn(35) Julia: %.2f ns   rustfft golden: 44.38 ns   ratio %.2f×\n", t * 1e9, 44.38 / (t * 1e9))
    else
        println("  out : ", out[1:4]); println("  want: ", want[1:4])
    end
end
