# Faithful port of RustFFT 6.4.1 Butterfly7Avx64::perform_fft_f64 (avx64_butterflies.rs).
# Verified bit-exact against the plan_fft(7) golden. Run:
#   julia -O3 --project=. port/butterfly7.jl
include(joinpath(@__DIR__, "avxport.jl"))
using SIMD: Vec

# twiddles, exactly as Butterfly7Avx64::new_with_avx (twiddle_k = compute_twiddle(k,7,dir))
function butterfly7_twiddles(forward::Bool)
    t1r, t1i = compute_twiddle(1, 7, forward)
    t2r, t2i = compute_twiddle(2, 7, forward)
    t3r, t3i = compute_twiddle(3, 7, forward)
    (V4f((t1r, t1r, t1i, t1i)),
     V4f((t2r, t2r, t2i, t2i)),
     V4f((t3r, t3r, t3i, t3i)),
     V4f((t3r, t3r, -t3i, -t3i)),
     V4f((t1r, t1r, -t1i, -t1i)))
end

# In-place size-7 FFT, transcribed op-for-op from perform_fft_f64.
function butterfly7!(buf::AbstractVector{Complex{Float64}}, tw, invmask::V4f, invmask_lo::V2f)
    c0 = avx_load_partial1(buf, 0)
    input0 = avx_merge(c0, c0)                       # loadu2_m128d(ptr,ptr) = [re0,im0,re0,im0]
    input12 = avx_load_complex(buf, 1)
    input3 = avx_load_partial1(buf, 3)
    input4 = avx_load_partial1(buf, 4)
    input56 = avx_load_complex(buf, 5)
    input65 = avx_reverse_complex(input56)

    sum12, diff65 = avx_butterfly2(input12, input65)
    sum3, diff4 = avx_butterfly2(input3, input4)

    rotated65 = avx_rotate90(diff65, invmask)
    rotated4 = avx_rotate90(diff4, invmask_lo)

    mid16, mid25 = avx_transpose_2x2(sum12, rotated65)
    mid34 = avx_merge(sum3, rotated4)

    output0_left = avx_add(avx_lo(mid16), avx_lo(mid25))
    output0_right = avx_add(avx_lo(input0), avx_lo(mid34))
    output0 = avx_add(output0_left, output0_right)

    tw16_1 = avx_mul(mid16, tw[1])
    tw25_1 = avx_mul(mid16, tw[2])
    tw34_1 = avx_mul(mid16, tw[3])
    tw16_2 = avx_fmadd(mid25, tw[2], tw16_1)
    tw25_2 = avx_fmadd(mid25, tw[4], tw25_1)
    tw34_2 = avx_fmadd(mid25, tw[5], tw34_1)
    tw16 = avx_fmadd(mid34, tw[3], tw16_2)
    tw25 = avx_fmadd(mid34, tw[5], tw25_2)
    tw34 = avx_fmadd(mid34, tw[2], tw34_2)

    tw12, tw65 = avx_transpose_2x2(tw16, tw25)
    tw03 = avx_add(avx_lo(tw34), avx_lo(input0))

    out12, out65 = avx_butterfly2(tw12, tw65)
    final12 = avx_add(out12, input0)
    out56 = avx_reverse_complex(out65)
    final56 = avx_add(out56, input0)
    final3, final4 = avx_butterfly2(tw03, avx_hi(tw34))

    avx_store_partial1!(buf, 0, output0)
    avx_store_complex!(buf, 1, final12)
    avx_store_partial1!(buf, 3, final3)
    avx_store_partial1!(buf, 4, final4)
    avx_store_complex!(buf, 5, final56)
    return buf
end

# ---- verify vs golden ----
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
    buf = seeded(7)
    tw = butterfly7_twiddles(true)
    butterfly7!(buf, tw, _ROT90_INV, _ROT90_INV2)
    want = golden_fft(7)
    gotbits = [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in buf]
    wantbits = [(reinterpret(UInt64, real(z)), reinterpret(UInt64, imag(z))) for z in want]
    exact = gotbits == wantbits
    rel = maximum(abs.(buf .- want)) / maximum(abs.(want))
    println("Butterfly7 vs rustfft plan_fft(7): ", exact ? "BIT-EXACT ✓" : "rel-error $rel")
    exact || (println("  got : ", buf); println("  want: ", want))
    exit(exact ? 0 : 1)
end
