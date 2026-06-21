# Parity check: Julia faithful-port kernels vs rust golden, MEDIAN times + comparable SIGMAS.
# Criterion: ratio = rust_median / julia_median ≥ 0.96, AND julia rel-sigma comparable to rust's.
# Run: taskset -c 2 julia -O3 --project=bench port/parity_check.jl
include(joinpath(@__DIR__, "kernels.jl"))
using Printf, Statistics

# rust golden medians+sigmas
const RUST = Dict{Int, Tuple{Float64, Float64}}()
for ln in eachline(joinpath(@__DIR__, "..", "bench", "rustfft_compare", "golden.txt"))
    startswith(ln, "T ") || continue
    p = split(ln); RUST[parse(Int, p[2])] = (parse(Float64, p[3]), parse(Float64, p[4]))
end

# stable in-place median+sigma: BLOCKS block-estimates (each kit in-place reps), no copy-subtract.
function med_sigma(run!, n; blocks = 101)
    w = copy(seeded(n)); kit = max(100, round(Int, 2.0e8 / (n * log2(n))))
    for _ in 1:5, _ in 1:kit; run!(w); end                      # warm
    times = Float64[]
    for _ in 1:blocks
        t = time_ns(); for _ in 1:kit; run!(w); end
        push!(times, Float64(time_ns() - t) / kit)
    end
    sink[] += real(w[1])                                        # defeat DCE
    (median(times), std(times))
end
const sink = Ref(0.0)

function report(name, n, run!)
    jm, js = med_sigma(run!, n)
    rm, rs = RUST[n]
    ratio = rm / jm
    ok = ratio ≥ 0.96
    @printf("%-10s n=%-5d  julia med=%8.2f (σ %.1f%%)  rust med=%8.2f (σ %.1f%%)  ratio=%.3f  %s\n",
        name, n, jm, 100js / jm, rm, 100rs / rm, ratio, ok ? "✓≥0.96" : "✗")
    ok
end

# kernels
const B7TW = bf7_twiddles(true)
@noinline r7!(w) = butterfly7!(w, B7TW, _ROT90_INV, _ROT90_INV2)
const B36TW = bf36_twiddles(true); const B36TW3 = avx_broadcast_twiddle(1, 3, true)
@noinline r36!(w) = butterfly36!(w, B36TW, B36TW3)
# MixedRadix4xn-144
const MR4 = ntuple(54) do k; x = (k - 1) ÷ 3; y = (k - 1) % 3 + 1; avx_mixedradix_twiddle_chunk(x * 2, y, 144, true); end
const OUT144 = zeros(ComplexF64, 144)
function mr144!(buf)
    for c in 0:17; ib = 2c
        o = avx_column_butterfly4(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 36), avx_load_complex(buf, ib + 72), avx_load_complex(buf, ib + 108), _ROT90_FWD)
        avx_store_complex!(buf, ib, o[1]); avx_store_complex!(buf, ib + 36, avx_mul_complex(MR4[c * 3 + 1], o[2])); avx_store_complex!(buf, ib + 72, avx_mul_complex(MR4[c * 3 + 2], o[3])); avx_store_complex!(buf, ib + 108, avx_mul_complex(MR4[c * 3 + 3], o[4]))
    end
    for i in 0:3; butterfly36!(view(buf, 36i + 1:36i + 36), B36TW, B36TW3); end
    for c in 0:17; ib = 2c; ob = 8c
        t = avx_transpose4_packed(avx_load_complex(buf, ib), avx_load_complex(buf, ib + 36), avx_load_complex(buf, ib + 72), avx_load_complex(buf, ib + 108))
        avx_store_complex!(OUT144, ob, t[1]); avx_store_complex!(OUT144, ob + 2, t[2]); avx_store_complex!(OUT144, ob + 4, t[3]); avx_store_complex!(OUT144, ob + 6, t[4])
    end
end
@noinline r144!(w) = mr144!(w)

all_ok = true
all_ok &= report("Butterfly7", 7, r7!)
all_ok &= report("Butterfly36", 36, r36!)
all_ok &= report("MixedRadix144", 144, r144!)
println(all_ok ? "\nALL ≥0.96× (median) ✓" : "\nsome below 0.96× (median)")
