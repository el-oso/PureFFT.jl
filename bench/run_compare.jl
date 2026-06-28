# Benchmark RUNNER: FFTW vs RustFFT vs PureFFT :fast — saves full per-sample datapoints.
#
# This is the ONE step that re-runs the benchmarks; it writes every sample to
#   bench/results/compare.json
# so the plots can be regenerated *from the saved data* (bench/plot_compare.jl), never by
# re-running (re-running is noisy). Run it only to capture a fresh measurement.
#
# Run from the package root, single-thread, core-pinned:
#   taskset -c 2 julia -O3 -t 1 --project=bench bench/run_compare.jl
# The plots are RELATIVE to FFTW (clock-independent), so a cpufreq pin is optional; pin for the
# tightest spreads: `sudo bench/cpufreq_lock.sh pin 4500` (restore after).
#
# bench env needs: FFTW, RustFFT, BenchmarkTools, PureFFT, JSON.

import FFTW, RustFFT, PureFFT
using BenchmarkTools, Printf, Statistics, Dates
import JSON

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9

const SAMPLES = 300    # ≥ this many measurements per size for a stable median + tight spread
const SECONDS = 4      # per-point cap
const MAXSAVE = 2000   # cap saved samples per (size, method) — well above SAMPLES

# Power-of-two sizes (PureFFT's design range).
pow2_sizes() = [2^e for e in 6:18]

# Non-power-of-two: nearest non-pow2 5-smooth (2^a·3^b·5^c) to each half-integer exponent, plus a few
# W=8-clean (2·3-smooth) sizes so the AVX-512 (W=8) path is visible. (Large-prime sizes fall to
# Bluestein ~5 GF/s and are summarized in docs, not plotted — sampling only primes hides the fast path.)
function nonpow2_sizes()
    smooth = sort!(unique(Int[2^a * 3^b * 5^c for a in 0:18 for b in 0:11 for c in 0:7
                                 if 64 <= 2^a * 3^b * 5^c <= 262144]))
    smooth = filter(!ispow2, smooth)
    base = [smooth[argmin(abs.(log2.(smooth) .- e))] for e in 6.5:1.0:18.5]
    w8clean = [768, 6144, 9216, 49152, 110592]
    small3 = [48, 96, 192, 384]   # single-factor-of-3 2^k·3 sizes (B16/B64 leaf + MR{3,6} route)
    small57 = [80, 112, 160, 224, 240, 320, 448, 480]   # 2^k·{5,7}, 2^k·3·5 (B32 leaf + MR5/MR7 route)
    return sort!(unique(vcat(small3, small57, base, w8clean)))
end

cap(t) = length(t) > MAXSAVE ? t[1:MAXSAVE] : t

# Returns the raw per-sample times (ns) for each backend at size `n`.
function sample_times(n)
    x = randn(ComplexF64, n)
    pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
    pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
    pp = PureFFT.plan_pfft(x; variant = :fast)
    bm = @benchmark $pm * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    br = @benchmark $pr * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    bp = @benchmark PureFFT.pfft!(y, $pp) setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    return cap(bm.times), cap(br.times), cap(bp.times)
end

relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

records = Dict{String, Any}[]
println("Benchmarking FFTW vs RustFFT vs PureFFT :fast  |  $(Sys.CPU_NAME)  |  ComplexF64")
println("Single-thread, in-place, planning excluded\n")
for (group, sizes) in (("pow2", pow2_sizes()), ("nonpow2", nonpow2_sizes()))
    println("$group sizes:")
    for n in sizes
        tm, tr, tp = sample_times(n)
        for (method, t) in (("FFTW", tm), ("RustFFT", tr), ("PureFFT", tp))
            push!(records, Dict("group" => group, "n" => n, "method" => method, "times_ns" => t))
        end
        @printf(
            "  n = %-8d  FFTW %6.1f (σ%4.1f%%)  RustFFT %6.1f (σ%4.1f%%)  PureFFT %6.1f (σ%4.1f%%) GFLOP/s\n",
            n, gflops(n, median(tm) / 1e9), 100relspread(tm), gflops(n, median(tr) / 1e9), 100relspread(tr),
            gflops(n, median(tp) / 1e9), 100relspread(tp),
        )
    end
end

data = Dict(
    "meta" => Dict(
        "cpu" => Sys.CPU_NAME, "julia" => string(VERSION), "date" => string(Dates.now()),
        "samples" => SAMPLES, "seconds" => SECONDS, "flop_model" => "5*N*log2(N)",
        "methods" => ["FFTW", "RustFFT", "PureFFT"], "units" => "times_ns are per-transform times in nanoseconds",
        "note" => "single-thread, in-place, planning excluded; plots are relative-to-FFTW (clock-independent)",
    ),
    "records" => records,
)
outdir = joinpath(@__DIR__, "results")
mkpath(outdir)
out = joinpath(outdir, "compare.json")
open(out, "w") do io
    JSON.print(io, data)
end
println("\nSaved datapoints → $out  ($(length(records)) records)")
println("Now regenerate plots from it:  julia -O3 --project=bench bench/plot_compare.jl")
