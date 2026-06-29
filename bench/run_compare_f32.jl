# Float32 comparison runner — FFTW / RustFFT / PureFFT on ComplexF32, plus PureFFT ComplexF64 for the
# F32/F64 throughput ratio. Re-runs the benchmarks ONCE and saves every median to
#   bench/results/compare_f32.json
# (the F32 sibling of run_compare.jl). Single-thread, in-place, planning excluded; medians + central-68%
# spread. Relative-to-FFTW columns are clock-independent; for low-noise absolute numbers pin the clock
# (sudo bench/cpufreq_lock.sh pin 4500) and run `taskset -c 2 julia --project=bench bench/run_compare_f32.jl`.
using BenchmarkTools, Statistics, Printf, Dates
import FFTW, RustFFT, PureFFT, JSON

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)
const SAMPLES = 1000
const SECONDS = 2.0

pow2_sizes() = [2^e for e in 8:16]
# Non-pow2: the small 2^k·{3,5,7,3·5} sizes (v2≥2) served by the Vec{8,Float32} W=8 tree (bases
# B4/B8/B16/B32/B64W8 + radix-3/5/7/9), plus the v2=1 (2·odd) sizes 54/90/162/270/486/810 served by the
# W=8 partial-column subsystem (B2W8 base + rem=2 tails), plus the larger W=8-clean sizes for the main solver.
# Plus the D2 bare-prime sizes (11/13/19 + 2·P/5·19 composites 22/26/95/190) served by the BP-W8 direct
# prime leaf + radix-2 (MR2W8).
nonpow2_sizes() = [11, 13, 19, 22, 24, 25, 26, 27, 36, 40, 48, 49, 54, 72, 80, 81, 90, 95, 96, 98, 112, 160, 162, 180, 190, 192, 224, 240, 270, 343, 360, 384, 448, 480, 486, 720, 810, 768, 2880, 9216, 23040]

function sample(n)
    xf = randn(ComplexF32, n); xd = randn(ComplexF64, n)
    pm = FFTW.plan_fft!(copy(xf); flags = FFTW.MEASURE)
    pr = RustFFT.plan_fft!(copy(xf); rustfft_checks = RustFFT.IgnoreArrayChecks())
    pp = PureFFT.plan_pfft(xf; variant = :fast)
    ppd = PureFFT.plan_pfft(xd; variant = :fast)
    tm = (@benchmark $pm * y setup = (y = copy($xf)) samples = SAMPLES seconds = SECONDS).times
    tr = (@benchmark $pr * y setup = (y = copy($xf)) samples = SAMPLES seconds = SECONDS).times
    tp = (@benchmark PureFFT.pfft!(y, $pp) setup = (y = copy($xf)) samples = SAMPLES seconds = SECONDS).times
    td = (@benchmark PureFFT.pfft!(y, $ppd) setup = (y = copy($xd)) samples = SAMPLES seconds = SECONDS).times
    (tm, tr, tp, td)
end

println("Float32: FFTW vs RustFFT vs PureFFT (+ PureFFT F64 ratio)  |  $(Sys.CPU_NAME)\n")
results = Dict{String, Any}[]
for (group, sizes) in (("pow2", pow2_sizes()), ("nonpow2", nonpow2_sizes()))
    println("$group:")
    for n in sizes
        tm, tr, tp, td = sample(n)
        g(t) = gflops(n, median(t) / 1e9)
        @printf("  n=%-7d FFTW %6.1f  Rust %6.1f  PureF32 %6.1f (σ%4.1f%%)  PureF64 %6.1f  F32/F64=%.2fx  PF/FFTW=%.2f PF/Rust=%.2f\n",
            n, g(tm), g(tr), g(tp), 100relspread(tp), g(td), median(td) / median(tp), g(tp) / g(tm), g(tp) / g(tr))
        # one record per (n, method): median time[s], GFLOP/s, relative spread — enough for the
        # relative-to-FFTW plots with ribbons (mirrors run_compare.jl's per-method records).
        for (method, t) in (("FFTW", tm), ("RustFFT", tr), ("PureFFT_F32", tp), ("PureFFT_F64", td))
            push!(results, Dict("group" => group, "n" => n, "method" => method,
                "median_s" => median(t) / 1e9, "gflops" => g(t), "relspread" => relspread(t)))
        end
    end
end

outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkdir(outdir)
open(joinpath(outdir, "compare_f32.json"), "w") do io
    JSON.print(io, Dict("meta" => Dict("cpu" => Sys.CPU_NAME, "date" => string(Dates.today()),
            "metric" => "GFLOP/s (median)",
            "note" => "ComplexF32 (+ PureFFT ComplexF64 for the F32/F64 ratio); single-thread, in-place, planning excluded"),
        "records" => results), 2)
end
println("\nsaved → bench/results/compare_f32.json")
