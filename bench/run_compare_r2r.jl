# DCT/DST comparison runner — FFTW vs PureFFT (r2r), saved to bench/results/compare_r2r.json.
# DCT-II (REDFT10), even-N, in-place (PureFFT) vs out-of-place (FFTW). Planning excluded.
#
# Run single-thread, core-pinned:
#   taskset -c 2 julia --project=bench bench/run_compare_r2r.jl
# Relative plots are clock-independent; for tight absolute numbers pin the clock first:
#   sudo bench/cpufreq_lock.sh pin 4500  (restore after)
using BenchmarkTools, Statistics, Printf, Dates, LinearAlgebra
import FFTW, JSON
using PureFFT   # brings REDFT10 + plan_r2r into scope

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9   # same model as the FFT bench (comparability)
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

const SAMPLES = 1000
const SECONDS = 2.0

even_sizes() = [2^e for e in 3:16]

function sample(n, T)
    x = randn(T, n)
    y = similar(x)
    pf = FFTW.plan_r2r(copy(x), FFTW.REDFT10; flags = FFTW.MEASURE)
    pp = PureFFT.plan_r2r(x, REDFT10)
    mul!(y, pp, x)      # warmup
    tf = (@benchmark $pf * _y setup = (_y = copy($x)) samples = SAMPLES seconds = SECONDS).times
    tp = (@benchmark mul!(_y, $pp, $x) setup = (_y = similar($x)) samples = SAMPLES seconds = SECONDS).times
    return tf, tp
end

println("DCT-II (REDFT10) FFTW vs PureFFT  |  $(Sys.CPU_NAME)\n")
results = Dict{String, Any}[]
for T in (Float64, Float32)
    println("$T:")
    for n in even_sizes()
        tf, tp = sample(n, T)
        g(t) = gflops(n, median(t) / 1e9)
        ratio = g(tp) / g(tf)
        @printf("  n=%-7d FFTW %6.1f  PureFFT %6.1f (σ%4.1f%%)  PF/FFTW=%.2f\n",
            n, g(tf), g(tp), 100relspread(tp), ratio)
        push!(results, Dict(
            "kind" => "REDFT10", "T" => string(T), "n" => n,
            "fftw_gflops" => g(tf), "purefft_gflops" => g(tp),
            "fftw_relspread" => relspread(tf), "purefft_relspread" => relspread(tp),
        ))
    end
end

outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkdir(outdir)
out = joinpath(outdir, "compare_r2r.json")
open(out, "w") do io
    JSON.print(io, Dict(
        "meta" => Dict(
            "cpu" => Sys.CPU_NAME, "date" => string(Dates.today()),
            "note" => "DCT-II (REDFT10), even N, PureFFT in-place, FFTW out-of-place, planning excluded",
        ),
        "records" => results,
    ), 2)
end
println("\nSaved → $out")
println("Regenerate plot:  julia -O3 --project=bench bench/plot_compare_r2r.jl")
