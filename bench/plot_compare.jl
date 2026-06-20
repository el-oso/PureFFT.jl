# Benchmark comparison plot: FFTW vs RustFFT vs PureFFT :fast
#
# Generates docs/src/assets/comparison.png
#
# Run from the package root:
#   julia -O3 --project=bench bench/plot_compare.jl
#
# The bench env must have: FFTW, RustFFT, BenchmarkTools, PureFFT, Plots.
# Plots can be added with:
#   julia --project=bench -e 'using Pkg; Pkg.add("Plots")'

import FFTW, RustFFT, PureFFT
using BenchmarkTools, Plots, Printf

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9

function run_benchmarks(; sizes = 2 .^ (6:18))
    ns = Int[]
    gf_fftw = Float64[]
    gf_rustfft = Float64[]
    gf_purefft = Float64[]

    for n in sizes
        x = randn(ComplexF64, n)

        pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
        pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
        pp = PureFFT.plan_pfft(x; variant = :fast)

        t_fftw = @belapsed $pm * y setup = (y = copy($x)) evals = 1 samples = 200
        t_rust = @belapsed $pr * y setup = (y = copy($x)) evals = 1 samples = 200
        t_pure = @belapsed PureFFT.pfft!(y, $pp) setup = (y = copy($x)) evals = 1 samples = 200

        @printf(
            "n = %-8d  FFTW %6.1f  RustFFT %6.1f  PureFFT %6.1f GFLOP/s\n",
            n,
            gflops(n, t_fftw),
            gflops(n, t_rust),
            gflops(n, t_pure),
        )

        push!(ns, n)
        push!(gf_fftw, gflops(n, t_fftw))
        push!(gf_rustfft, gflops(n, t_rust))
        push!(gf_purefft, gflops(n, t_pure))
    end

    return ns, gf_fftw, gf_rustfft, gf_purefft
end

println("Benchmarking FFTW vs RustFFT vs PureFFT :fast  |  $(Sys.CPU_NAME)  |  ComplexF64")
println("Single-thread, in-place, planning excluded\n")

ns, gf_fftw, gf_rustfft, gf_purefft = run_benchmarks()

# --- Plot ---

xtick_labels = ["2^$(round(Int, log2(n)))" for n in ns]

p = plot(;
    xlabel = "Transform size N",
    ylabel = "GFLOP/s",
    title = "FFT performance: FFTW vs RustFFT vs PureFFT\n(Zen 5, single-thread, ComplexF64, planning excluded)",
    xscale = :log2,
    xticks = (ns, xtick_labels),
    xrotation = 45,
    legend = :bottomleft,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)

plot!(p, ns, gf_fftw; label = "FFTW (MEASURE)", color = :steelblue, linewidth = 2, marker = :circle, markersize = 5)
plot!(p, ns, gf_rustfft; label = "RustFFT (AVX)", color = :tomato, linewidth = 2, marker = :circle, markersize = 5)
plot!(p, ns, gf_purefft; label = "PureFFT :fast", color = :seagreen, linewidth = 2, marker = :circle, markersize = 5)

outpath = joinpath(@__DIR__, "..", "docs", "src", "assets", "comparison.png")
mkpath(dirname(outpath))
savefig(p, outpath)

println("\nSaved to: $outpath")
