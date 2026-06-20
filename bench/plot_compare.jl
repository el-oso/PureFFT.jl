# Benchmark comparison plots: FFTW vs RustFFT vs PureFFT :fast
#
# Generates docs/src/assets/comparison.png          (GFLOP/s vs N, power-of-two)
#       and docs/src/assets/comparison_time.png     (time per transform vs N, power-of-two)
#       and docs/src/assets/comparison_nonpow2.png  (GFLOP/s vs N, non-power-of-two / Bluestein)
#
# Power-of-two is PureFFT's primary design range, so it gets clean throughput + runtime plots.
# Non-power-of-two sizes (handled by Bluestein chirp-Z) are shown separately: they do ~3× the
# nominal flops, so plotting them alongside pow2 just produces an unreadable sawtooth.
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

# Per-point timing budget. We want ≥ SAMPLES measurements at each size for a stable minimum;
# BenchmarkTools stops at whichever of SAMPLES / SECONDS comes first, so SECONDS is set high
# enough that even the largest size reaches the sample count.
const SAMPLES = 1_000   # min-time stabilizes well before this; 1000 keeps regen fast (run on every push)
const SECONDS = 30

# Power-of-two sizes (PureFFT's design range). Non-power-of-two sizes: half-integer exponents
# rounded to ints, dropping any that landed on a power of two — every one has a large prime
# factor, so they exercise the Bluestein path.
pow2_sizes() = [2^e for e in 6:18]
nonpow2_sizes() = filter(!ispow2, unique(round.(Int, 2 .^ (6.5:1.0:18.5))))

function run_benchmarks(sizes)
    ns = Int[]
    t_fftw = Float64[]
    t_rust = Float64[]
    t_pure = Float64[]

    for n in sizes
        x = randn(ComplexF64, n)

        pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
        pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
        pp = PureFFT.plan_pfft(x; variant = :fast)

        tm = @belapsed $pm * y setup = (y = copy($x)) evals = 1 samples = SAMPLES seconds = SECONDS
        tr = @belapsed $pr * y setup = (y = copy($x)) evals = 1 samples = SAMPLES seconds = SECONDS
        tp = @belapsed PureFFT.pfft!(y, $pp) setup = (y = copy($x)) evals = 1 samples = SAMPLES seconds = SECONDS

        @printf(
            "n = %-8d  FFTW %6.1f  RustFFT %6.1f  PureFFT %6.1f GFLOP/s\n",
            n, gflops(n, tm), gflops(n, tr), gflops(n, tp),
        )

        push!(ns, n)
        push!(t_fftw, tm)
        push!(t_rust, tr)
        push!(t_pure, tp)
    end

    return ns, t_fftw, t_rust, t_pure
end

const COLORS = (fftw = :steelblue, rust = :tomato, pure = :seagreen)
const LABELS = (fftw = "FFTW (MEASURE)", rust = "RustFFT (AVX)", pure = "PureFFT :fast")

# power-of-two axis ticks (2^6 … 2^18)
pow2_ticks() = ([2^e for e in 6:18], ["2^$e" for e in 6:18])

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)

println("Benchmarking FFTW vs RustFFT vs PureFFT :fast  |  $(Sys.CPU_NAME)  |  ComplexF64")
println("Single-thread, in-place, planning excluded\n")

# --- power-of-two ---
println("Power-of-two sizes:")
ns, t_fftw, t_rust, t_pure = run_benchmarks(pow2_sizes())
tickvals, ticklabels = pow2_ticks()

p1 = plot(;
    xlabel = "Transform size N",
    ylabel = "GFLOP/s",
    title = "FFT throughput: FFTW vs RustFFT vs PureFFT\n(Zen 5, single-thread, ComplexF64, power-of-two, planning excluded)",
    xscale = :log2,
    xticks = (tickvals, ticklabels),
    xrotation = 45,
    legend = :bottomleft,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)
plot!(p1, ns, gflops.(ns, t_fftw); label = LABELS.fftw, color = COLORS.fftw, linewidth = 2, marker = :circle, markersize = 4)
plot!(p1, ns, gflops.(ns, t_rust); label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p1, ns, gflops.(ns, t_pure); label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p1, joinpath(assets, "comparison.png"))

# Runtime normalized to FFTW: FFTW is the flat 1.0 baseline; a curve at 1.2 means 20 % slower
# than FFTW, below 1.0 means faster. Reads parity far more clearly than overlapping log-log lines.
p2 = plot(;
    xlabel = "Transform size N",
    ylabel = "time relative to FFTW  (t / t_FFTW)",
    title = "FFT runtime relative to FFTW\n(Zen 5, single-thread, ComplexF64, power-of-two; lower = faster)",
    xscale = :log2,
    xticks = (tickvals, ticklabels),
    xrotation = 45,
    legend = :topleft,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)
hline!(p2, [1.0]; label = "FFTW (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p2, ns, t_rust ./ t_fftw; label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p2, ns, t_pure ./ t_fftw; label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p2, joinpath(assets, "comparison_time.png"))

# --- non-power-of-two (Bluestein) ---
println("\nNon-power-of-two sizes (PureFFT → Bluestein chirp-Z):")
nq, q_fftw, q_rust, q_pure = run_benchmarks(nonpow2_sizes())

p3 = plot(;
    xlabel = "Transform size N (non-power-of-two)",
    ylabel = "GFLOP/s (nominal 5·N·log₂N)",
    title = "FFT throughput on non-power-of-two sizes\n(Zen 5, single-thread, ComplexF64; PureFFT uses Bluestein chirp-Z)",
    xscale = :log2,
    xticks = (tickvals, ticklabels),
    xrotation = 45,
    legend = :topright,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)
plot!(p3, nq, gflops.(nq, q_fftw); label = LABELS.fftw, color = COLORS.fftw, linewidth = 2, marker = :circle, markersize = 4)
plot!(p3, nq, gflops.(nq, q_rust); label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p3, nq, gflops.(nq, q_pure); label = "PureFFT :fast (Bluestein)", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p3, joinpath(assets, "comparison_nonpow2.png"))

println("\nSaved: $(joinpath(assets, "comparison.png"))")
println("Saved: $(joinpath(assets, "comparison_time.png"))")
println("Saved: $(joinpath(assets, "comparison_nonpow2.png"))")
