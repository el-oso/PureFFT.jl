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
using BenchmarkTools, Plots, Printf, Statistics

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9

# Per-point timing budget. We want ≥ SAMPLES measurements at each size for a stable MEDIAN (+ a tight
# sigma); BenchmarkTools stops at whichever of SAMPLES / SECONDS comes first, so SECONDS is set high
# enough that even the largest size reaches the sample count.
const SAMPLES = 300    # auto-evals averages each sample → tight median+σ with fewer samples
const SECONDS = 4     # per-point cap; keeps full regen to a few minutes

# Power-of-two sizes (PureFFT's design range).
pow2_sizes() = [2^e for e in 6:18]

# Non-power-of-two sizes: the COMMON case is highly-composite (smooth) sizes, which route to the
# codelet (small) / four-step (larger) path — that's what this plot samples, by picking the
# non-pow2 5-smooth number (2^a·3^b·5^c) nearest each half-integer exponent. (Large-PRIME non-pow2
# sizes fall to Bluestein at ~5 GF/s; that regime is summarized in docs benchmarks.md, not plotted
# here, since sampling only primes — as a naive sweep does — hides the fast smooth-composite path.)
function nonpow2_sizes()
    smooth = sort!(unique(Int[2^a * 3^b * 5^c for a in 0:18 for b in 0:11 for c in 0:7
                                 if 64 <= 2^a * 3^b * 5^c <= 262144]))
    smooth = filter(!ispow2, smooth)
    base = [smooth[argmin(abs.(log2.(smooth) .- e))] for e in 6.5:1.0:18.5]
    # also include a few W=8-clean (2·3-smooth, n=2^(6+3a+2b)·3^b) sizes so the AVX-512 (W=8) path,
    # which autoplan routes for these, is visible in the plot (it covers 2·3-smooth, not 5-smooth).
    w8clean = [768, 6144, 9216, 49152, 110592]
    return sort!(unique(vcat(base, w8clean)))
end

function run_benchmarks(sizes)
    ns = Int[]
    t_fftw = Float64[]
    t_rust = Float64[]
    t_pure = Float64[]
    s_fftw = Float64[]   # relative σ (std/median) per point → rendered as error bars
    s_rust = Float64[]
    s_pure = Float64[]

    for n in sizes
        x = randn(ComplexF64, n)

        pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
        pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
        pp = PureFFT.plan_pfft(x; variant = :fast)

        # MEDIAN time (not min): fairer + less noise-sensitive than min (which rewards lucky outliers).
        # Report relative sigma too so we can confirm the distributions are tight + comparable.
        bm = @benchmark $pm * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
        br = @benchmark $pr * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
        bp = @benchmark PureFFT.pfft!(y, $pp) setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
        tm = median(bm).time / 1.0e9; tr = median(br).time / 1.0e9; tp = median(bp).time / 1.0e9
        # Robust relative spread for the ribbon: the central-68% half-width (q84−q16)/2, normalized to the
        # median. Unlike std/median it is NOT inflated by a handful of GC/OS-preempted outlier samples
        # (which is why the median lines were clean but the std ribbons were wide).
        relspread(b) = (quantile(b.times, 0.84) - quantile(b.times, 0.16)) / 2 / median(b.times)

        @printf(
            "n = %-8d  FFTW %6.1f (σ%4.1f%%)  RustFFT %6.1f (σ%4.1f%%)  PureFFT %6.1f (σ%4.1f%%) GFLOP/s\n",
            n, gflops(n, tm), 100relspread(bm), gflops(n, tr), 100relspread(br), gflops(n, tp), 100relspread(bp),
        )

        push!(ns, n)
        push!(t_fftw, tm)
        push!(t_rust, tr)
        push!(t_pure, tp)
        push!(s_fftw, relspread(bm))
        push!(s_rust, relspread(br))
        push!(s_pure, relspread(bp))
    end

    return ns, t_fftw, t_rust, t_pure, s_fftw, s_rust, s_pure
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
ns, t_fftw, t_rust, t_pure, s_fftw, s_rust, s_pure = run_benchmarks(pow2_sizes())
tickvals, ticklabels = pow2_ticks()
# All plots are RELATIVE to FFTW (FFTW = 1.0 baseline) so they are CLOCK-INDEPENDENT — the absolute
# GFLOP/s (and thus the CPU clock) cancels out, leaving only the speed ratio (so base/boost/pinned all
# give the same plot). Throughput ratio = t_FFTW / t_x (>1 ⇒ faster than FFTW). Ribbon on a ratio r is
# the first-order propagation of both relative spreads: rerr = r·√(σx²+σf²).
relthru(t_x, t_f) = t_f ./ t_x
rerr(r, sx, sf) = r .* sqrt.(sx .^ 2 .+ sf .^ 2)
relband(t_x, t_f, sx, sf) = rerr(relthru(t_x, t_f), sx, sf)

p1 = plot(;
    xlabel = "Transform size N",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "FFT throughput relative to FFTW\n(Zen 5, single-thread, ComplexF64, power-of-two; FFTW = 1.0)",
    xscale = :log2,
    xticks = (tickvals, ticklabels),
    xrotation = 45,
    legend = :bottomleft,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)
hline!(p1, [1.0]; label = LABELS.fftw * " (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p1, ns, relthru(t_rust, t_fftw); ribbon = relband(t_rust, t_fftw, s_rust, s_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p1, ns, relthru(t_pure, t_fftw); ribbon = relband(t_pure, t_fftw, s_pure, s_fftw), fillalpha = 0.18, label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
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
plot!(p2, ns, t_rust ./ t_fftw; ribbon = rerr(t_rust ./ t_fftw, s_rust, s_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p2, ns, t_pure ./ t_fftw; ribbon = rerr(t_pure ./ t_fftw, s_pure, s_fftw), fillalpha = 0.18, label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p2, joinpath(assets, "comparison_time.png"))

# --- non-power-of-two (Bluestein) ---
println("\nNon-power-of-two smooth-composite sizes (PureFFT → codelet / four-step):")
nq, q_fftw, q_rust, q_pure, qs_fftw, qs_rust, qs_pure = run_benchmarks(nonpow2_sizes())
# clip y so a noisy point can't blow up the scale (relative ratios now)
relmax3 = 1.15 * maximum(vcat(relthru(q_rust, q_fftw), relthru(q_pure, q_fftw)))

p3 = plot(;
    xlabel = "Transform size N (non-power-of-two)",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "FFT throughput relative to FFTW — non-power-of-two (smooth composite)\n(Zen 5, single-thread, ComplexF64; FFTW = 1.0)",
    xscale = :log2,
    xticks = (tickvals, ticklabels),
    xrotation = 45,
    ylims = (0, relmax3),
    legend = :topright,
    size = (800, 500),
    dpi = 150,
    margin = 5Plots.mm,
)
hline!(p3, [1.0]; label = LABELS.fftw * " (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p3, nq, relthru(q_rust, q_fftw); ribbon = relband(q_rust, q_fftw, qs_rust, qs_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p3, nq, relthru(q_pure, q_fftw); ribbon = relband(q_pure, q_fftw, qs_pure, qs_fftw), fillalpha = 0.18, label = "PureFFT :fast (mixed-radix)", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p3, joinpath(assets, "comparison_nonpow2.png"))

println("\nSaved: $(joinpath(assets, "comparison.png"))")
println("Saved: $(joinpath(assets, "comparison_time.png"))")
println("Saved: $(joinpath(assets, "comparison_nonpow2.png"))")
