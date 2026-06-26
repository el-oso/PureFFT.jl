# Benchmark comparison PLOTS: FFTW vs RustFFT vs PureFFT :fast — regenerated from SAVED data.
#
# Reads bench/results/compare.json (written by bench/run_compare.jl) and regenerates:
#   docs/src/assets/comparison.png          (throughput relative to FFTW, power-of-two)
#   docs/src/assets/comparison_time.png     (runtime relative to FFTW, power-of-two)
#   docs/src/assets/comparison_nonpow2.png  (throughput relative to FFTW, non-power-of-two / smooth)
#
# It does NOT re-run the benchmarks — re-running is noisy. To capture a fresh measurement, run
# bench/run_compare.jl once (it saves the datapoints), then run this. All plots are RELATIVE to FFTW
# (FFTW = 1.0), so they are clock-independent: the absolute GFLOP/s and CPU clock cancel out.
#
#   julia -O3 --project=bench bench/plot_compare.jl
#
# bench env needs: Plots, JSON.

using Plots, Printf, Statistics
import JSON

const RESULTS = joinpath(@__DIR__, "results", "compare.json")
isfile(RESULTS) || error("no saved datapoints at $RESULTS — run `bench/run_compare.jl` first.")
const DATA = JSON.parsefile(RESULTS)
const RECORDS = DATA["records"]

relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

# (group) → aligned arrays (ns, median time[s] + relative spread per backend), sorted by N.
function load_group(group)
    rg = filter(r -> r["group"] == group, RECORDS)
    byn = Dict{Int, Dict{String, Vector{Float64}}}()
    for r in rg
        n = Int(r["n"])
        get!(byn, n, Dict{String, Vector{Float64}}())[r["method"]] = Float64.(r["times_ns"])
    end
    ns = sort!(collect(keys(byn)))
    med(n, m) = median(byn[n][m]) / 1.0e9      # seconds
    spr(n, m) = relspread(byn[n][m])
    return (ns,
        [med(n, "FFTW") for n in ns], [med(n, "RustFFT") for n in ns], [med(n, "PureFFT") for n in ns],
        [spr(n, "FFTW") for n in ns], [spr(n, "RustFFT") for n in ns], [spr(n, "PureFFT") for n in ns])
end

const COLORS = (fftw = :steelblue, rust = :tomato, pure = :seagreen)
const LABELS = (fftw = "FFTW (MEASURE)", rust = "RustFFT (AVX)", pure = "PureFFT :fast")
pow2_ticks() = ([2^e for e in 6:18], ["2^$e" for e in 6:18])

# All plots RELATIVE to FFTW (clock-independent). Throughput ratio = t_FFTW / t_x (>1 ⇒ faster).
# Ribbon on a ratio r = first-order propagation of both relative spreads: r·√(σx²+σf²).
relthru(t_x, t_f) = t_f ./ t_x
rerr(r, sx, sf) = r .* sqrt.(sx .^ 2 .+ sf .^ 2)
relband(t_x, t_f, sx, sf) = rerr(relthru(t_x, t_f), sx, sf)

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
tickvals, ticklabels = pow2_ticks()

println("Regenerating plots from $RESULTS  ($(get(DATA["meta"], "cpu", "?")), captured $(get(DATA["meta"], "date", "?")))")

# --- power-of-two: throughput + runtime ---
ns, t_fftw, t_rust, t_pure, s_fftw, s_rust, s_pure = load_group("pow2")

p1 = plot(;
    xlabel = "Transform size N",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "FFT throughput relative to FFTW\n(Zen 5, single-thread, ComplexF64, power-of-two; FFTW = 1.0)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    legend = :bottomleft, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p1, [1.0]; label = LABELS.fftw * " (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p1, ns, relthru(t_rust, t_fftw); ribbon = relband(t_rust, t_fftw, s_rust, s_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p1, ns, relthru(t_pure, t_fftw); ribbon = relband(t_pure, t_fftw, s_pure, s_fftw), fillalpha = 0.18, label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p1, joinpath(assets, "comparison.png"))

p2 = plot(;
    xlabel = "Transform size N",
    ylabel = "time relative to FFTW  (t / t_FFTW)",
    title = "FFT runtime relative to FFTW\n(Zen 5, single-thread, ComplexF64, power-of-two; lower = faster)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    legend = :topleft, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p2, [1.0]; label = "FFTW (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p2, ns, t_rust ./ t_fftw; ribbon = rerr(t_rust ./ t_fftw, s_rust, s_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p2, ns, t_pure ./ t_fftw; ribbon = rerr(t_pure ./ t_fftw, s_pure, s_fftw), fillalpha = 0.18, label = LABELS.pure, color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p2, joinpath(assets, "comparison_time.png"))

# --- non-power-of-two (smooth composite) ---
nq, q_fftw, q_rust, q_pure, qs_fftw, qs_rust, qs_pure = load_group("nonpow2")
relmax3 = 1.15 * maximum(vcat(relthru(q_rust, q_fftw), relthru(q_pure, q_fftw)))

p3 = plot(;
    xlabel = "Transform size N (non-power-of-two)",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "FFT throughput relative to FFTW — non-power-of-two (smooth composite)\n(Zen 5, single-thread, ComplexF64; FFTW = 1.0)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    ylims = (0, relmax3), legend = :topright, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p3, [1.0]; label = LABELS.fftw * " (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p3, nq, relthru(q_rust, q_fftw); ribbon = relband(q_rust, q_fftw, qs_rust, qs_fftw), fillalpha = 0.18, label = LABELS.rust, color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p3, nq, relthru(q_pure, q_fftw); ribbon = relband(q_pure, q_fftw, qs_pure, qs_fftw), fillalpha = 0.18, label = "PureFFT :fast (mixed-radix)", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p3, joinpath(assets, "comparison_nonpow2.png"))

for f in ("comparison.png", "comparison_time.png", "comparison_nonpow2.png")
    println("Saved: $(joinpath(assets, f))")
end
