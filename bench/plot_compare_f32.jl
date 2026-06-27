# Float32 benchmark PLOTS — regenerated from SAVED data (bench/results/compare_f32.json, written by
# bench/run_compare_f32.jl). The F32 sibling of plot_compare.jl. Does NOT re-run the benchmarks.
#
#   docs/src/assets/comparison_f32.png          (ComplexF32 throughput relative to FFTW, power-of-two)
#   docs/src/assets/comparison_f32_nonpow2.png  (ComplexF32 throughput relative to FFTW, non-power-of-two)
#   docs/src/assets/comparison_f32_vs_f64.png   (PureFFT ComplexF32 throughput relative to its own ComplexF64)
#
# Relative plots are clock-independent (the absolute GFLOP/s and CPU clock cancel out).
#
#   julia -O3 --project=bench bench/plot_compare_f32.jl
#
# bench env needs: Plots, JSON.

using Plots, Printf
import JSON

const RESULTS = joinpath(@__DIR__, "results", "compare_f32.json")
isfile(RESULTS) || error("no saved datapoints at $RESULTS — run `bench/run_compare_f32.jl` first.")
const DATA = JSON.parsefile(RESULTS)
const RECORDS = DATA["records"]

# (group) → aligned arrays of (gflops, relspread) per method, sorted by N.
function load_group(group)
    rg = filter(r -> r["group"] == group, RECORDS)
    byn = Dict{Int, Dict{String, NTuple{2, Float64}}}()
    for r in rg
        n = Int(r["n"])
        get!(byn, n, Dict{String, NTuple{2, Float64}}())[r["method"]] = (Float64(r["gflops"]), Float64(r["relspread"]))
    end
    ns = sort!(collect(keys(byn)))
    gf(n, m) = byn[n][m][1]; sp(n, m) = byn[n][m][2]
    methods = ("FFTW", "RustFFT", "PureFFT_F32", "PureFFT_F64")
    g = Dict(m => [gf(n, m) for n in ns] for m in methods)
    s = Dict(m => [sp(n, m) for n in ns] for m in methods)
    return ns, g, s
end

const COLORS = (fftw = :steelblue, rust = :tomato, pure = :seagreen, f64 = :slateblue)
# throughput ratio of GFLOP/s (higher = faster); ribbon = first-order propagation of both spreads.
relthru(g_x, g_b) = g_x ./ g_b
relband(g_x, g_b, s_x, s_b) = relthru(g_x, g_b) .* sqrt.(s_x .^ 2 .+ s_b .^ 2)
pow2_ticks(ns) = (ns, ["2^$(Int(round(log2(n))))" for n in ns])

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
println("Regenerating Float32 plots from $RESULTS  ($(get(DATA["meta"], "cpu", "?")), captured $(get(DATA["meta"], "date", "?")))")

# --- power-of-two: ComplexF32 throughput relative to FFTW ---
ns, g, s = load_group("pow2")
tickvals, ticklabels = pow2_ticks(ns)
p1 = plot(;
    xlabel = "Transform size N",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "ComplexF32 FFT throughput relative to FFTW\n(Zen 5, single-thread, power-of-two; FFTW = 1.0)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    legend = :bottomleft, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p1, [1.0]; label = "FFTW Float32 (MEASURE, baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p1, ns, relthru(g["RustFFT"], g["FFTW"]); ribbon = relband(g["RustFFT"], g["FFTW"], s["RustFFT"], s["FFTW"]), fillalpha = 0.18, label = "RustFFT Float32", color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p1, ns, relthru(g["PureFFT_F32"], g["FFTW"]); ribbon = relband(g["PureFFT_F32"], g["FFTW"], s["PureFFT_F32"], s["FFTW"]), fillalpha = 0.18, label = "PureFFT :fast Float32", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p1, joinpath(assets, "comparison_f32.png"))

# --- non-power-of-two: ComplexF32 throughput relative to FFTW ---
nq, gq, sq = load_group("nonpow2")
qtv, qtl = pow2_ticks(nq)
relmax = 1.15 * maximum(vcat(relthru(gq["RustFFT"], gq["FFTW"]), relthru(gq["PureFFT_F32"], gq["FFTW"])))
p2 = plot(;
    xlabel = "Transform size N (non-power-of-two, smooth composite)",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "ComplexF32 FFT throughput relative to FFTW — non-power-of-two\n(Zen 5, single-thread; FFTW = 1.0)",
    xscale = :log2, xticks = (qtv, [string(n) for n in nq]), xrotation = 45,
    ylims = (0, relmax), legend = :topright, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p2, [1.0]; label = "FFTW Float32 (baseline)", color = COLORS.fftw, linewidth = 2, linestyle = :dash)
plot!(p2, nq, relthru(gq["RustFFT"], gq["FFTW"]); ribbon = relband(gq["RustFFT"], gq["FFTW"], sq["RustFFT"], sq["FFTW"]), fillalpha = 0.18, label = "RustFFT Float32", color = COLORS.rust, linewidth = 2, marker = :circle, markersize = 4)
plot!(p2, nq, relthru(gq["PureFFT_F32"], gq["FFTW"]); ribbon = relband(gq["PureFFT_F32"], gq["FFTW"], sq["PureFFT_F32"], sq["FFTW"]), fillalpha = 0.18, label = "PureFFT :fast Float32 (V8f32 tree)", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p2, joinpath(assets, "comparison_f32_nonpow2.png"))

# --- PureFFT ComplexF32 throughput relative to its own ComplexF64 (the half-precision win) ---
p3 = plot(;
    xlabel = "Transform size N (power-of-two)",
    ylabel = "PureFFT Float32 throughput ÷ Float64  (higher = more half-precision win)",
    title = "PureFFT: ComplexF32 vs ComplexF64 throughput\n(Zen 5, single-thread, power-of-two; 2.0 = ideal half-precision speedup)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    legend = :bottomright, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p3, [1.0]; label = "Float64 (baseline)", color = COLORS.f64, linewidth = 2, linestyle = :dash)
hline!(p3, [2.0]; label = "ideal half-precision (2×)", color = :gray, linewidth = 1, linestyle = :dot)
plot!(p3, ns, relthru(g["PureFFT_F32"], g["PureFFT_F64"]); ribbon = relband(g["PureFFT_F32"], g["PureFFT_F64"], s["PureFFT_F32"], s["PureFFT_F64"]), fillalpha = 0.18, label = "ComplexF32 / ComplexF64", color = COLORS.pure, linewidth = 2, marker = :circle, markersize = 4)
savefig(p3, joinpath(assets, "comparison_f32_vs_f64.png"))

for f in ("comparison_f32.png", "comparison_f32_nonpow2.png", "comparison_f32_vs_f64.png")
    println("Saved: $(joinpath(assets, f))")
end
