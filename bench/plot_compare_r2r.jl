# DCT/DST benchmark PLOTS — regenerated from bench/results/compare_r2r.json (written by
# bench/run_compare_r2r.jl). Does NOT re-run the benchmarks.
#
#   docs/src/assets/comparison_r2r.png   (DCT-II throughput relative to FFTW, F64+F32)
#
# Relative plots are clock-independent (GFLOP/s and CPU clock cancel out).
#
#   julia -O3 --project=bench bench/plot_compare_r2r.jl
#
# bench env needs: Plots, JSON.

using Plots, Printf
import JSON

const RESULTS = joinpath(@__DIR__, "results", "compare_r2r.json")
isfile(RESULTS) || error("No saved data at $RESULTS — run `bench/run_compare_r2r.jl` first.")
const DATA = JSON.parsefile(RESULTS)
const RECORDS = DATA["records"]

# Build (ns, fftw_gflops, purefft_gflops, purefft_relspread) arrays per type, sorted by n.
function load_type(T_str)
    recs = filter(r -> r["T"] == T_str, RECORDS)
    isempty(recs) && error("No records for T=$T_str")
    sorted = sort(recs; by = r -> Int(r["n"]))
    ns    = Int[r["n"] for r in sorted]
    gf    = Float64[r["fftw_gflops"] for r in sorted]
    gp    = Float64[r["purefft_gflops"] for r in sorted]
    sp    = Float64[r["purefft_relspread"] for r in sorted]
    sf    = haskey(first(sorted), "fftw_relspread") ?
                Float64[r["fftw_relspread"] for r in sorted] : zeros(length(ns))
    return ns, gf, gp, sp, sf
end

# Ratio + propagated ribbon (first-order, both spreads).
rel(gp, gf) = gp ./ gf
relband(gp, gf, sp, sf) = rel(gp, gf) .* sqrt.(sp .^ 2 .+ sf .^ 2)

pow2_ticks(ns) = (ns, ["2^$(Int(round(log2(n))))" for n in ns])

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
println("Regenerating r2r plots from $RESULTS  ($(get(DATA["meta"], "cpu", "?")), $(get(DATA["meta"], "date", "?")))")

ns64, gf64, gp64, sp64, sf64 = load_type("Float64")
ns32, gf32, gp32, sp32, sf32 = load_type("Float32")
# All sizes should be identical across types; use F64 for ticks.
tickvals, ticklabels = pow2_ticks(ns64)

p1 = plot(;
    xlabel = "Transform size N",
    ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "DCT-II (REDFT10) throughput relative to FFTW\n($(get(DATA["meta"], "cpu", "Zen 5")), single-thread; FFTW = 1.0)",
    xscale = :log2, xticks = (tickvals, ticklabels), xrotation = 45,
    legend = :bottomleft, size = (800, 500), dpi = 150, margin = 5Plots.mm,
)
hline!(p1, [1.0]; label = "FFTW (MEASURE, baseline)", color = :steelblue,
    linewidth = 2, linestyle = :dash)
plot!(p1, ns64, rel(gp64, gf64);
    ribbon = relband(gp64, gf64, sp64, sf64), fillalpha = 0.18,
    label = "PureFFT Float64", color = :seagreen, linewidth = 2,
    marker = :circle, markersize = 4)
plot!(p1, ns32, rel(gp32, gf32);
    ribbon = relband(gp32, gf32, sp32, sf32), fillalpha = 0.18,
    label = "PureFFT Float32", color = :darkorange, linewidth = 2,
    marker = :diamond, markersize = 4)

out = joinpath(assets, "comparison_r2r.png")
savefig(p1, out)
println("Saved: $out")
