# DCT/DST (r2r) benchmark PLOTS — regenerated from bench/results/compare_r2r.json (written by
# bench/run_compare_r2r.jl). Does NOT re-run the benchmarks.
#
#   docs/src/assets/comparison_r2r.png   (all 8 kinds, throughput relative to FFTW, F64)
#
# Relative plots are clock-independent (GFLOP/s and CPU clock cancel out). Markers on the small-N
# codelet region (n ≤ 64) are filled; the FFT-wrap mid-N region is open.
#
#   julia -O3 --project=bench bench/plot_compare_r2r.jl
using Plots, Printf
import JSON

const RESULTS = joinpath(@__DIR__, "results", "compare_r2r.json")
isfile(RESULTS) || error("No saved data at $RESULTS — run `bench/run_compare_r2r.jl` first.")
const DATA = JSON.parsefile(RESULTS)
const RECORDS = DATA["records"]

const KINDS = ["DCT-II", "DCT-III", "DST-II", "DST-III", "DCT-I", "DST-I", "DCT-IV", "DST-IV"]
const COLORS = Dict(zip(KINDS, [:seagreen, :darkorange, :steelblue, :crimson, :purple, :goldenrod, :teal, :gray]))

# (ns, ratio, route) per kind, sorted by n, for one element type
function series(kind, T_str)
    recs = sort(filter(r -> r["kind"] == kind && r["T"] == T_str, RECORDS); by = r -> Int(r["n"]))
    ns    = Int[r["n"] for r in recs]
    ratio = Float64[r["purefft_gflops"] / r["fftw_gflops"] for r in recs]
    iscdl = Bool[r["route"] == "codelet" for r in recs]
    return ns, ratio, iscdl
end

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
println("Regenerating r2r plot from $RESULTS  ($(get(DATA["meta"], "cpu", "?")), $(get(DATA["meta"], "date", "?")))")

allns = sort(unique(Int[r["n"] for r in RECORDS]))
tickv = (allns, ["2^$(Int(round(log2(n))))" for n in allns])

p = plot(;
    xlabel = "Transform size N", ylabel = "throughput relative to FFTW  (higher = faster)",
    title = "DCT/DST (r2r) throughput relative to FFTW — all 8 kinds, Float64\n" *
            "($(get(DATA["meta"], "cpu", "?")), single-thread; filled marker = small-N @generated codelet)",
    xscale = :log2, xticks = tickv, xrotation = 45,
    legend = :topleft, size = (900, 560), dpi = 150, margin = 6Plots.mm,
)
hline!(p, [1.0]; label = "FFTW (MEASURE, baseline)", color = :black, linewidth = 1.5, linestyle = :dash)
for kind in KINDS
    ns, ratio, iscdl = series(kind, "Float64")
    isempty(ns) && continue
    c = COLORS[kind]
    plot!(p, ns, ratio; label = kind, color = c, linewidth = 2)
    # filled markers where the codelet is used, open elsewhere
    scatter!(p, ns[iscdl], ratio[iscdl]; color = c, markersize = 6, markerstrokecolor = c, label = "")
    scatter!(p, ns[.!iscdl], ratio[.!iscdl]; color = :white, markersize = 5, markerstrokecolor = c, label = "")
end

out = joinpath(assets, "comparison_r2r.png")
savefig(p, out)
println("Saved: $out")
