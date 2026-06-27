# N-D benchmark PLOTS — regenerated from SAVED data (bench/results/compare_ndim.json,
# written by bench/run_compare_ndim.jl). Does NOT re-run the benchmarks.
#
#   docs/src/assets/comparison_ndim.png   (PureFFT/FFTW ratio per shape, F64 + F32)
#
#   julia -O3 --project=bench bench/plot_compare_ndim.jl

using Plots, Printf
import JSON

const RESULTS = joinpath(@__DIR__, "results", "compare_ndim.json")
isfile(RESULTS) || error("no saved data at $RESULTS — run bench/run_compare_ndim.jl first.")
const DATA = JSON.parsefile(RESULTS)
const RECORDS = DATA["records"]

# For each (T, label) pair return (fftw_gflops, fftw_spread, pf_gflops, pf_spread)
function load_ratios(T_str)
    # index by label
    fftw_g = Dict{String, Float64}(); fftw_s = Dict{String, Float64}()
    pf_g   = Dict{String, Float64}(); pf_s   = Dict{String, Float64}()
    for r in RECORDS
        r["T"] == T_str || continue
        lbl = r["label"]
        if r["method"] == "FFTW"
            fftw_g[lbl] = Float64(r["gflops"]); fftw_s[lbl] = Float64(r["relspread"])
        else
            pf_g[lbl]   = Float64(r["gflops"]); pf_s[lbl]   = Float64(r["relspread"])
        end
    end
    labels = [r["label"] for r in RECORDS if r["T"] == T_str && r["method"] == "FFTW"]
    unique!(labels)
    ratio  = [pf_g[l] / fftw_g[l] for l in labels]
    # first-order error propagation
    ribbon = [ratio[i] * sqrt(pf_s[labels[i]]^2 + fftw_s[labels[i]]^2) for i in eachindex(labels)]
    return labels, ratio, ribbon
end

assets = joinpath(@__DIR__, "..", "docs", "src", "assets")
mkpath(assets)
cpu  = get(DATA["meta"], "cpu", "?")
date = get(DATA["meta"], "date", "?")
println("Regenerating N-D plots from $RESULTS  ($cpu, captured $date)")

lbls64, rat64, rib64 = load_ratios("F64")
lbls32, rat32, rib32 = load_ratios("F32")

xs64 = 1:length(lbls64)
xs32 = 1:length(lbls32)

p = plot(;
    xlabel  = "Shape",
    ylabel  = "PureFFT / FFTW throughput  (higher = faster)",
    title   = "N-D complex FFT throughput relative to FFTW\n($cpu, single-thread, in-place; FFTW = 1.0)",
    legend  = :topright,
    size    = (900, 520), dpi = 150, margin = 6Plots.mm,
    xticks  = (xs64, lbls64), xrotation = 30,
)
hline!(p, [1.0];  label = "FFTW (baseline)",       color = :steelblue, linewidth = 2, linestyle = :dash)
hline!(p, [0.96]; label = "0.96× gate",             color = :gray,      linewidth = 1, linestyle = :dot)
plot!(p, xs64, rat64; ribbon = rib64, fillalpha = 0.20,
    label = "PureFFT ComplexF64", color = :seagreen, linewidth = 2, marker = :circle, markersize = 5)
plot!(p, xs32, rat32; ribbon = rib32, fillalpha = 0.20,
    label = "PureFFT ComplexF32", color = :tomato,   linewidth = 2, marker = :diamond, markersize = 5)

out = joinpath(assets, "comparison_ndim.png")
savefig(p, out)
println("Saved: $out")
