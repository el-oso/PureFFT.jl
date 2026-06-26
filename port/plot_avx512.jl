# AVX-512 (W=8) vs AVX2 (W=4) vs RustFFT PLOT — regenerated from SAVED data.
#
# Reads bench/results/avx512.json (written by port/run_avx512.jl) and regenerates
# docs/src/assets/avx512_nonpow2.png. Does NOT re-run the benchmark. Plot is RELATIVE to PureFFT W=4
# (W=4 = 1.0), so it is clock-independent. Central line = median; ribbon = robust central-68%
# (q84−q16)/2 spread (same as the main comparison plots).
#
#   julia -O3 --project=bench port/plot_avx512.jl

using Statistics, Plots
import JSON

const RESULTS = joinpath(@__DIR__, "..", "bench", "results", "avx512.json")
isfile(RESULTS) || error("no saved datapoints at $RESULTS — run `port/run_avx512.jl` first.")
const DATA = JSON.parsefile(RESULTS)

relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)
gf(n, t) = 5 * n * log2(n) / t   # t in ns → GFLOP/s

byn = Dict{Int, Dict{String, Vector{Float64}}}()
for r in DATA["records"]
    n = Int(r["n"])
    get!(byn, n, Dict{String, Vector{Float64}}())[r["method"]] = Float64.(r["times_ns"])
end
sizes = sort!(collect(keys(byn)))
g(method) = [gf(n, median(byn[n][method])) for n in sizes]   # GFLOP/s per size
s(method) = [relspread(byn[n][method]) for n in sizes]
g8, g4, gr = g("W8"), g("W4"), g("rust")
S8, S4, SR = s("W8"), s("W4"), s("rust")

# Relative to PureFFT W=4 (W=4 = 1.0): throughput ratio = g_x / g_4; ribbon propagates both spreads.
r8 = g8 ./ g4
rr = gr ./ g4
b8 = r8 .* sqrt.(S8 .^ 2 .+ S4 .^ 2)
br = rr .* sqrt.(SR .^ 2 .+ S4 .^ 2)
ylo = 0.95 * min(minimum(r8), minimum(rr), 1.0)
yhi = 1.05 * max(maximum(r8), maximum(rr), 1.0)

p = plot(;
    xlabel = "N (Butterfly64 · radix-12^k, non-power-of-two)",
    ylabel = "throughput relative to PureFFT W=4  (higher = faster)",
    title = "AVX-512 (W=8) vs RustFFT, relative to PureFFT W=4 (same tree)\n(Zen 5, single-thread, ComplexF64; W=4 = 1.0)",
    xscale = :log2, xticks = (sizes, string.(sizes)), legend = :bottomright,
    size = (800, 500), dpi = 150, margin = 5Plots.mm, ylims = (ylo, yhi),
)
hline!(p, [1.0]; label = "PureFFT W=4 (AVX2, baseline)", color = :steelblue, lw = 2, linestyle = :dash)
plot!(p, sizes, rr; ribbon = br, fillalpha = 0.18, label = "RustFFT (AVX2)", color = :tomato, lw = 2, marker = :circle, ms = 5)
plot!(p, sizes, r8; ribbon = b8, fillalpha = 0.18, label = "PureFFT W=8 (AVX-512)", color = :seagreen, lw = 2, marker = :circle, ms = 5)
out = joinpath(@__DIR__, "..", "docs", "src", "assets", "avx512_nonpow2.png")
savefig(p, out)
println("Saved: $out")
