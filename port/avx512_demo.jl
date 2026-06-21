# AVX-512 (W=8) vs AVX2 (W=4) demo on non-pow2 — same decomposition (Butterfly64 · radix-12^k), so it
# isolates the vector-width effect. Uses the real src kernels: AvxMixedRadixPlanW8 (W=8) vs an equivalent
# W=4 faithful tree built from AvxRadix internals, vs RustFFT. Plots GFLOP/s to docs/src/assets/avx512_nonpow2.png.
#
# After the @nexprs store-loop unroll (which removed the runtime-tuple-indexing regression at large n),
# W=8 beats W=4 across L1→L3 and is at/above RustFFT parity — see docs/src/performance.md §16.
#
# Run: taskset -c 2 julia -O3 --project=bench port/avx512_demo.jl
using PureFFT, FFTW, Printf, Statistics, Plots
P = PureFFT; AR = PureFFT.AvxRadix

const LIB = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "rust", "target", "release", "librustfft_bench.so")
rpl(n) = ccall((:rfft_plan, LIB), Ptr{Cvoid}, (Csize_t,), n)
rpr(h, d, n) = ccall((:rfft_process, LIB), Cvoid, (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), h, d, n)
# median + relative σ (std/median) so the plot can show σ ribbons like the main comparison plots
medσ(f) = (for _ in 1:20; f(); end; ts = Float64[]; for _ in 1:151; t = time_ns(); for _ in 1:20; f(); end; push!(ts, (time_ns()-t)/20); end; (median(ts), std(ts) / median(ts)))
gf(n, t) = 5 * n * log2(n) / t
w4tree(k) = (t = AR.B64(true); for _ in 1:k; t = AR.MR12(t, true); end; AR.RPlan(t))   # W=4 same decomposition

sizes = [768, 9216, 110592]
g8 = Float64[]; g4 = Float64[]; gr = Float64[]; e8 = Float64[]; e4 = Float64[]; er = Float64[]
for (n, k) in zip(sizes, (1, 2, 3))
    p8 = P.AvxMixedRadixPlanW8(ComplexF64, n); r4 = w4tree(k)
    x = randn(ComplexF64, n); ref = FFTW.fft(x)
    y = copy(x); P.apply_unnormalized!(p8, y); rel = maximum(abs.(y .- ref)) / maximum(abs.(ref))
    b8 = copy(x); b4 = copy(x); rb = copy(x); h = rpl(n)
    (m8, s8) = medσ(() -> P.apply_unnormalized!(p8, b8)); (m4, s4) = medσ(() -> AR.applyplan!(r4, b4)); GC.@preserve rb ((mr, sr) = medσ(() -> rpr(h, rb, n)))
    push!(g8, gf(n, m8)); push!(g4, gf(n, m4)); push!(gr, gf(n, mr))
    push!(e8, gf(n, m8) * s8); push!(e4, gf(n, m4) * s4); push!(er, gf(n, mr) * sr)   # GFLOP/s σ = gflops·relσ
    @printf("n=%-7d rel=%.0e  W8 %.1f  W4 %.1f  rust %.1f GF   W8/W4=%.2f  rust:W8=%.2f\n", n, rel, gf(n,m8), gf(n,m4), gf(n,mr), m4/m8, mr/m8)
end
p = plot(; xlabel = "N (Butterfly64 · radix-12^k, non-power-of-two)", ylabel = "GFLOP/s",
    title = "AVX-512 (W=8) vs AVX2 (W=4) vs RustFFT on non-pow2\n(Zen 5, single-thread, ComplexF64, same decomposition)",
    xscale = :log2, xticks = (sizes, string.(sizes)), legend = :topright, size = (800, 500), dpi = 150,
    margin = 5Plots.mm, ylims = (0, 1.1 * maximum(vcat(g8, g4, gr))))
plot!(p, sizes, gr; ribbon = er, fillalpha = 0.18, label = "RustFFT (AVX2)", color = :tomato, lw = 2, marker = :circle, ms = 5)
plot!(p, sizes, g4; ribbon = e4, fillalpha = 0.18, label = "PureFFT W=4 (AVX2, same tree)", color = :steelblue, lw = 2, marker = :circle, ms = 5)
plot!(p, sizes, g8; ribbon = e8, fillalpha = 0.18, label = "PureFFT W=8 (AVX-512)", color = :seagreen, lw = 2, marker = :circle, ms = 5)
out = joinpath(@__DIR__, "..", "docs", "src", "assets", "avx512_nonpow2.png"); savefig(p, out)
println("Saved: ", out)
