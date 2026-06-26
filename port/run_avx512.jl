# AVX-512 (W=8) vs AVX2 (W=4) vs RustFFT RUNNER — saves full per-sample datapoints.
#
# Same decomposition (Butterfly64 · radix-12^k) so it isolates the vector-width effect: the real src
# kernels AvxMixedRadixPlanW8 (W=8) vs an equivalent W=4 faithful tree (AvxRadix internals) vs RustFFT.
# This is the ONE step that re-runs; it writes every sample to bench/results/avx512.json so the plot can
# be regenerated from the saved data (port/plot_avx512.jl), never by re-running.
#
# Run: taskset -c 2 julia -O3 --project=bench port/run_avx512.jl
# (builds the RustFFT cdylib first if cargo is present and it's missing).

using PureFFT, FFTW, Printf, Statistics, Dates
import JSON
P = PureFFT
AR = PureFFT.AvxRadix

const LIB = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "rust", "target", "release", "librustfft_bench.so")
if !isfile(LIB) && Sys.which("cargo") !== nothing
    @info "building RustFFT cdylib"
    run(Cmd(`cargo build --release`; dir = joinpath(@__DIR__, "..", "bench", "rustfft_compare", "rust")))
end
isfile(LIB) || error("RustFFT cdylib not found at $LIB (and cargo unavailable to build it).")
rpl(n) = ccall((:rfft_plan, LIB), Ptr{Cvoid}, (Csize_t,), n)
rpr(h, d, n) = ccall((:rfft_process, LIB), Cvoid, (Ptr{Cvoid}, Ptr{ComplexF64}, Csize_t), h, d, n)

# Tiny kernels: median over 151 samples of 20-inner-rep in-place averages via time_ns (the PureFFT
# benchmarking rule — NOT a BenchmarkTools closure). Returns the raw per-sample ns times.
function sample_times(f)
    for _ in 1:20
        f()
    end
    ts = Float64[]
    for _ in 1:151
        t = time_ns()
        for _ in 1:20
            f()
        end
        push!(ts, (time_ns() - t) / 20)
    end
    return ts
end

gf(n, t) = 5 * n * log2(n) / t   # t in ns → GFLOP/s
w4tree(k) = (t = AR.B64(true); for _ in 1:k; t = AR.MR12(t, true); end; AR.RPlan(t))   # W=4 same decomposition

const SIZES = [768, 9216, 110592]
const KS = (1, 2, 3)

records = Dict{String, Any}[]
println("AVX-512 W=8 vs AVX2 W=4 vs RustFFT  |  $(Sys.CPU_NAME)  |  ComplexF64, single-thread\n")
for (n, k) in zip(SIZES, KS)
    p8 = P.AvxMixedRadixPlanW8(ComplexF64, n)
    r4 = w4tree(k)
    x = randn(ComplexF64, n)
    ref = FFTW.fft(x)
    y = copy(x)
    P.apply_unnormalized!(p8, y)
    rel = maximum(abs.(y .- ref)) / maximum(abs.(ref))   # correctness sanity (must be ~1e-15)
    b8 = copy(x); b4 = copy(x); rb = copy(x); h = rpl(n)
    t8 = sample_times(() -> P.apply_unnormalized!(p8, b8))
    t4 = sample_times(() -> AR.applyplan!(r4, b4))
    GC.@preserve rb (tr = sample_times(() -> rpr(h, rb, n)))
    for (method, t) in (("W8", t8), ("W4", t4), ("rust", tr))
        push!(records, Dict("n" => n, "method" => method, "times_ns" => t))
    end
    @printf("n=%-7d rel=%.0e   W8 %.1f  W4 %.1f  rust %.1f GF\n",
        n, rel, gf(n, median(t8)), gf(n, median(t4)), gf(n, median(tr)))
end

data = Dict(
    "meta" => Dict(
        "cpu" => Sys.CPU_NAME, "julia" => string(VERSION), "date" => string(Dates.now()),
        "flop_model" => "5*N*log2(N)", "methods" => ["W8", "W4", "rust"],
        "tree" => "Butterfly64 · radix-12^k (same decomposition; isolates vector width)",
        "units" => "times_ns are per-transform times in nanoseconds",
        "note" => "AVX-512 W=8 vs AVX2 W=4 vs RustFFT; plot is relative to W=4 (clock-independent)",
    ),
    "records" => records,
)
outdir = joinpath(@__DIR__, "..", "bench", "results")
mkpath(outdir)
out = joinpath(outdir, "avx512.json")
open(out, "w") do io
    JSON.print(io, data)
end
println("\nSaved datapoints → $out  ($(length(records)) records)")
println("Now regenerate the plot:  julia -O3 --project=bench port/plot_avx512.jl")
