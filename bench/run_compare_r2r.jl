# DCT/DST (r2r) comparison runner — FFTW vs PureFFT, ALL 8 kinds × a size sweep, F64+F32.
# Saved to bench/results/compare_r2r.json. This is the r2r perf gate + measurement of record.
#
# Small N (8/16/32/64) the targeted kinds (DCT/DST II/III/I) route to the @generated straight-line
# codelet; mid N (128/256/1024/4096) take the FFT-wrap route. For the codelet sizes we ALSO time the
# FFT-wrap plan explicitly ("wrap" = the pre-codelet baseline) so the JSON records before→after.
#
# Interleaved harness (CLAUDE.md rule 6 + the N-D runner): alternate one FFTW batch and one PureFFT
# batch (cancels slow drift in the RATIO); both read a constant source into scratch — no per-rep copy,
# no alloc, no Inf-compounding. FFTW and PureFFT both out-of-place for an apples-to-apples comparison.
#
#   taskset -c 2 julia --project=bench bench/run_compare_r2r.jl
using BenchmarkTools, Statistics, Printf, Dates, LinearAlgebra
import FFTW, JSON
using PureFFT
const ErrorTypes = PureFFT.ErrorTypes

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

const SAMPLES = 4000
const SECONDS = 1.5

# (PureFFT kind, FFTW kind, label)
const KINDS = [
    (REDFT10, FFTW.REDFT10, "DCT-II"),
    (REDFT01, FFTW.REDFT01, "DCT-III"),
    (REDFT11, FFTW.REDFT11, "DCT-IV"),
    (REDFT00, FFTW.REDFT00, "DCT-I"),
    (RODFT10, FFTW.RODFT10, "DST-II"),
    (RODFT01, FFTW.RODFT01, "DST-III"),
    (RODFT11, FFTW.RODFT11, "DST-IV"),
    (RODFT00, FFTW.RODFT00, "DST-I"),
]
const SIZES = [8, 16, 32, 64, 128, 256, 1024, 4096]

# Concrete @noinline batch appliers (CLAUDE.md rule 6: NOT closures — closure indirection would land
# in the timed region and corrupt the tiny-kernel ratio; plan types are concrete → these devirtualize).
# Both sides read a CONSTANT source `s` and write scratch `y`: no per-rep copy, no alloc, AND — unlike
# in-place reps — the data never compounds to ±Inf (transforming Inf hits slow FP microcode that skews
# the tiny-N ratio). R inner reps so the timed region ≫ time_ns() overhead (~20ns) for n=8 (~10ns).
reps_for(n) = clamp(4096 ÷ n, 8, 512)
@noinline function batch_pf!(p, y, s, R)
    @inbounds for _ in 1:R; PureFFT._apply!(p, y, s); end
    return y
end
@noinline function batch_fftw!(pf, y, s, R)
    @inbounds for _ in 1:R; mul!(y, pf, s); end
    return y
end

# Interleaved timing (run_compare_ndim.jl structure): alternate one FFTW batch and one PureFFT batch
# so both see the same machine state (cancels slow drift in the RATIO).
function interleaved_times(pf, pp, x)
    R = reps_for(length(x))
    yf = similar(x); yp = similar(x)
    batch_fftw!(pf, yf, x, R); batch_pf!(pp, yp, x, R)
    tf = Float64[]; tp = Float64[]; el = 0.0
    while el < SECONDS && length(tf) < SAMPLES
        a = time_ns(); batch_fftw!(pf, yf, x, R); d = (time_ns() - a) / R; push!(tf, d); el += d * R / 1e9
        a = time_ns(); batch_pf!(pp, yp, x, R); push!(tp, (time_ns() - a) / R)
    end
    return tf, tp
end

function time_pf(p, x)                                                # the FFT-wrap baseline at codelet sizes
    R = reps_for(length(x))
    yp = similar(x); batch_pf!(p, yp, x, R)
    t = Float64[]; el = 0.0
    while el < SECONDS && length(t) < SAMPLES
        a = time_ns(); batch_pf!(p, yp, x, R); d = (time_ns() - a) / R; push!(t, d); el += d * R / 1e9
    end
    return t
end

println("DCT/DST (r2r) FFTW vs PureFFT — all 8 kinds  |  $(Sys.CPU_NAME)\n")
results = Dict{String, Any}[]

for T in (Float64, Float32)
    println("$T:")
    for (pk, fk, label) in KINDS
        for n in SIZES
            x = randn(T, n)
            pf = FFTW.plan_r2r(copy(x), fk; flags = FFTW.MEASURE)   # out-of-place (constant-src, no Inf)
            pp = PureFFT.plan_r2r(x, pk)                      # current routing (codelet small, wrap mid)
            iscodelet = typeof(pp) <: PureFFT.R2RCodeletPlan

            tf, tp = interleaved_times(pf, pp, x)

            gf = gflops(n, median(tf) / 1e9)
            gp = gflops(n, median(tp) / 1e9)

            # before→after: when the codelet is used, also time the FFT-wrap plan it replaced
            gw = NaN; spw = NaN
            if iscodelet
                pw = ErrorTypes.unwrap(PureFFT._build_r2r(pk, T, n))
                tw = time_pf(pw, x)
                gw = gflops(n, median(tw) / 1e9); spw = relspread(tw)
            end

            ratio = gp / gf
            @printf("  %-8s n=%-5d FFTW %6.1f  PF %6.1f  PF/FFTW=%.2f%s\n",
                label, n, gf, gp, ratio,
                iscodelet ? @sprintf("  [codelet/wrap=%.2f, wrap/FFTW=%.2f]", gp / gw, gw / gf) : "")
            push!(results, Dict(
                "kind" => label, "T" => string(T), "n" => n,
                "fftw_gflops" => gf, "purefft_gflops" => gp,
                "wrap_gflops" => isnan(gw) ? nothing : gw,    # nothing for mid-N (no codelet baseline)
                "route" => iscodelet ? "codelet" : "wrap",
                "fftw_relspread" => relspread(tf), "purefft_relspread" => relspread(tp),
                "wrap_relspread" => isnan(spw) ? nothing : spw,
            ))
        end
    end
    println()
end

outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkdir(outdir)
out = joinpath(outdir, "compare_r2r.json")
open(out, "w") do io
    JSON.print(io, Dict(
        "meta" => Dict(
            "cpu" => Sys.CPU_NAME, "date" => string(Dates.today()),
            "note" => "All 8 r2r kinds; out-of-place batched-rep interleaved (constant src). " *
                      "wrap_gflops = FFT-wrap baseline at codelet sizes (before→after).",
        ),
        "records" => results,
    ), 2)
end
println("\nSaved → $out")
println("Regenerate plot:  julia -O3 --project=bench bench/plot_compare_r2r.jl")
