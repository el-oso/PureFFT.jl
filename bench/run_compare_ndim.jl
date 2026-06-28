# N-D complex FFT runner — FFTW vs PureFFT, ComplexF64+F32, representative 2-D and 3-D shapes.
# FFTW only — RustFFT has no N-D transforms.
# Re-runs the benchmarks ONCE and saves medians + spreads to bench/results/compare_ndim.json.
# Single-thread, in-place, planning excluded. Medians + central-68% spread.
#
#   taskset -c 2 julia --project=bench bench/run_compare_ndim.jl
#
# PureFFT plan built via _pure_plan_fft_nd (NOT plan_fft! — FFTW is more specific and hijacks that).

using BenchmarkTools, LinearAlgebra, Statistics, Printf, Dates
import FFTW, PureFFT, JSON

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1e9
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

const SAMPLES = 800
const SECONDS = 2.5

# Interleaved in-place timing (noise control for the memory-bandwidth-bound N-D ratio):
#  (1) alternate one FFTW rep and one PureFFT rep so both see the same machine state — cancels slow drift
#      in the RATIO (separate measurement blocks let the box drift between them and bias the ratio);
#  (2) in-place reps, NO per-sample copy — the data overflows to ±Inf but transform throughput is identical
#      and far steadier (CLAUDE.md rule 6); both sides are the RAW in-place transform (no API copy).
function interleaved_times(applyf, applyp, x)
    yf = copy(x); yp = copy(x)
    applyf(yf); applyp(yp)                                  # warm / force compile
    tf = Float64[]; tp = Float64[]; el = 0.0
    while el < SECONDS && length(tf) < SAMPLES
        s = time_ns(); applyf(yf); d = time_ns() - s; push!(tf, d); el += d / 1e9
        s = time_ns(); applyp(yp); push!(tp, time_ns() - s)
    end
    return tf, tp
end

# (label, shape, group)
const SHAPES = [
    ("128×128",   (128, 128),      "2d_pow2"),
    ("256×256",   (256, 256),      "2d_pow2"),
    ("512×512",   (512, 512),      "2d_pow2"),
    ("384×384",   (384, 384),      "2d_nonpow2"),
    ("512×384",   (512, 384),      "2d_nonpow2"),
    ("64×64×64",  (64, 64, 64),   "3d_pow2"),
    ("96×96×96",  (96, 96, 96),   "3d_nonpow2"),
    ("48×48×48",  (48, 48, 48),   "3d_pow2"),
    ("240×240",   (240, 240),      "2d_nonpow2"),    # 2^4·3·5 — radix-5 batched (Task 6w)
    ("224×224",   (224, 224),      "2d_nonpow2"),    # 2^5·7   — radix-7 batched
    ("160×160×160", (160, 160, 160), "3d_nonpow2"),  # 2^5·5   — radix-5 batched
    ("112×112×112", (112, 112, 112), "3d_nonpow2"),  # 2^4·7   — radix-7 batched
    ("127×127",   (127, 127),      "2d_prime"),      # 127 prime, 126=2·3²·7 — batched Rader (Task 7f)
    ("251×251",   (251, 251),      "2d_prime"),      # 251 prime, 250=2·5³   — batched Rader
    ("113×113×113", (113, 113, 113), "3d_prime"),    # 113 prime, 112=2⁴·7   — batched Rader
    ("256×127",   (256, 127),      "2d_prime"),      # mixed: strided dim-2 prime (Rader) + pow2 dim-1
]

println("N-D complex FFT: FFTW vs PureFFT (F64 + F32)  |  $(Sys.CPU_NAME)\n")
results = Dict{String, Any}[]

for (T, tname) in ((Float64, "F64"), (Float32, "F32"))
    println("$tname:")
    for (label, sz, group) in SHAPES
        x = randn(Complex{T}, sz...)
        n = prod(sz)
        dims = ntuple(identity, ndims(x))

        pf = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
        # ponytail: _pure_plan_fft_nd bypasses the FFTW method-override on plan_fft!(::StridedArray)
        pp = PureFFT._pure_plan_fft_nd(x, dims; inverse = false)

        # raw in-place transforms (no copy): FFTW in-place plan `*` mutates in place; PureFFT apply_unnormalized!
        applyf = yf -> (pf * yf)
        applyp = yp -> PureFFT.apply_unnormalized!(pp, yp)
        tf, tp = interleaved_times(applyf, applyp, x)

        gf = gflops(n, median(tf) / 1e9)
        gp = gflops(n, median(tp) / 1e9)
        ratio = gp / gf
        @printf("  %-12s  FFTW %6.1f  PureFFT %6.1f  PF/FFTW=%.3f  σ(FFTW)=%.1f%% σ(PF)=%.1f%%\n",
            label, gf, gp, ratio, 100 * relspread(tf), 100 * relspread(tp))

        for (method, t, g) in (("FFTW", tf, gf), ("PureFFT", tp, gp))
            push!(results, Dict(
                "T"         => tname,
                "label"     => label,
                "sz"        => collect(sz),
                "group"     => group,
                "method"    => method,
                "gflops"    => g,
                "median_ns" => median(t),
                "relspread" => relspread(t),
            ))
        end
    end
    println()
end

outdir = joinpath(@__DIR__, "results")
isdir(outdir) || mkdir(outdir)
outpath = joinpath(outdir, "compare_ndim.json")
open(outpath, "w") do io
    JSON.print(io, Dict(
        "meta" => Dict(
            "cpu"    => Sys.CPU_NAME,
            "date"   => string(Dates.today()),
            "metric" => "GFLOP/s (median, 5·n·log2(n) model)",
            "note"   => "N-D c2c; FFTW only (RustFFT has no N-D); single-thread, in-place, planning excluded",
        ),
        "records" => results,
    ), 2)
end
println("saved → $outpath")
