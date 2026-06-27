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

const SAMPLES = 400
const SECONDS = 3.0

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

        tf = (@benchmark $pf * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS).times
        tp = (@benchmark mul!(y, $pp, $x) setup = (y = similar($x)) samples = SAMPLES seconds = SECONDS).times

        gf = gflops(n, median(tf) / 1e9)
        gp = gflops(n, median(tp) / 1e9)
        ratio = gp / gf
        @printf("  %-12s  FFTW %6.1f  PureFFT %6.1f  PF/FFTW=%.3f  σ(PF)=%.1f%%\n",
            label, gf, gp, ratio, 100 * relspread(tp))

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
