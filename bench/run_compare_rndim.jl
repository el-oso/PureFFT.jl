# N-D REAL FFT runner — FFTW.rfft vs PureFFT, Float64+Float32, representative 2-D and 3-D shapes.
# FFTW only — RustFFT has no N-D transforms. Mirrors run_compare_ndim.jl, adapted for the OUT-OF-PLACE
# r2c transform (real input → complex half-spectrum of size n÷2+1 along first(region)).
# Re-runs the benchmarks ONCE and saves medians + spreads to bench/results/compare_rndim.json.
# Single-thread, planning excluded. Medians + central-68% spread.
#
#   taskset -c 2 julia --project=bench bench/run_compare_rndim.jl
#
# PureFFT plan built via _pure_plan_rfft_nd (NOT plan_rfft — FFTW is more specific and hijacks that).

using BenchmarkTools, LinearAlgebra, Statistics, Printf, Dates
import FFTW, PureFFT, JSON

FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1e9
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)

const SAMPLES = 800
const SECONDS = 2.5

# Interleaved OUT-OF-PLACE timing (noise control for the memory-bandwidth-bound N-D ratio):
#  (1) alternate one FFTW rep and one PureFFT rep so both see the same machine state — cancels slow drift
#      in the RATIO (separate measurement blocks let the box drift between them and bias the ratio);
#  (2) rfft is real→complex out-of-place: both sides mul! into a PREALLOCATED complex half-spectrum buffer
#      (yf / yp), reusing it every rep — no per-sample allocation. Input x (real) is untouched, reused.
function interleaved_times(applyf, applyp)
    applyf(); applyp()                                       # warm / force compile
    tf = Float64[]; tp = Float64[]; el = 0.0
    while el < SECONDS && length(tf) < SAMPLES
        s = time_ns(); applyf(); d = time_ns() - s; push!(tf, d); el += d / 1e9
        s = time_ns(); applyp(); push!(tp, time_ns() - s)
    end
    return tf, tp
end

# (label, shape, group)
const SHAPES = [
    ("256×256",     (256, 256),      "2d_pow2"),
    ("512×512",     (512, 512),      "2d_pow2"),
    ("128×128",     (128, 128),      "2d_pow2"),
    ("384×384",     (384, 384),      "2d_nonpow2"),
    ("240×240",     (240, 240),      "2d_nonpow2"),
    ("64×64×64",    (64, 64, 64),    "3d_pow2"),
    ("96×96×96",    (96, 96, 96),    "3d_nonpow2"),
]

println("N-D REAL FFT: FFTW.rfft vs PureFFT (F64 + F32)  |  $(Sys.CPU_NAME)\n")
results = Dict{String, Any}[]

for (T, tname) in ((Float64, "F64"), (Float32, "F32"))
    println("$tname:")
    for (label, sz, group) in SHAPES
        x = randn(T, sz...)
        region = ntuple(identity, ndims(x))      # rfft along all dims (r2c on dim 1)
        n = prod(sz)

        pf = FFTW.plan_rfft(copy(x), region; flags = FFTW.MEASURE)
        # ponytail: _pure_plan_rfft_nd bypasses the FFTW method-override on plan_rfft(::StridedArray)
        pp = PureFFT._pure_plan_rfft_nd(x, region)

        yf = Array{Complex{T}}(undef, pp.cplxsz...)
        yp = Array{Complex{T}}(undef, pp.cplxsz...)
        applyf = () -> mul!(yf, pf, x)
        applyp = () -> mul!(yp, pp, x)
        tf, tp = interleaved_times(applyf, applyp)

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
outpath = joinpath(outdir, "compare_rndim.json")
open(outpath, "w") do io
    JSON.print(io, Dict(
        "meta" => Dict(
            "cpu"    => Sys.CPU_NAME,
            "date"   => string(Dates.today()),
            "metric" => "GFLOP/s (median, 5·n·log2(n) model)",
            "note"   => "N-D real rfft (r2c on dim 1 + c2c rest); FFTW only; single-thread, out-of-place into preallocated buffer, planning excluded",
        ),
        "records" => results,
    ), 2)
end
println("saved → $outpath")
