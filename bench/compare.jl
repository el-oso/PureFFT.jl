# Benchmark harness: FFTW vs RustFFT vs PureFFT variants on power-of-two complex FFTs.
#
# Fairness controls (the whole point of the investigation):
#   * single-threaded everywhere (FFTW threads = 1; rustfft plans are single-thread);
#   * planning excluded — every plan built once, only the apply is timed;
#   * in-place, on a fresh copy per sample so repeated transforms don't blow up the data;
#   * FFTW measured at BOTH ESTIMATE and MEASURE, since that flip is the usual reason
#     "rustfft beats FFTW" appears or disappears.
#
# Run:  julia --project=bench bench/compare.jl

import FFTW, RustFFT, PureFFT
using BenchmarkTools, Printf

FFTW.set_num_threads(1)
BLAS_INFO = "single-threaded"

relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
gflops(n, t) = 5 * n * log2(n) / t / 1.0e9      # standard radix-2 FFT flop count

# Each method is (name, make_plan(x) -> apply!::(y -> nothing-ish)).
function methods_for(x::Vector{Complex{T}}) where {T}
    ms = Pair{String, Function}[]

    pe = FFTW.plan_fft!(copy(x); flags = FFTW.ESTIMATE)
    push!(ms, "FFTW-ESTIMATE" => (y -> pe * y))

    pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)   # overwrites its argument
    push!(ms, "FFTW-MEASURE" => (y -> pm * y))

    pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
    push!(ms, "RustFFT" => (y -> pr * y))

    for v in (:recursive, :fourstep, :fast)
        p = PureFFT.plan_pfft(x; variant = v)
        push!(ms, "PureFFT-$v" => (y -> PureFFT.pfft!(y, p)))
    end
    return ms
end

function run(; T = ComplexF64, sizes = 2 .^ (6:18))
    println("PureFFT benchmark  |  $(Sys.CPU_NAME)  |  $T  |  FFTW $BLAS_INFO")
    println("flop model: 5·N·log2(N)\n")
    rows = []
    for n in sizes
        x = randn(T, n)
        ref = FFTW.fft(x)
        ms = methods_for(x)
        @printf("n = %-8d\n", n)
        @printf("  %-16s %12s %10s %10s %12s\n", "method", "time", "GFLOP/s", "vs EST", "relerr")
        base_t = Ref(0.0)
        for (name, apply!) in ms
            t = @belapsed $apply!(y) setup = (y = copy($x)) evals = 1 samples = 400
            y = copy(x); apply!(y)
            err = relerr(y, ref)
            name == "FFTW-ESTIMATE" && (base_t[] = t)
            sp = base_t[] / t
            tstr = t < 1.0e-6 ? @sprintf("%.1f ns", t * 1.0e9) : @sprintf("%.2f µs", t * 1.0e6)
            @printf("  %-16s %12s %10.2f %9.2fx %12.1e\n", name, tstr, gflops(n, t), sp, err)
            push!(rows, (; n, name, t, gflops = gflops(n, t), speedup = sp, err))
        end
        println()
    end
    return rows
end

run()
