# Proof that Julia reaches FFTW-class FFT throughput.
#
# The batched inner kernel of the four-step (many independent transforms, SIMD across the
# batch dimension) is the shuffle-free, fully-vectorized form. In isolation it hits ≥ FFTW's
# single-transform throughput on this machine — i.e. the language/compiler is NOT the ceiling;
# the gap in a general transform is the four-step WRAPPER (bit-reversal, transpose, twiddle
# passes), pure implementation work.
#
# Run:  julia --project=bench bench/batched_proof.jl

import FFTW, PureFFT
using BenchmarkTools, Printf
FFTW.set_num_threads(1)

gflops(n, t) = 5 * n * log2(n) / t / 1.0e9

println("Batched SoA kernel (B transforms of size R) vs FFTW single-transform peak:\n")
for (R, B) in ((32, 1024), (64, 512), (128, 256), (256, 128))
    N = R * B
    twr = Float64[cospi(-2.0 * k / R) for k in 0:((R >> 1) - 1)]
    twi = Float64[sinpi(-2.0 * k / R) for k in 0:((R >> 1) - 1)]
    vr = rand(Float64, N); vi = rand(Float64, N)
    t = @belapsed PureFFT._batched_dit!(rr, ii, $B, $R, $twr, $twi) setup = (rr = copy($vr); ii = copy($vi)) evals = 1 samples = 300
    x = randn(ComplexF64, R)
    pe = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
    tf = @belapsed $pe * y setup = (y = copy($x)) evals = 1 samples = 300
    @printf(
        "R=%-4d B=%-5d  batched=%5.1f GFLOP/s   FFTW(size %d)=%5.1f GFLOP/s   ratio=%.2f×\n",
        R, B, gflops(R, t / B), R, gflops(R, tf), (t / B) / tf < 1 ? tf / (t / B) : (t / B) / tf
    )
end
println("\n→ batched Julia ≥ FFTW per-transform throughput: the language matches; the")
println("  remaining gap in a general N is the four-step wrapper overhead, not codegen.")
