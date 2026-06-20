# PkgBenchmark suite for PureFFT (Julia-internal variants). Run with:
#   using PkgBenchmark; benchmarkpkg("PureFFT")
#   # or compare two commits:  judge("PureFFT", "HEAD", "main")
#
# This covers the Julia variants against each other for regression tracking. The cross-LANGUAGE
# comparisons (vs the rustfft crate, vs FFTW) live in bench/ with a manual time_ns harness that
# is byte-identical to the Rust side — PkgBenchmark/BenchmarkTools have no Rust analog, so they
# can't host that comparison fairly.

using BenchmarkTools
using PureFFT

const SUITE = BenchmarkGroup()

const VARIANTS = (:recursive, :radix4, :radix4simd, :fourstep, :fast)
const SIZES = 2 .^ (6:18)

for v in VARIANTS
    SUITE[string(v)] = BenchmarkGroup()
    for n in SIZES
        v === :fourstep && n < 16 && continue
        x = randn(ComplexF64, n)
        p = plan_pfft(x; variant = v)
        SUITE[string(v)][n] =
            @benchmarkable PureFFT.pfft!(y, $p) setup = (y = copy($x)) evals = 1
    end
end
