# Head-to-head par gate for a kernel change (branch vs master). Benchmarks pfft! on sizes that exercise
# the changed kernels, median ns via BenchmarkTools setup=copy (fresh data → NO overflow-to-NaN artifact).
# Writes "n median_ns" lines to ARGS[1]. Run the SAME size list against this branch AND a master worktree,
# then ratio master÷branch per size (≥0.96 = par). For the binding gate: pin the clock first
# (sudo bench/cpufreq_lock.sh lock — boost off → deterministic; see cpufreq-pin-amd-pstate note), taskset -c 2.
using PureFFT, BenchmarkTools, Statistics
include(joinpath(@__DIR__, "pin_check.jl"))
assert_pinned()   # warn (not abort) if the bench core isn't pinned — note: use `lock` (boost off), which
                  # holds the base clock deterministically; `pin` can false-positive here (boost overrides).
println("PureFFT: ", pathof(PureFFT))
# cb8: 24(B8 V4f), 512(B512W8), 4096(MR8W8). cb9: 81/729(MR9/B9 V4f), 576(MR9W8). controls (no cb8/9): 1024
# (radix-4 pow2 → must be ~1.0), 720 (mixed cb-other).
sizes = [24, 81, 512, 576, 729, 4096, 1024, 720]
open(ARGS[1], "w") do io
    for n in sizes
        x0 = randn(ComplexF64, n)
        p = PureFFT.plan_pfft(x0)
        b = @benchmark PureFFT.pfft!(y, $p) setup = (y = copy($x0)) samples = 3000
        println(io, n, " ", round(median(b.times), digits = 2)); flush(io)
        println("  n=$n  median=", round(median(b.times), digits = 1), " ns")
    end
end
println("done → ", ARGS[1])
