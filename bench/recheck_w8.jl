# floors-are-often-bugs recheck of the W8 radix-9 "shuffle floor" (110592=0.85×, 46080=0.92× vs rust).
# For each size, time BOTH PureFFT routes explicitly — W8 (AvxMixedRadixPlanW8) and W4 (AvxMixedRadixPlan) —
# plus rust + FFTW. If W4 ≥ W8 here, autoplan mis-selected W8 (a fixable selection bug, not a floor). If W8
# is genuinely the fastest PureFFT route but still < rust, it's the real AVX-512 radix-9 shuffle floor.
import FFTW, RustFFT, PureFFT
using BenchmarkTools, Printf, Statistics
include(joinpath(@__DIR__, "pin_check.jl")); assert_pinned()
FFTW.set_num_threads(1)
const P = PureFFT

const SIZES = [110592, 46080, 55296, 27648, 221184, 9216, 23040]   # W8-heavy 3-smooth (+ two that were OK)

med(f, x) = median((@benchmark $f(y) setup=(y=copy($x)) samples=800 seconds=3).times)
apply!(p) = y -> P.apply_unnormalized!(p, y)

println("W8-vs-W4 recheck  |  pinned  |  PF=PureFFT; ratios are RustFFT_ns / route_ns (>1 = PF faster)\n")
@printf("  %-8s %6s  %8s %8s %8s   %7s %7s   %s\n", "n", "auto", "W4", "W8", "Rust", "W8/Rust", "W4/Rust", "verdict")
for n in SIZES
    x = randn(ComplexF64, n)
    autop = P.plan_pfft(x; variant = :fast)
    isw8 = occursin("W8", string(typeof(autop)))
    pw8 = P.AvxMixedRadixPlanW8(ComplexF64, n)
    pw4 = P.AvxMixedRadixPlan(ComplexF64, n)
    pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
    tr = median((@benchmark $pr * y setup=(y=copy($x)) samples=800 seconds=3).times)
    t8 = isnothing(pw8) ? NaN : med(apply!(pw8), x)
    t4 = isnothing(pw4) ? NaN : med(apply!(pw4), x)
    r8 = tr / t8; r4 = tr / t4
    verdict = (!isnan(t4) && !isnan(t8) && t4 < t8) ? "W4 FASTER — autoplan picked $(isw8 ? "W8 (BUG?)" : "W4 ok")" :
              (r8 < 0.96 ? "W8 best-PF but < rust → real floor" : "OK")
    @printf("  %-8d %6s  %8.0f %8.0f %8.0f   %6.3fx %6.3fx   %s\n", n, isw8 ? "W8" : "W4", t4, t8, tr, r8, r4, verdict)
end
