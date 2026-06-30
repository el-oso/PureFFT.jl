# Fast-breadth test: does the GENERATED column-packed P² codelet (src/gen/colgen.jl) beat the CURRENT
# autoplan route on UNCOVERED prime-power sizes? Probe — gen codelet is NOT wired into autoplan.
#
#   gen_pp_codelet! over a length-P² buffer at base 0 IS the full size-P² DFT (n1=n2=P), exactly as
#   butterfly25!/49! are the complete B25/B49 transforms. We bench it head-to-head against
#   PureFFT.plan_pfft(:fast) (Bluestein for 121/289/361; AvxMixedRadix for 169) and FFTW.MEASURE.
#
# Run pinned:  taskset -c 2 julia -O3 -t 1 --project bench/colgen_breadth.jl

module ColGenBreadth

using BenchmarkTools, Statistics, Printf, SIMD
import FFTW, PureFFT

include(joinpath(@__DIR__, "..", "src", "avxradix", "kernels.jl"))   # primitives + butterfly25!/49!
include(joinpath(@__DIR__, "..", "src", "gen", "colgen.jl"))         # gen_pp_codelet!
include(joinpath(@__DIR__, "pin_check.jl"))
assert_pinned()
FFTW.set_num_threads(1)

# twiddle bundles (same as the B25/B49 structs build, generalized to any odd prime P)
colbf_tw(P, fwd)    = ntuple(a -> avx_broadcast_twiddle(a, P, fwd), (P - 1) ÷ 2)
chunk_tw(P, fwd)    = ntuple(g -> ntuple(r -> avx_mixedradix_twiddle_chunk(2g - 1, r, P * P, fwd), P - 1),
                             (P - 1) ÷ 2)

# return out[1] so BenchmarkTools cannot dead-code-eliminate the stores (the result is consumed).
@noinline run_gen!(out, inp, tch, tc, tcl) = (gen_pp_codelet!(out, inp, 0, tch, tc, tcl); @inbounds out[1])

gflops(n, t_ns) = 5 * n * log2(n) / (t_ns * 1e-9) / 1e9
const SAMPLES = 3000
const SECONDS = 6

function bench_P(P)
    n = P * P; fwd = true
    x = randn(ComplexF64, n)
    tch = chunk_tw(P, fwd); tc = colbf_tw(P, fwd); tcl = map(avx_lo, tc)

    # (a) bit-exact vs the current route's output (same convention, independent algorithm)
    pp = PureFFT.plan_pfft(x; variant = :fast)
    ref = copy(x); PureFFT.pfft!(ref, pp)
    go = similar(x); run_gen!(go, x, tch, tc, tcl)
    relerr = maximum(abs.(go .- ref)) / maximum(abs.(ref))

    # (b) benchmark gen vs current vs FFTW.MEASURE
    pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
    bgen = @benchmark run_gen!(o, $x, $tch, $tc, $tcl) setup = (o = similar($x)) samples = SAMPLES seconds = SECONDS
    bcur = @benchmark PureFFT.pfft!(y, $pp) setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    bm   = @benchmark $pm * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS

    (; P, n, route = string(typeof(pp).name.name), relerr,
       gen = median(bgen.times), cur = median(bcur.times), fftw = median(bm.times))
end

println("Fast-breadth: generated column-packed P² codelet vs current route vs FFTW.MEASURE")
println("$(Sys.CPU_NAME) | ComplexF64 | single-thread | $SAMPLES samples median\n")
@printf("%-6s %-6s %-16s %-9s %8s %8s %8s   %7s %7s   %s\n",
        "P", "n", "route", "relerr", "gen_ns", "cur_ns", "fftw_ns", "gen/FW", "cur/FW", "gen beats route?")
for P in (5, 7, 11, 13, 17, 19)   # 5,7 are sanity anchors: gen should ≈ the known-good B25/B49 route
    r = bench_P(P)
    @printf("%-6d %-6d %-16s %-9.1e %8.1f %8.1f %8.1f   %7.2f %7.2f   %s\n",
            r.P, r.n, r.route, r.relerr, r.gen, r.cur, r.fftw,
            r.fftw / r.gen, r.fftw / r.cur, r.gen < r.cur ? "YES" : "no")
end

end # module
