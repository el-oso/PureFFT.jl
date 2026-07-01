# Targeted measurement for the radix-9/12 vs-rust gap (task #21, step 1). Times FFTW/RustFFT/PureFFT on
# MR9- and MR12-heavy sizes + controls, mirroring run_compare.jl's method (plan once, @benchmark setup=copy,
# median). Prints PureFFT÷RustFFT and PureFFT÷FFTW ratios + the autoplan ROUTE per size (per
# floors-are-often-bugs: confirm it actually goes through MR9/MR12). Writes bench/results/radix912.json.
# The RATIO is relative → clock-independent, so an unpinned run gives a valid gap signal (pin for the
# post-fix binding gate). Run:  taskset -c 2 julia -O3 -t 1 --project=bench bench/measure_radix912.jl
import FFTW, RustFFT, PureFFT
using BenchmarkTools, Printf, Statistics, Dates
import JSON
include(joinpath(@__DIR__, "pin_check.jl")); assert_pinned()
FFTW.set_num_threads(1)

const SAMPLES = 1000
const SECONDS = 3

# MR9-heavy: 81=9², 729=9³, 6561=9⁴, 576=2⁶·9 (W8), 2916=2²·3⁶.  MR12-heavy: 144=12², 1728=12³, 5184=12³·3,
# 20736=12⁴.  Controls: 512 (radix-8, expect ≥parity), 4096 (pow2 radix-4).
# hunt the ROADMAP-flagged still-under-rust candidates: W8 5-smooth + high-5-power (radix-5/9 shuffle floor)
# + 2^a·5³ (the documented architectural floor) + 9/12-at-W8. Looking for ANY PF/Rust < 0.96.
const SIZES = [2880, 5760, 11520, 23040, 46080, 9216, 92160, 110592,
               2000, 4000, 2025, 10125, 3375, 30375, 50000, 500, 250,
               1296, 46656, 15552, 3888]

route(n) = replace(string(typeof(PureFFT.autoplan(ComplexF64, n))), "PureFFT." => "", "AvxRadix." => "")

function sample_times(n)
    x = randn(ComplexF64, n)
    pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
    pr = RustFFT.plan_fft!(copy(x); rustfft_checks = RustFFT.IgnoreArrayChecks())
    pp = PureFFT.plan_pfft(x; variant = :fast)
    bm = @benchmark $pm * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    br = @benchmark $pr * y setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    bp = @benchmark PureFFT.pfft!(y, $pp) setup = (y = copy($x)) samples = SAMPLES seconds = SECONDS
    median(bm.times), median(br.times), median(bp.times)
end

records = Dict{String, Any}[]
println("radix-9/12 gap probe  |  $(Sys.CPU_NAME)  |  ComplexF64  |  PF=PureFFT\n")
@printf("  %-7s %-34s %8s %8s %8s   %7s %7s\n", "n", "route", "FFTW", "RustFFT", "PF", "PF/Rust", "PF/FFTW")
for n in SIZES
    r = route(n)
    tm, tr, tp = sample_times(n)
    pf_rust = tr / tp   # >1 means PureFFT faster than rust; <0.96 = gap (gate miss)
    pf_fftw = tm / tp
    push!(records, Dict("n" => n, "route" => r, "fftw_ns" => tm, "rust_ns" => tr, "pf_ns" => tp,
                        "pf_over_rust" => pf_rust, "pf_over_fftw" => pf_fftw))
    flag = pf_rust < 0.96 ? "  <-- GAP vs rust" : (pf_fftw < 0.96 ? "  <-- GAP vs FFTW" : "")
    @printf("  %-7d %-34s %8.1f %8.1f %8.1f   %6.3fx %6.3fx%s\n", n, r, tm, tr, tp, pf_rust, pf_fftw, flag)
end

open(joinpath(@__DIR__, "results", "radix912.json"), "w") do io
    JSON.print(io, Dict("meta" => Dict("cpu" => Sys.CPU_NAME, "date" => string(Dates.now()),
        "samples" => SAMPLES, "note" => "PF/Rust and PF/FFTW ratios; >1 = PureFFT faster; <0.96 = gate miss"),
        "records" => records))
end
println("\nSaved → bench/results/radix912.json")
