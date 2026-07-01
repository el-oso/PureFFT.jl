module PureFFT

# Pure-Julia FFT. Built in stages to (1) isolate where FFTW's speed comes from
# (algorithm vs LLVM vs language — see REPORT.md) and (2) close the gap toward parity.
#
# Stage 1: scalar radix-2 baseline. Stage 2: mixed-radix (any N). Stage 3: staged radix-2
# (scalar + Base @simd). Stage 4: cache-oblivious recursive. Stage 5+: SoA / @generated
# codelets / cache-blocking / autotuning toward FFTW parity.

export plan_pfft, pfft, pfft!, ipfft, ipfft!
export plan_prfft, plan_pirfft, prfft, pirfft
export ESTIMATE, MEASURE
export REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
export r2r, r2r!, plan_r2r, dct, dct!, idct, idct!, plan_dct, plan_idct
export tryr2r, tryplan_r2r

using MLStyle: @match
import AbstractFFTs

include("contracts.jl")
include("avxradix.jl")
include("avx_mixedradix_plan.jl")
include("cpuinfo.jl")
include("twiddles.jl")
include("butterflies.jl")
include("radix2.jl")
include("mixedradix.jl")
include("staged.jl")
include("codelets.jl")
include("recursive.jl")
include("soa.jl")
include("blocked.jl")
include("radix4.jl")
include("radix4_avx.jl")
include("bluestein.jl")
include("fourstep_codelet.jl")
include("mixedradix_recursive.jl")
include("estimate.jl")
include("plan.jl")
include("rader.jl")
include("autotune.jl")
include("rfft.jl")
include("r2r.jl")
include("abstractfft.jl")
include("ndim_batched.jl")
include("ndim.jl")
include("ndim_real.jl")

# Amortize the fully-unrolled P² codelets' superlinear LLVM compile into the precompile cache (cost moves
# off interactive first-use → ~0). PrecompileTools (not bare `precompile()`) caches the generated NATIVE
# code, which is what these @generated kernels need. Compile cost grows fast with P (large P² dominate),
# so the PRECOMPILED set is capped by a Preferences key. MEASURED cumulative precompile time at each
# cutoff (`Base.compilecache`, this machine), on top of a ~2.7 s no-workload base:
#     cutoff P:   (none)   19       23       29       31
#     precompile: 2.6 s    10.0 s   15.1 s   25.8 s   40.3 s
# Default = 31 (full eligible family; ~40 s added fits the ≤60 s budget — every routed P² then JIT-free).
# Lower it for faster precompile — rarer P² then compile on first use (correct regardless: the autotune
# invariant fix never routes to a slower plan). It only controls which eligible sizes are PRECOMPILED;
# GENPP_MAX_P (autotune.jl) is the separate ELIGIBILITY cap. Set+recompile with:
#   using Preferences; set_preferences!(PureFFT, "genpp_precompile_max_p" => 19)
using PrecompileTools: @compile_workload
using Preferences: @load_preference
const _GENPP_PRECOMPILE_MAX_P = @load_preference("genpp_precompile_max_p", 31)::Int
@compile_workload begin
    for P in (11, 13, 17, 19, 23, 29, 31)
        P <= _GENPP_PRECOMPILE_MAX_P || continue
        n = P * P
        # gen_pp_codelet!{H,M} is keyed on tuple TYPES (H=(P-1)/2, M=P-1), identical fwd/inv — so the
        # forward specialization is the same native code the inverse plan reuses. One direction suffices.
        apply_unnormalized!(GenPPCodeletPlan(ComplexF64, n), zeros(ComplexF64, n))
    end
    # Composite radix-M DIT plans M·P² (autotune-routed family). Cheap vs the P² codelets — they REUSE the
    # gen_pp(P²) native code precompiled above; only the per-(P,M) gather/twiddle/combine is new. Gated by
    # the same cutoff (composite P also ≤ 31). One direction suffices (combine is type-keyed on M).
    for P in (17, 19, 23, 29, 31), M in (2, 4)
        P <= _GENPP_PRECOMPILE_MAX_P || continue
        apply_unnormalized!(GenPPCompositePlan(ComplexF64, M * P * P, P, M), zeros(ComplexF64, M * P * P))
    end
end

end # module
