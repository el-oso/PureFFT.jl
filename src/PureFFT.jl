module PureFFT

# Pure-Julia FFT. Built in stages to (1) isolate where rustfft/FFTW's speed comes from
# (algorithm vs LLVM vs language — see REPORT.md) and (2) close the gap toward parity.
#
# Stage 1: scalar radix-2 baseline. Stage 2: mixed-radix (any N). Stage 3: staged radix-2
# (scalar + Base @simd). Stage 4: cache-oblivious recursive. Stage 5+: SoA / @generated
# codelets / cache-blocking / autotuning toward FFTW parity.

export plan_pfft, pfft, pfft!, ipfft, ipfft!
export plan_prfft, plan_pirfft, prfft, pirfft

using MLStyle: @match
import AbstractFFTs

include("contracts.jl")
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
include("plan.jl")
include("autotune.jl")
include("rfft.jl")

end # module
