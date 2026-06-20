# Stage 9: a tiny plan autotuner (FFTW-`MEASURE`-lite).
#
# Different kernels win at different sizes (recursive at small n, cache-blocked four-step at
# medium/large n). Rather than guess, `autoplan` builds the viable candidates, times each on a
# real buffer at plan time, and wraps the fastest in an `AutoPlan`. Selection happens once;
# `apply_unnormalized!` then dispatches straight to the chosen kernel (type-stable, zero-alloc).

"""
    AutoPlan{T,P} <: AbstractFFTPlan{T}

Wrapper holding the fastest concrete plan chosen by [`autoplan`](@ref). Forwards the
`AbstractFFTPlan` contract to `inner`.
"""
struct AutoPlan{T, P <: AbstractFFTPlan{T}} <: AbstractFFTPlan{T}
    inner::P
end

plan_length(p::AutoPlan)::Int = plan_length(p.inner)
plan_inverse(p::AutoPlan)::Bool = plan_inverse(p.inner)
apply_unnormalized!(p::AutoPlan, x::AbstractVector) = apply_unnormalized!(p.inner, x)

# Best (minimum) per-call time for a candidate, applied repeatedly to a scratch buffer.
function _besttime(c::AbstractFFTPlan{T}, y::AbstractVector{Complex{T}}) where {T}
    apply_unnormalized!(c, y)               # warm up / force compile
    best = Inf
    for _ in 1:7
        t = @elapsed apply_unnormalized!(c, y)
        best = min(best, t)
    end
    return best
end

# Non-power-of-two routing. A dynamically-generated mixed-radix codelet wins for small "smooth"
# sizes (largest prime factor ≤ CODELET_MAX_PRIME) up to CODELET_MAX_N, where its straight-line
# code stays compact and beats Bluestein's three-FFT overhead. Above that, or for a large prime
# factor (whose O(p²) codelet leaf is expensive), Bluestein's O(n log n) chirp-Z wins.
const CODELET_MAX_PRIME = 5
const CODELET_MAX_N = 128

# Rader's algorithm wins for primes with a VERY smooth p-1 (p-1 = 2^a·3^b): its length-(p-1)
# convolution then runs on a fast four-step/radix4avx inner FFT and beats Bluestein's larger
# power-of-two M. Measured: a 5 or 7 factor in p-1 makes Rader LOSE (e.g. n=181, p-1=180=2²·3²·5:
# Rader 4.6 < Bluestein 5.3), so gate strictly at largest-prime(p-1) ≤ 3, p ≥ RADER_MIN_P. Else Bluestein.
const RADER_MIN_P = 128
const RADER_MAX_PM1_PRIME = 3

# All valid four-step splits n = n1·n2 (both factors smooth and in [FOURSTEP_MIN_FACTOR,
# FOURSTEP_MAX_FACTOR]). The balanced one isn't always fastest (the batched codelet's efficiency
# varies with R), so `autoplan` times them.
function _foursplit_candidates(n::Int)
    cands = Tuple{Int, Int}[]
    for n1 in FOURSTEP_MIN_FACTOR:FOURSTEP_MAX_FACTOR
        n % n1 == 0 || continue
        n2 = n ÷ n1
        (FOURSTEP_MIN_FACTOR <= n2 <= FOURSTEP_MAX_FACTOR &&
            _max_prime_factor(n1) <= 7 && _max_prime_factor(n2) <= 7) || continue
        push!(cands, (n1, n2))
    end
    return cands
end

# Build a FourStepCodeletPlan for each candidate split, time it, and keep the fastest (FFTW
# MEASURE-style). Returns `nothing` if no valid split. Plan-time only; first use of a given factor
# compiles its @generated codelet (cached thereafter). Measured up to ~26 % over the balanced split.
function _best_foursplit_plan(::Type{Complex{T}}, n::Int; inverse::Bool) where {T}
    cands = _foursplit_candidates(n)
    isempty(cands) && return nothing
    y = randn(Complex{T}, n)
    best = FourStepCodeletPlan(Complex{T}, cands[1][1], cands[1][2]; inverse)
    bt = _besttime(best, y)
    for k in 2:length(cands)
        p = FourStepCodeletPlan(Complex{T}, cands[k][1], cands[k][2]; inverse)
        t = _besttime(p, y)
        t < bt && (bt = t; best = p)
    end
    return best
end

"""
    autoplan(Complex{T}, n; inverse=false) -> AbstractFFTPlan

Pick the fastest available kernel for length `n`. Power-of-two times `:recursive` against the
four-step (for `n ≥ 256`). Non-power-of-two uses a generated mixed-radix [`CodeletPlan`] for small
smooth sizes (largest prime ≤ $CODELET_MAX_PRIME, n ≤ $CODELET_MAX_N), else Bluestein (chirp-Z) —
both far faster than the allocating recursive mixed-radix.
"""
function autoplan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    if !ispow2(n)
        ni = Int(n)
        if ni <= CODELET_MAX_N && _max_prime_factor(ni) <= CODELET_MAX_PRIME
            return CodeletPlan(Complex{T}, n; inverse)
        end
        # prime with smooth p-1 → Rader (length-(p-1) convolution beats Bluestein's larger pow2 M)
        if ni >= RADER_MIN_P && _max_prime_factor(ni) == ni && _max_prime_factor(ni - 1) <= RADER_MAX_PM1_PRIME
            return RaderPlan(Complex{T}, ni; inverse)
        end
        fsp = _best_foursplit_plan(Complex{T}, ni; inverse)   # smooth composite → autotuned four-step
        isnothing(fsp) || return fsp
        return BluesteinPlan(Complex{T}, n; inverse)          # large prime factor → chirp-Z
    end
    # candidate kernels (all power-of-two); time each on a real buffer, keep the fastest.
    cands = AbstractFFTPlan{T}[
        Radix4AvxPlan(Complex{T}, n; inverse),
        Radix4Plan(Complex{T}, n; inverse),
        plan_pfft(Complex{T}, n; inverse, variant = :recursive),
    ]
    n >= 256 && push!(cands, FourStepPlan(Complex{T}, n; inverse))
    y = randn(Complex{T}, Int(n))
    best = cands[1]
    bt = _besttime(best, y)
    for c in cands[2:end]
        t = _besttime(c, y)
        t < bt && (bt = t; best = c)
    end
    return AutoPlan{T, typeof(best)}(best)
end
