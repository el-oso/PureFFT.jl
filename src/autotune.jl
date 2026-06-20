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

"""
    autoplan(Complex{T}, n; inverse=false) -> AbstractFFTPlan

Pick the fastest available kernel for length `n`. Power-of-two times `:recursive` against the
four-step (for `n ≥ 256`). Non-power-of-two uses a generated mixed-radix [`CodeletPlan`] for small
smooth sizes (largest prime ≤ $CODELET_MAX_PRIME, n ≤ $CODELET_MAX_N), else Bluestein (chirp-Z) —
both far faster than the allocating recursive mixed-radix.
"""
function autoplan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    if !ispow2(n)
        if n <= CODELET_MAX_N && _max_prime_factor(Int(n)) <= CODELET_MAX_PRIME
            return CodeletPlan(Complex{T}, n; inverse)
        end
        return BluesteinPlan(Complex{T}, n; inverse)
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
