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

"""
    autoplan(Complex{T}, n; inverse=false) -> AbstractFFTPlan

Pick the fastest available kernel for length `n` by timing candidates. Non-power-of-two falls
back to mixed-radix; power-of-two times `:recursive` against the four-step (for `n ≥ 256`).
"""
function autoplan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    if !ispow2(n)
        return plan_pfft(Complex{T}, n; inverse, variant = :mixedradix)
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
