# ESTIMATE-style fast-plan classifier. `_estimate_plan` picks ONE plan by size-class heuristic (no timing),
# so autoplan(...; flags=ESTIMATE) compiles ~1 tree instead of ~7 → ms first-call vs seconds. Mirrors FFTW
# ESTIMATE/MEASURE. Uncertainty → `nothing` → autoplan falls back to the MEASURE competition. See
# docs/superpowers/specs/2026-07-01-estimate-fast-plan-design.md.

@enum PlanRigor ESTIMATE MEASURE

# Structural size-class pick, reusing autoplan's exact routing predicates so ESTIMATE and MEASURE agree on
# WHICH plan applies (they differ only in whether the choice among applicable candidates is timed). pow2
# returns the AutoPlan-wrapped Radix4Avx (matching autoplan's pow2 return); other classes return the raw
# plan (matching autoplan's non-pow2 return). Never throws — returns `nothing` on any uncertainty.
function _estimate_plan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ni = Int(n)
    if ispow2(ni)
        p = Radix4AvxPlan(Complex{T}, ni; inverse)
        return AutoPlan{T, typeof(p)}(p)
    end
    # large prime with smooth p-1 → Rader (autoplan's exact short-circuit)
    if ni >= RADER_MIN_P && _max_prime_factor(ni) == ni && _max_prime_factor(ni - 1) <= RADER_MAX_PM1_PRIME
        return RaderPlan(Complex{T}, ni; inverse)
    end
    if T === Float64
        !isnothing(_gen_pp_prime(ni)) && return GenPPCodeletPlan(Complex{T}, ni; inverse)
        gppc = _gen_pp_composite(ni)
        !isnothing(gppc) && return GenPPCompositePlan(Complex{T}, ni, gppc[1], gppc[2]; inverse)
    end
    # 2·3·5-smooth → AvxMixedRadix (W4); plan_tree may still decline a smooth size → nothing → fallback
    if _max_prime_factor(ni) <= 5
        p = AvxMixedRadixPlan(Complex{T}, ni; inverse)
        !isnothing(p) && return p
    end
    return nothing   # unclassified → MEASURE fallback
end
