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

# Median (not min) per-call time for a candidate, applied repeatedly to a scratch buffer. Min rewards
# lucky outliers and mis-ranks candidates (CLAUDE.md rule 6 / perf §15) — the median is robust. Time-
# budgeted (≥ a few ms total, ≥ 8 iters); 7 fixed iterations was too noisy and mis-ranked them. Plan-time
# only, so the sample Vector is fine (not the hot transform path). Inline median to avoid a Statistics dep.
function _besttime(c::AbstractFFTPlan{T}, y::AbstractVector{Complex{T}}) where {T}
    apply_unnormalized!(c, y); apply_unnormalized!(c, y)   # warm up / force compile
    ts = Float64[]; elapsed = 0.0
    while length(ts) < 8 || (elapsed < 3.0e-3 && length(ts) < 500)
        t = @elapsed apply_unnormalized!(c, y)
        push!(ts, t); elapsed += t
    end
    sort!(ts)
    m = length(ts)
    return isodd(m) ? ts[(m + 1) ÷ 2] : (ts[m ÷ 2] + ts[m ÷ 2 + 1]) / 2
end

# Non-power-of-two routing. A dynamically-generated mixed-radix codelet wins for small "smooth"
# sizes (largest prime factor ≤ CODELET_MAX_PRIME) up to CODELET_MAX_N, where its straight-line
# code stays compact and beats Bluestein's three-FFT overhead. Above that, or for a large prime
# factor (whose O(p²) codelet leaf is expensive), Bluestein's O(n log n) chirp-Z wins.
const CODELET_MAX_PRIME = 5
const CODELET_MAX_N = 128

# Generated column-packed prime-square (P²) codelet (GenPPCodeletPlan): one in-register @generated DFT
# that beats Bluestein/AvxMixedRadix on uncovered prime-squares. Gated to P prime in [GENPP_MIN_P,
# GENPP_MAX_P]: smaller prime-squares are already fast (4=pow2, 9=AvxMixedRadix, 25/49=hand B25/B49).
# Upper cap = 31: the fully-unrolled codelet emits O(P²) straight-line SIMD, so LLVM compile is
# superlinear — P=37/41/43 cost 12–23 s first-use for ≤1.7× FFTW (43 is a non-win), not worth it;
# 11..31 all land ≥1.99× FFTW at ≤6.4 s (amortized to ~0 by the PrecompileTools workload in PureFFT.jl).
# Returns the prime P if n=P² qualifies, else nothing.
const GENPP_MIN_P = 11
const GENPP_MAX_P = 31
function _gen_pp_prime(n::Int)
    p = isqrt(n)
    (p * p == n && GENPP_MIN_P <= p <= GENPP_MAX_P && _max_prime_factor(p) == p) ? p : nothing
end

# Generated radix-M DIT over the gen_pp P² codelet (GenPPCompositePlan) for n = M·P². Wins where FFTW /
# the AvxMixedRadix radix-13 tree have NO codelet so the prior route is Bluestein: P ∈ {17,19,23,29,31},
# M ∈ {2,4} — measured 1.1–1.65× FFTW AND RustFFT, beating Bluestein. P ∈ {11,13} EXCLUDED (FFTW / radix-13
# already win); P³ (M=P) NOT wired (O(P⁴) size-P combine loses — measure report). Returns (P,M) or nothing.
const GENPP_COMPOSITE_PRIMES = (17, 19, 23, 29, 31)
const GENPP_COMPOSITE_M = (2, 4)
function _gen_pp_composite(n::Int)
    for M in GENPP_COMPOSITE_M
        n % M == 0 || continue
        q = n ÷ M; p = isqrt(q)
        (p * p == q && p in GENPP_COMPOSITE_PRIMES) && return (p, M)
    end
    return nothing
end

# Rader's algorithm wins for primes with a VERY smooth p-1 (p-1 = 2^a·3^b): its length-(p-1)
# convolution then runs on a fast four-step/radix4avx inner FFT and beats Bluestein's larger
# power-of-two M. Measured: a 5 or 7 factor in p-1 makes Rader LOSE (e.g. n=181, p-1=180=2²·3²·5:
# Rader 4.6 < Bluestein 5.3), so gate strictly at largest-prime(p-1) ≤ 3, p ≥ RADER_MIN_P. Else Bluestein.
const RADER_MIN_P = 128
const RADER_MAX_PM1_PRIME = 3

# Valid four-step splits n = n1·n2: both factors smooth and in [FOURSTEP_MIN_FACTOR,
# FOURSTEP_MAX_FACTOR], with max/min ≤ 4. The balanced one isn't always fastest (codelet efficiency
# varies with R), so `autoplan` times them — but only the _FOURSPLIT_MAX_CANDIDATES most balanced,
# to bound plan-time codelet compilation (each candidate may compile two @generated codelets).
const _FOURSPLIT_MAX_RATIO = 4
const _FOURSPLIT_MAX_CANDIDATES = 5
function _foursplit_candidates(n::Int)
    cands = Tuple{Int, Int}[]
    for n1 in FOURSTEP_MIN_FACTOR:FOURSTEP_MAX_FACTOR
        n % n1 == 0 || continue
        n2 = n ÷ n1
        (FOURSTEP_MIN_FACTOR <= n2 <= FOURSTEP_MAX_FACTOR &&
            max(n1, n2) <= _FOURSPLIT_MAX_RATIO * min(n1, n2) &&
            _max_prime_factor(n1) <= 7 && _max_prime_factor(n2) <= 7) || continue
        push!(cands, (n1, n2))
    end
    # keep the most balanced few (smallest max/min) — bounds compile cost without losing the winner
    sort!(cands; by = c -> max(c...) / min(c...))
    return length(cands) > _FOURSPLIT_MAX_CANDIDATES ? cands[1:_FOURSPLIT_MAX_CANDIDATES] : cands
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

# A few balanced small-factor factorizations to try for the recursive mixed-radix plan.
function _recursive_candidates(n::Int)
    cands = Vector{Int}[]
    for maxf in (16, 24, 36, 48)   # wider range → fewer, larger factors for big n (fewer memory passes)
        f = _recursive_factors(n; maxf = maxf)
        (isnothing(f) || length(f) < 2 || f in cands) && continue
        push!(cands, f)
    end
    return cands
end

# Fastest smooth-composite plan: the autotuned 2-factor four-step vs a few recursive multi-factor
# factorizations. The recursive path wins for large n (where the four-step needs huge, register-
# spilling codelets, or has no valid split and would fall to Bluestein); the four-step wins for
# smaller n. Times them and keeps the fastest. Returns nothing if neither applies.
function _best_smooth_plan(::Type{Complex{T}}, n::Int; inverse::Bool) where {T}
    y = randn(Complex{T}, n)
    best = _best_foursplit_plan(Complex{T}, n; inverse)
    bt = isnothing(best) ? Inf : _besttime(best, y)
    for facs in _recursive_candidates(n)
        p = RecursiveMixedRadixPlan(Complex{T}, facs; inverse)
        t = _besttime(p, y)
        if t < bt
            bt = t; best = p
        end
    end
    return best
end

# Score a candidate for `autoplan`: its median per-call time (via `_besttime`), or `Inf` for an
# inapplicable (`nothing`) candidate. Two concrete methods — NOT a runtime branch on the value — so timing
# the candidates with `map(_score, plans::Tuple)` stays type-stable + dispatch-free (dispatch resolves
# statically per tuple slot from the element's concrete type), unlike iterating an `AbstractFFTPlan[]`
# vector, where `_besttime(c, y)` over the abstract element forces a dynamic dispatch (and is trim-hostile).
@inline _score(::Nothing, y) = Inf
@inline _score(p::AbstractFFTPlan, y) = _besttime(p, y)

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
        # prime with smooth p-1 → Rader (length-(p-1) convolution beats Bluestein's larger pow2 M). This
        # is a distinct algorithm (a convolution), so it short-circuits rather than joining the timing below.
        if ni >= RADER_MIN_P && _max_prime_factor(ni) == ni && _max_prime_factor(ni - 1) <= RADER_MAX_PM1_PRIME
            return RaderPlan(Complex{T}, ni; inverse)
        end
        # Candidates as a STATIC TUPLE — not an `AbstractFFTPlan[]` vector. A tuple keeps each element's
        # concrete type, so timing them with `map(_score, plans)` below is type-stable, dispatch-free, and
        # trim-compatible. `nothing` marks an inapplicable candidate (scored `Inf`). All are timed and the
        # fastest kept: a compact codelet can be ~2× slower than the four-step (n=90: 6.4 vs 13 GFLOP/s);
        # W=8 wins only where it beats W=4. Construction is outside `_besttime`'s timed region.
        codelet = (ni <= CODELET_MAX_N && _max_prime_factor(ni) <= CODELET_MAX_PRIME) ?
            CodeletPlan(Complex{T}, n; inverse) : nothing
        # Additive candidate: the generated column-packed P² codelet (Float64-only). Timed with the rest and
        # kept by `argmin` only where it wins — so it CANNOT regress any other size (the additive invariant).
        genpp = (T === Float64 && !isnothing(_gen_pp_prime(ni))) ?
            GenPPCodeletPlan(Complex{T}, ni; inverse) : nothing
        # Additive candidate: the generated radix-M DIT over gen_pp for large-prime-square composites M·P²
        # (Float64-only). Same additive invariant — timed with the rest, kept only where it wins.
        genppc_pm = T === Float64 ? _gen_pp_composite(ni) : nothing
        genppc = isnothing(genppc_pm) ? nothing :
            GenPPCompositePlan(Complex{T}, ni, genppc_pm[1], genppc_pm[2]; inverse)
        # Invariant guard: a prime-square (or M·P² composite) otherwise falls to the UNTIMED Bluestein
        # fallback (the smooth/Avx candidates are all `nothing` for these). Without timing it, GenPP/GenPPC
        # could win the tuple yet be slower than Bluestein → a regression. So when (and ONLY when) a GenPP
        # candidate competes, time Bluestein too and let `argmin` pick the genuine fastest. No Bluestein-
        # timing cost on the general non-pow2 path.
        bluestein = (isnothing(genpp) && isnothing(genppc)) ? nothing : BluesteinPlan(Complex{T}, ni; inverse)
        plans = (
            codelet,
            genpp,
            genppc,
            bluestein,
            _best_smooth_plan(Complex{T}, ni; inverse),
            AvxMixedRadixPlan(Complex{T}, ni; inverse),
            AvxMixedRadixPlanW8(Complex{T}, ni; inverse),
        )
        y = randn(Complex{T}, ni)
        scores = map(p -> _score(p, y), plans)            # NTuple{7,Float64} — concrete, unrolled, no dispatch
        all(isinf, scores) && return BluesteinPlan(Complex{T}, n; inverse)   # large prime factor → chirp-Z
        return something(plans[argmin(scores)])
    end
    # Power-of-two: same static-tuple timing. Radix4Avx / Radix4 / recursive always apply; FourStep and the
    # monolithic B256/B512 + 8xn tree (rustfft scheme) only for n ≥ 256 (else `nothing`, scored `Inf`). The
    # W=8 tree (`AvxMixedRadixPlanW8`) is the *only* monolith path available for `ComplexF32` (the W=4
    # `AvxMixedRadixPlan` is Float64-only) — it covers the W=8-clean pow2 sizes (2^(6+3a)) and is timed here
    # so odd-power F32 sizes that the Radix4Avx 256-bit base-32 handles below the 0.96× gate can route to it.
    plans = (
        Radix4AvxPlan(Complex{T}, n; inverse),
        Radix4Plan(Complex{T}, n; inverse),
        plan_pfft(Complex{T}, n; inverse, variant = :recursive),
        n >= 256 ? FourStepPlan(Complex{T}, n; inverse) : nothing,
        n >= 256 ? AvxMixedRadixPlan(Complex{T}, n; inverse) : nothing,
        n >= 256 ? AvxMixedRadixPlanW8(Complex{T}, n; inverse) : nothing,
    )
    y = randn(Complex{T}, Int(n))
    scores = map(p -> _score(p, y), plans)
    best = something(plans[argmin(scores)])
    return AutoPlan{T, typeof(best)}(best)
end
