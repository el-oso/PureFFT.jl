# Plan + public interface. A plan holds the size, direction, chosen kernel variant, and
# precomputed twiddles. Execution dispatches on the variant tag (a `Val`), so Stage 3's
# SIMD kernels slot in as extra `_execute!` methods without touching callers.
#
# Inverse transforms reuse the same machinery with an inverse twiddle table, then apply
# the 1/n normalization here (AbstractFFTs `ifft` convention).

struct PureFFTPlan{T, V} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    variant::V
    tw::Vector{Complex{T}}                  # radix-2 twiddle table (:scalar variant)
    factors::Vector{Int}                    # radix factorization (mixed-radix variants)
    stages::Vector{Vector{Complex{T}}}      # per-stage twiddles (staged + recursive variants)
    scratch::Vector{Complex{T}}             # out-of-place buffer (:recursive variant)
end

# AbstractFFTPlan contract implementation (see src/contracts.jl). Explicit return types so
# the precompile-time @verify's inference matches the declared contract.
plan_length(p::PureFFTPlan)::Int = p.n
plan_inverse(p::PureFFTPlan)::Bool = p.inverse
apply_unnormalized!(p::PureFFTPlan, x::AbstractVector) = _execute!(x, p, p.variant)

# Variants that run the SIMD-friendly staged radix-2 (power-of-two only).
const STAGED_VARIANTS = (:staged, :base)
# Variants that need per-stage (per-level) twiddles; all power-of-two only.
const POW2_VARIANTS = (STAGED_VARIANTS..., :recursive)

"""
    plan_pfft(Complex{T}, n; inverse=false, variant=:scalar) -> PureFFTPlan
    plan_pfft(x; ...) -> PureFFTPlan

Build a plan for length-`n` complex transforms. `variant` selects the kernel:

  * `:scalar`     — Stage 1 radix-2 baseline (requires `n` a power of two).
  * `:mixedradix` — Stage 2 mixed-radix (any `n`, including primes).
  * `:staged`     — Stage 3 scalar staged radix-2 reference (power of two).
  * `:base`       — Stage 3 Base-Julia `@simd` staged radix-2 (power of two).
  * `:recursive`  — Stage 4 cache-oblivious recursive radix-2 with generated codelets (power of two).
  * `:soa`        — Stage 5 split-layout (SoA) recursive FFT (power of two); see [`SoAPlan`](@ref).
  * `:radix4`     — faithful port of rustfft's Radix4 (power of two); see [`Radix4Plan`](@ref).
  * `:fourstep`   — Stage 7 cache-blocked four-step (power of two, n ≥ 16); see [`FourStepPlan`](@ref).
  * `:fast`       — autotuned: builds candidate plans, times them, keeps the fastest.
"""
function plan_pfft(
        ::Type{Complex{T}}, n::Integer; inverse::Bool = false, variant::Symbol = :scalar
    ) where {T}
    nostage = Vector{Complex{T}}[]
    noscratch = Complex{T}[]
    if variant === :soa
        return SoAPlan(Complex{T}, n; inverse)
    elseif variant === :radix4
        return Radix4Plan(Complex{T}, n; inverse)
    elseif variant === :radix4simd
        return Radix4SoAPlan(Complex{T}, n; inverse)
    elseif variant === :radix4avx
        return Radix4AvxPlan(Complex{T}, n; inverse)
    elseif variant === :fourstep
        return FourStepPlan(Complex{T}, n; inverse)
    elseif variant === :fast
        return autoplan(Complex{T}, n; inverse)
    elseif variant === :scalar
        ispow2(n) ||
            throw(ArgumentError(":scalar supports power-of-two sizes only; got n=$n"))
        tw = twiddle_table(Complex{T}, n; inverse)
        return PureFFTPlan{T, Val{:scalar}}(
            Int(n), inverse, Val(:scalar), tw, Int[], nostage, noscratch
        )
    elseif variant in POW2_VARIANTS
        ispow2(n) ||
            throw(ArgumentError(":$variant supports power-of-two sizes only; got n=$n"))
        stages = staged_twiddles(Complex{T}, n; inverse)
        scratch = variant === :recursive ? Vector{Complex{T}}(undef, Int(n)) : noscratch
        return PureFFTPlan{T, Val{variant}}(
            Int(n), inverse, Val(variant), Complex{T}[], Int[], stages, scratch
        )
    else
        factors = factorize(n)
        return PureFFTPlan{T, Val{variant}}(
            Int(n), inverse, Val(variant), Complex{T}[], factors, nostage, noscratch
        )
    end
end

plan_pfft(x::AbstractVector{Complex{T}}; kw...) where {T} =
    plan_pfft(Complex{T}, length(x); kw...)

# --- variant dispatch -------------------------------------------------------
_execute!(x, p::PureFFTPlan, ::Val{:scalar}) = radix2_dit!(x, p.tw)

function _execute!(x, p::PureFFTPlan, ::Val{:mixedradix})
    copyto!(x, mixedradix(x, p.factors, p.inverse))
    return x
end

_execute!(x, p::PureFFTPlan, ::Val{:staged}) = radix2_staged!(x, p.stages)
_execute!(x, p::PureFFTPlan, ::Val{:base}) = radix2_base_simd!(x, p.stages)

function _execute!(x, p::PureFFTPlan, ::Val{:recursive})
    recursive_fft!(x, p.scratch, p.stages, p.inverse)
    return x
end

_execute!(x, p::PureFFTPlan, ::Val{V}) where {V} = error("unknown PureFFT variant :$V")

function _pfft_run!(x::AbstractVector{Complex{T}}, p) where {T}
    n = plan_length(p)::Int
    length(x) == n || throw(DimensionMismatch("plan length $n ≠ input $(length(x))"))
    apply_unnormalized!(p, x)
    if plan_inverse(p)
        invn = inv(T(n))
        @inbounds @simd for i in eachindex(x)
            x[i] *= invn
        end
    end
    return x
end

"""
    pfft!(x, plan) -> x

Apply `plan` to `x` in place. Direction (forward/inverse) is fixed by the plan. Works for **any**
type satisfying the [`AbstractFFTPlan`](@ref) contract — even one that does not subtype it —
selected by TypeContracts' `interface_trait` Holy-Trait dispatch (hasmethod-based, juliac-safe).
"""
pfft!(x::AbstractVector{<:Complex}, p) =
    _pfft_dispatch!(interface_trait(AbstractFFTPlan, typeof(p)), x, p)

_pfft_dispatch!(::Implemented{AbstractFFTPlan}, x, p) = _pfft_run!(x, p)
_pfft_dispatch!(::NotImplemented{AbstractFFTPlan}, x, p) = throw(
    ArgumentError(
        "$(typeof(p)) does not satisfy the AbstractFFTPlan contract " *
            "(needs plan_length, plan_inverse, apply_unnormalized!)",
    ),
)

"""
    pfft(x; kw...) -> Vector

Out-of-place forward transform. Keyword args forwarded to [`plan_pfft`](@ref).
"""
function pfft(x::AbstractVector{Complex{T}}; kw...) where {T}
    y = copy(x)
    return pfft!(y, plan_pfft(y; kw...))
end

"""
    ipfft(x; kw...) / ipfft!(x; kw...)

Inverse transform (includes the 1/n normalization).
"""
ipfft(x::AbstractVector{<:Complex}; kw...) = pfft(x; inverse = true, kw...)

function ipfft!(x::AbstractVector{<:Complex}; kw...)
    return pfft!(x, plan_pfft(x; inverse = true, kw...))
end

# Precompile-time contract enforcement (zero runtime cost; eliminated by the trimmer).
# The interface methods are generic over the type parameters, so verifying one concrete
# instantiation per precision proves method existence + return types for all variants.
@verify PureFFTPlan{Float64, Val{:recursive}}
@verify PureFFTPlan{Float32, Val{:base}}
@verify Radix4Plan{Float64}
@verify Radix4Plan{Float32}
@verify Radix4SoAPlan{Float64}
@verify Radix4SoAPlan{Float32}
@verify Radix4AvxPlan{Float64}
@verify Radix4AvxPlan{Float32}
