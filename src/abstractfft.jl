# AbstractFFTs.jl plan interface.
#
# Wraps a PureFFT plan in an `AbstractFFTs.Plan` so PureFFT plugs into the Julia FFT ecosystem:
# `using AbstractFFTs; p = plan_fft(x); p * x`, `p \ y`, `inv(p)`, `mul!`/`ldiv!`, and `ifft`
# (AbstractFFTs derives `plan_ifft` as a `ScaledPlan` over our unnormalized `plan_bfft`).
#
# A PureFFT plan's `apply_unnormalized!` is exactly the unnormalized forward (fft) or backward
# (bfft) transform depending on its `inverse` flag — precisely what AbstractFFTs wants; the 1/N
# normalization for `ifft` is supplied generically by AbstractFFTs via `ScaledPlan`.
#
# NOTE: these methods extend `AbstractFFTs.plan_fft` etc. on plain vectors. If FFTW.jl is also
# loaded, its `StridedVector` methods are more specific and win for `Vector`s (no ambiguity); use
# the `_pure_plan_*` builders below to force the PureFFT path regardless.

import LinearAlgebra

"""
    PureFFTPlanWrapper{T,P} <: AbstractFFTs.Plan{T}

Adapts a PureFFT plan (`inner`, satisfying the `AbstractFFTPlan` contract) to the AbstractFFTs
`Plan{T}` interface, where `T` is the input element type. `inplace` selects `plan_fft` vs
`plan_fft!` semantics; `pinv` caches the inverse plan (left undefined until first `inv`).
"""
mutable struct PureFFTPlanWrapper{T, P} <: AbstractFFTs.Plan{T}
    inner::P
    n::Int
    region::Any
    inplace::Bool
    pinv::AbstractFFTs.Plan{T}
    PureFFTPlanWrapper{T, P}(inner, n, region, inplace) where {T, P} =
        new{T, P}(inner, n, region, inplace)
end

# region for a vector transform is dim 1; canonicalize an Int to a tuple so `normalization`'s
# `prod(sz[r] for r in region)` works.
_region(r::Integer) = (Int(r),)
_region(r) = r
_checkdim1(region) = all(==(1), region) ||
    throw(ArgumentError("PureFFT's AbstractFFTs interface supports 1-D transforms (region over dim 1) only; got $region"))

# Internal builders (force the PureFFT path even when FFTW.jl is loaded).
function _pure_plan_fft(x::AbstractVector{Complex{F}}, region = 1:1; inplace::Bool = false, inverse::Bool = false, flags::PlanRigor = MEASURE) where {F}
    reg = _region(region)
    _checkdim1(reg)
    inner = plan_pfft(Complex{F}, length(x); inverse, variant = :fast, flags)
    return PureFFTPlanWrapper{Complex{F}, typeof(inner)}(inner, length(x), reg, inplace)
end

# AbstractFFTs entry points (forward + unnormalized backward; in-place and out-of-place).
AbstractFFTs.plan_fft(x::AbstractVector{<:Complex}, region; kws...) = _pure_plan_fft(x, region; inplace = false, inverse = false, kws...)
AbstractFFTs.plan_fft!(x::AbstractVector{<:Complex}, region; kws...) = _pure_plan_fft(x, region; inplace = true, inverse = false, kws...)
AbstractFFTs.plan_bfft(x::AbstractVector{<:Complex}, region; kws...) = _pure_plan_fft(x, region; inplace = false, inverse = true, kws...)
AbstractFFTs.plan_bfft!(x::AbstractVector{<:Complex}, region; kws...) = _pure_plan_fft(x, region; inplace = true, inverse = true, kws...)

Base.size(p::PureFFTPlanWrapper) = (p.n,)
AbstractFFTs.fftdims(p::PureFFTPlanWrapper) = p.region

function Base.:*(p::PureFFTPlanWrapper, x::AbstractVector)
    length(x) == p.n || throw(DimensionMismatch("plan length $(p.n) ≠ input $(length(x))"))
    y = p.inplace ? x : copy(x)
    apply_unnormalized!(p.inner, y)
    return y
end

function LinearAlgebra.mul!(y::AbstractVector, p::PureFFTPlanWrapper, x::AbstractVector)
    length(x) == p.n || throw(DimensionMismatch("plan length $(p.n) ≠ input $(length(x))"))
    copyto!(y, x)
    apply_unnormalized!(p.inner, y)
    return y
end

# Inverse: AbstractFFTs wraps in a ScaledPlan, so we return the opposite-direction (unnormalized)
# plan scaled by 1/N. inv(forward)=ifft=(1/N)·bfft; inv(bfft)=(1/N)·fft.
function AbstractFFTs.plan_inv(p::PureFFTPlanWrapper{T}) where {T}
    invinner = plan_pfft(T, p.n; inverse = !plan_inverse(p.inner), variant = :fast)
    invwrap = PureFFTPlanWrapper{T, typeof(invinner)}(invinner, p.n, p.region, p.inplace)
    return AbstractFFTs.ScaledPlan(invwrap, AbstractFFTs.normalization(real(T), (p.n,), p.region))
end
