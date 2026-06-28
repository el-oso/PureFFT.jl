# Real N-dimensional FFT (rfft / irfft / brfft), built on the proven 1-D real plans (rfft.jl)
# + the complex N-D engine (ndim.jl). Separable, FFTW/AbstractFFTs convention:
#
#   rfft(x, region):  r2c along FIRST(region) (that dim n → n÷2+1, real→complex) THEN c2c along the
#                     remaining region dims (on the now-complex half-spectrum). first(region) is NOT
#                     sorted — order matters (matches AbstractFFTs.rfft_output_size / FFTW).
#   irfft/brfft:      reverse — c2c⁻¹ along the rest, THEN c2r along first(region) (needs original len d).
#
# Correctness-first cut: the r2c/c2r along the chosen dim uses a reshape-to-(inner,len,outer) +
# per-column strided view of the 1-D real plan (apply_rfft!/apply_irfft!). Bit-exact, not the
# dispatch-free hot path the complex engine has. The c2c-on-the-rest reuses the complex NDPlan directly.

# Own plan (NOT <: NDPlan — the apply is r2c-then-c2c, a different shape). IS <: AbstractFFTs.Plan so
# rfft/brfft/irfft/plan_irfft all derive from the AbstractFFTs generics (ScaledPlan does the irfft
# normalization). T = input eltype (real Tr forward, Complex{Tr} inverse); Tr = real float type.
struct RealNDPlan{T, Tr, D, RP, CP, N} <: AbstractFFTs.Plan{T}
    d::Int                       # r2c/c2r dim = first(region)
    n::Int                       # real length on dim d (even)
    rplan::RP                    # RealFFTPlan (forward) or RealIFFTPlan (inverse)
    cplan::CP                    # NDPlan over the remaining dims, or `nothing` (single-dim region)
    dims::NTuple{D, Int}         # full transformed region: (d, rest...)
    realsz::NTuple{N, Int}       # real-array shape
    cplxsz::NTuple{N, Int}       # half-spectrum shape (dim d → n÷2+1)
    inverse::Bool
    scale::Tr                    # output scale (brfft: n_d; forward: 1)
end

@inline function _prod_before(sz, d)
    p = 1; @inbounds for i in 1:(d - 1); p *= sz[i]; end; p
end
@inline function _prod_after(sz, d)
    p = 1; @inbounds for i in (d + 1):length(sz); p *= sz[i]; end; p
end

# Split a region into the r2c dim (literal first, unsorted) and the c2c rest. Accepts Int / tuple /
# range / Colon. Validates 1 ≤ d ≤ N (rest is validated by the complex engine's _canon_region).
function _split_region(region, N::Int)
    if region isa Colon
        d = 1; rest = ntuple(i -> i + 1, N - 1)
    elseif region isa Integer
        d = Int(region); rest = ()
    else
        r = Int.(collect(region))
        isempty(r) && throw(ArgumentError("empty region"))
        d = first(r)
        rest = Tuple(filter(!=(d), r))
    end
    1 <= d <= N || throw(ArgumentError("region $region: r2c dim $d out of bounds for a $N-d array"))
    return d, rest
end

# ── Plan constructors (force the PureFFT path even when FFTW.jl is loaded) ─────
function _pure_plan_rfft_nd(x::AbstractArray{<:Real, N}, region) where {N}
    Tr = float(eltype(x))
    realsz = size(x)
    d, rest = _split_region(region, N)
    n = realsz[d]
    rplan = plan_prfft(Tr, n)                                   # throws ArgumentError if n is odd
    cplxsz = ntuple(i -> i == d ? n ÷ 2 + 1 : realsz[i], N)
    cplan = isempty(rest) ? nothing :
        _pure_plan_fft_nd(Array{Complex{Tr}}(undef, cplxsz), rest; inverse = false)
    dims = (d, rest...)
    return RealNDPlan{Tr, Tr, length(dims), typeof(rplan), typeof(cplan), N}(
        d, n, rplan, cplan, dims, realsz, cplxsz, false, one(Tr))
end

function _pure_plan_brfft_nd(X::AbstractArray{<:Complex, N}, n::Integer, region) where {N}
    Tr = real(float(eltype(X)))
    cplxsz = size(X)
    d, rest = _split_region(region, N)
    cplxsz[d] == n ÷ 2 + 1 ||
        throw(ArgumentError("brfft: size(X, $d)=$(cplxsz[d]) ≠ n÷2+1=$(n ÷ 2 + 1) for real length n=$n"))
    rplan = plan_pirfft(Tr, n)                                  # throws ArgumentError if n is odd
    realsz = ntuple(i -> i == d ? Int(n) : cplxsz[i], N)
    cplan = isempty(rest) ? nothing :
        _pure_plan_fft_nd(Array{Complex{Tr}}(undef, cplxsz), rest; inverse = true)
    dims = (d, rest...)
    return RealNDPlan{Complex{Tr}, Tr, length(dims), typeof(rplan), typeof(cplan), N}(
        d, Int(n), rplan, cplan, dims, realsz, cplxsz, true, Tr(n))
end

# ── Cores ─────────────────────────────────────────────────────────────────────
# Forward: r2c along dim d (per strided column), then c2c on the rest (in place on Y).
function _rfft_core!(p::RealNDPlan{T, Tr}, Y::AbstractArray, x::AbstractArray) where {T, Tr}
    d = p.d; n = p.n; rsz = p.realsz; h = n ÷ 2 + 1
    inner = _prod_before(rsz, d); outer = _prod_after(rsz, d)
    xf = eltype(x) === Tr ? x : Tr.(x)
    xr = reshape(xf, inner, n, outer)
    Yr = reshape(Y, inner, h, outer)
    @inbounds for o in 1:outer, i in 1:inner
        apply_rfft!(p.rplan, view(xr, i, :, o), view(Yr, i, :, o))
    end
    isnothing(p.cplan) || apply_unnormalized!(p.cplan, Y)
    return Y
end

# Inverse (UNNORMALIZED brfft): c2c⁻¹ on the rest (in place on the owned X), then c2r along dim d.
# apply_irfft! already normalizes dim d; the bfft on the rest is unnormalized (factor ∏rest); the
# stored scale = n_d makes the result the brfft (= irfft · ∏region). ScaledPlan→irfft divides it back.
function _brfft_core!(p::RealNDPlan{T, Tr}, y::AbstractArray, X::AbstractArray) where {T, Tr}
    isnothing(p.cplan) || apply_unnormalized!(p.cplan, X)
    d = p.d; n = p.n; rsz = p.realsz; h = n ÷ 2 + 1
    inner = _prod_before(rsz, d); outer = _prod_after(rsz, d)
    Xr = reshape(X, inner, h, outer)
    yr = reshape(y, inner, n, outer)
    @inbounds for o in 1:outer, i in 1:inner
        apply_irfft!(p.rplan, view(Xr, i, :, o), view(yr, i, :, o))
    end
    s = p.scale
    s == one(Tr) || (y .*= s)
    return y
end

# ── Apply surface ─────────────────────────────────────────────────────────────
function Base.:*(p::RealNDPlan{T, Tr}, x::AbstractArray) where {T, Tr}
    if p.inverse
        Xc = Array{Complex{Tr}}(undef, p.cplxsz); copyto!(Xc, x)   # owned copy (c2c is in place)
        y = Array{Tr}(undef, p.realsz)
        return _brfft_core!(p, y, Xc)
    else
        Y = Array{Complex{Tr}}(undef, p.cplxsz)
        return _rfft_core!(p, Y, x)
    end
end

function LinearAlgebra.mul!(y::AbstractArray, p::RealNDPlan{T, Tr}, x::AbstractArray) where {T, Tr}
    out = p.inverse ? p.realsz : p.cplxsz
    in_ = p.inverse ? p.cplxsz : p.realsz
    size(y) == out && size(x) == in_ ||
        throw(DimensionMismatch("RealNDPlan expects in $in_ → out $out; got $(size(x)) → $(size(y))"))
    if p.inverse
        Xc = Array{Complex{Tr}}(undef, p.cplxsz); copyto!(Xc, x)
        _brfft_core!(p, y, Xc)
    else
        _rfft_core!(p, y, x)
    end
    return y
end

Base.size(p::RealNDPlan) = p.inverse ? p.cplxsz : p.realsz
AbstractFFTs.fftdims(p::RealNDPlan) = p.dims

# ── AbstractFFTs drop-in. plan_rfft + plan_brfft are enough: rfft/brfft/irfft/plan_irfft all derive
# from the AbstractFFTs generics (definitions.jl). NOTE FFTW.jl's StridedArray methods are more
# specific and win when FFTW is loaded — use _pure_plan_*_nd to force PureFFT (as the tests do).
AbstractFFTs.plan_rfft(x::AbstractArray{<:Real}, region; kws...) = _pure_plan_rfft_nd(x, region)
AbstractFFTs.plan_brfft(x::AbstractArray{<:Complex}, d::Integer, region; kws...) = _pure_plan_brfft_nd(x, d, region)

# ── Prefixed convenience (matches pfft(::AbstractArray, dims)); thin wrappers over the plan path.
prfft(x::AbstractArray{<:Real}, region) = _pure_plan_rfft_nd(x, region) * x
function pirfft(X::AbstractArray{<:Complex}, d::Integer, region)
    p = _pure_plan_brfft_nd(X, d, region)
    y = p * X                                                   # unnormalized brfft
    y .*= AbstractFFTs.normalization(real(float(eltype(X))), p.realsz, p.dims)
    return y
end
