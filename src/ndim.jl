# N-dimensional complex FFT (separable: 1-D FFTs along each transformed dim, reusing 1-D kernels).
# D = number of transformed dims (type parameter so apply @generated-unrolls over it — inner
# `plans` tuple is heterogeneous, so runtime indexing would box, CLAUDE.md rule #1). N = array rank.

struct NDPlan{T, D, P, N} <: AbstractFFTPlan{T}
    dims::NTuple{D, Int}         # transformed dims, sorted + deduped
    plans::P                     # NTuple{D} of inner 1-D plans
    sz::NTuple{N, Int}           # full array shape
    scratch::Vector{Complex{T}}  # reused work buffer (sized to max transformed dim)
    inverse::Bool
end

plan_length(p::NDPlan) = prod(p.sz)
plan_inverse(p::NDPlan) = p.inverse

# canonicalize a region (Int / tuple / range / Colon) over an N-d array → sorted, deduped
# NTuple{D,Int}, validated ⊆ 1:N.
_canon_region(::Colon, N::Int) = ntuple(identity, N)
_canon_region(r::Integer, N::Int) = _canon_region((Int(r),), N)
function _canon_region(r, N::Int)
    t = Tuple(sort!(unique(Int.(collect(r)))))
    isempty(t) && throw(ArgumentError("empty region"))
    all(d -> 1 <= d <= N, t) || throw(ArgumentError("region $r out of bounds for a $N-d array"))
    return t
end

function _pure_plan_fft_nd(x::AbstractArray{Complex{T}, N}, region; inverse::Bool) where {T, N}
    dims = _canon_region(region, N)
    sz = size(x)
    # ponytail: map over dims builds a heterogeneous NTuple — each plan is specialized to its dim size
    plans = map(d -> plan_pfft(Complex{T}, sz[d]; inverse, variant=:fast), dims)
    D = length(dims)
    NDPlan{T, D, typeof(plans), N}(dims, plans, sz, Vector{Complex{T}}(undef, maximum(sz[d] for d in dims)), inverse)
end
