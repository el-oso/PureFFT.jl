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

# Apply each transformed dim in turn. @generated unrolls over D with LITERAL indices so the
# heterogeneous `plans`/`dims` tuples are never indexed by a runtime variable (CLAUDE.md rule #1).
@generated function apply_unnormalized!(p::NDPlan{T, D, P, N}, x::AbstractArray) where {T, D, P, N}
    body = Expr(:block)
    for i in 1:D
        push!(body.args, :(_apply_dim!(p.plans[$i], x, p.dims[$i], p.sz, p.scratch)))
    end
    push!(body.args, :(return x))
    body
end

# Apply the 1-D `plan` along dim `d`. Flat layout: inner = ∏size[1:d-1], n_d = size[d], outer = ∏size[d+1:N].
@inline function _apply_dim!(plan, x::AbstractArray{Complex{T}}, d::Int, sz, scratch) where {T}
    inner = 1; @inbounds for i in 1:(d-1); inner *= sz[i]; end
    n_d = @inbounds sz[d]
    outer = 1; @inbounds for i in (d+1):length(sz); outer *= sz[i]; end
    if inner == 1
        # dim 1: each of `outer` runs of n_d is contiguous ⇒ apply in place on a unit-stride view.
        @inbounds for o in 0:(outer-1)
            apply_unnormalized!(plan, view(x, (o*n_d + 1):(o*n_d + n_d)))
        end
    else
        _apply_dim_transpose!(plan, x, inner, n_d, outer, scratch)   # Task 3
    end
    return x
end
