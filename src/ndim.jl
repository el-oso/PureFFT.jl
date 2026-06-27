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
    # Scratch must hold the largest transpose block used by the dim>1 path: inner*n_d = (∏sz[1:d-1])*sz[d].
    # Size at construction so the hot path never allocates (Task 5 gates zero-alloc apply). The dim-1 path
    # only needs one n_d run, covered by maximum(sz[d]); take the max of both.
    nscratch = maximum(sz[d] for d in dims)
    for d in dims
        d == 1 && continue
        nscratch = max(nscratch, prod(sz[1:d-1]) * sz[d])
    end
    NDPlan{T, D, typeof(plans), N}(dims, plans, sz, Vector{Complex{T}}(undef, nscratch), inverse)
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

# dim d>1: dim d has stride `inner` so its runs aren't contiguous. For each of `outer` slices, transpose
# the inner×n_d block to n_d×inner (so each n_d run is contiguous), apply the 1-D plan to each of the
# `inner` runs in scratch, transpose back into x.
@inline function _apply_dim_transpose!(plan, x::AbstractArray{Complex{T}}, inner::Int, n_d::Int, outer::Int, scratch) where {T}
    need = inner * n_d
    length(scratch) >= need || resize!(scratch, need)   # plan-time scratch is pre-sized; this is a cold fallback
    GC.@preserve x scratch begin
        px = reinterpret(Ptr{Complex{T}}, pointer(x)); ps = pointer(scratch)
        @inbounds for o in 0:(outer-1)
            off = o * need
            _transpose_block!(ps, px + off*sizeof(Complex{T}), inner, n_d)        # scratch[n_d×inner] = block[inner×n_d]ᵀ
            for j in 0:(inner-1)
                apply_unnormalized!(plan, view(scratch, (j*n_d+1):(j*n_d+n_d)))   # n_d contiguous
            end
            _transpose_block!(px + off*sizeof(Complex{T}), ps, n_d, inner)        # transpose back into x
        end
    end
    return x
end

# Complex AoS block transpose N1×N2 → N2×N1 (cache-blocked). NOTE: blocked.jl's `_btranspose!` is SoA
# (separate real/imag arrays) and CANNOT be reused for AoS `Complex` blocks — this is a new, AoS-complex
# transpose. Correctness reference: dst[k2 + N2*k1] = src[k1 + N1*k2] for k1∈0:N1-1, k2∈0:N2-1.
function _transpose_block!(dst::Ptr{Complex{T}}, src::Ptr{Complex{T}}, N1::Int, N2::Int) where {T}
    blk = 32                                          # cache tile (cf. blocked.jl _BTRANSPOSE_BLK); tune later
    @inbounds for j0 in 0:blk:(N2-1), i0 in 0:blk:(N1-1)
        for j in j0:min(j0+blk, N2)-1, i in i0:min(i0+blk, N1)-1
            unsafe_store!(dst, unsafe_load(src, i + N1*j + 1), j + N2*i + 1)
        end
    end
    return
end

# AbstractFFTs surface for N-D arrays.
# AbstractVector methods in abstractfft.jl are more specific → a Vector still routes 1-D (no shadowing).
import LinearAlgebra

AbstractFFTs.plan_fft(x::AbstractArray{<:Complex}, region; kws...)  = _pure_plan_fft_nd(x, region; inverse=false)
AbstractFFTs.plan_fft!(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=false)
AbstractFFTs.plan_bfft(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=true)
AbstractFFTs.plan_bfft!(x::AbstractArray{<:Complex}, region; kws...)= _pure_plan_fft_nd(x, region; inverse=true)

Base.size(p::NDPlan) = p.sz
AbstractFFTs.fftdims(p::NDPlan) = p.dims

Base.:*(p::NDPlan, x::AbstractArray) = apply_unnormalized!(p, copy(x))
LinearAlgebra.mul!(y::AbstractArray, p::NDPlan, x::AbstractArray) = apply_unnormalized!(p, copyto!(y, x))

# NDPlan is not <: AbstractFFTs.Plan, so ScaledPlan(::NDPlan,...) would fail (constructor requires Plan{T}).
# ponytail: minimal scaled-plan wrapper — add AbstractFFTs.Plan inheritance to NDPlan if ecosystem compat needed.
struct _NDScaledPlan{T, P}
    plan::P
    scale::T
end
Base.:*(sp::_NDScaledPlan, x::AbstractArray) = sp.scale .* (sp.plan * x)

function AbstractFFTs.plan_inv(p::NDPlan{T}) where {T}
    ip = _pure_plan_fft_nd(Array{Complex{T}}(undef, p.sz...), p.dims; inverse=!p.inverse)
    _NDScaledPlan(ip, AbstractFFTs.normalization(real(T), p.sz, p.dims))
end

Base.inv(p::NDPlan) = AbstractFFTs.plan_inv(p)

# N-D prefixed entry point; 1-D vector still routes to pfft(::AbstractVector) in plan.jl.
pfft(x::AbstractArray{<:Complex}, dims=1:ndims(x)) = plan_fft(x, dims) * x
