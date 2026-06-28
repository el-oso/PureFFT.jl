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

# Per-dim descriptors. Each transformed dim is routed to ONE of these CONCRETE types; the apply loop
# dispatches on the descriptor type (multiple methods of `_apply_dim!`), so the hot path never branches
# on a non-concrete field (CLAUDE.md rule 5). They form a heterogeneous NTuple in `plans` and are only
# ever indexed with LITERAL indices inside the @generated apply (rule 1).
#   Dim1Plan         — dim 1: each column is a unit-stride contiguous run, apply the 1-D plan in place.
#   BatchedDim1      — dim 1, small F32 only: batch transforms ACROSS the trailing dims (fills the SIMD
#                      width FFTW-style) — transpose-pack a chunk, run the batched kernel, transpose back.
#   BatchedDim       — pow2 dim d>1: batched radix-8 column kernel, NO transpose (ndim_batched.jl).
#   BatchedSmoothDim — 2^a·3^b·5^c·7^d dim d>1: batched mixed-radix (radix-3/5/7 + radix-8/4/2) kernel, NO transpose.
#   TransposeDim     — other non-pow2 dim d>1: transpose → 1-D plan → transpose-back (the fallback).
struct Dim1Plan{P}
    plan::P
end
# dim 1, batched across the trailing dims. bp is a BatchPlan8 (pow2 n1) or BatchPlanMR (2^a·3^b n1).
struct BatchedDim1{T, BP}
    bp::BP
    stage::Vector{Complex{T}}    # n1 × M staging buffer (M = transforms per chunk), reused per chunk
    n1::Int
    M::Int                       # transforms per chunk (multiple of W=L÷2, ≤ outer)
end
struct BatchedDim{T, L}
    d::Int
    bp::BatchPlan8{T, L}
end
struct BatchedSmoothDim{T, L}
    d::Int
    bp::BatchPlanMR{T, L}
end
struct TransposeDim{P}
    d::Int
    plan::P
end

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

# Route one transformed dim to its descriptor: dim-1 → Dim1Plan; pow2 d>1 → BatchedDim (no transpose);
# 2^a·3^b·5^c·7^d d>1 → BatchedSmoothDim (mixed-radix batched, no transpose); else → TransposeDim. Each branch
# returns a distinct CONCRETE type (the heterogeneous tuple's element types stay concrete ⇒ the
# @generated apply specializes fully).
# Route dim-1 to the batched-across-trailing-dims kernel ONLY where the per-column Dim1Plan underfills the
# SIMD register AND batching beats the transpose-pack overhead. Both gates are Float32 only (F64 per-column
# already ≥ parity) with ≥1 full vector group (outer ≥ W):
#   • 2^a·3^b lengths (BatchPlanMR): per-column can't fill the F32 width (radix-3 codelets) ⇒ batching is a
#     big win at any small n1 (measured 48³ 2.6×, 96³ 2.4×, 384² 1.6× on the dim-1 region). Cap n1 ≤ 384.
#   • pow2 lengths (BatchPlan8): the radix-8 codelet ALREADY vectorizes F32 per single transform, so batching
#     only helps small n1 with many trailing transforms (64³ 1.07×); for n1 ≥ 128 the extra transpose-pack
#     traffic makes it a net loss (128² 0.92×, 256² 0.45×, 512² 0.31×). Cap pow2 at n1 ≤ 64.
# Thresholds chosen by direct dim-1-region measurement (task 6o, scratchpad/dim1_measure.jl).
@inline function _use_batched_dim1(::Type{T}, n1::Int, outer::Int) where {T}
    # NB _is_smooth_2a3 is TRUE for pure pow2 too, so the smooth branch must exclude pow2 (else 256/512
    # would route here and regress) — mirror _mk_dim's pow2-before-smooth precedence.
    T === Float32 && outer >= (_batch_lanes(T) >> 1) &&
        ((ispow2(n1) && n1 <= 64) || (!ispow2(n1) && _is_smooth_2a3(n1) && n1 <= 384))
end

function _mk_dim(::Type{Complex{T}}, d::Int, sz; inverse::Bool) where {T}
    if d == 1
        n1 = sz[1]
        outer = prod(@inbounds(sz[i]) for i in 2:length(sz); init=1)
        if _use_batched_dim1(T, n1, outer)
            W = _batch_lanes(T) >> 1
            M = clamp((8192 ÷ n1) & ~(W - 1), W, outer)          # ~L2-resident chunk, multiple of W
            bp = ispow2(n1) ? BatchPlan8(T, n1; forward=!inverse) : BatchPlanMR(T, n1; forward=!inverse)
            BatchedDim1{T, typeof(bp)}(bp, Vector{Complex{T}}(undef, n1 * M), n1, M)
        else
            Dim1Plan(plan_pfft(Complex{T}, n1; inverse, variant=:fast))
        end
    elseif ispow2(sz[d])
        BatchedDim(d, BatchPlan8(T, sz[d]; forward=!inverse))
    elseif _is_smooth_2a3(sz[d])
        BatchedSmoothDim(d, BatchPlanMR(T, sz[d]; forward=!inverse))
    else
        TransposeDim(d, plan_pfft(Complex{T}, sz[d]; inverse, variant=:fast))
    end
end

function _pure_plan_fft_nd(x::AbstractArray{Complex{T}, N}, region; inverse::Bool) where {T, N}
    dims = _canon_region(region, N)
    sz = size(x)
    # ponytail: map over dims builds a heterogeneous NTuple of per-dim descriptors (concrete per dim)
    plans = map(d -> _mk_dim(Complex{T}, d, sz; inverse), dims)
    D = length(dims)
    # Shared scratch holds the largest TRANSPOSE block (inner*n_d) used by the non-pow2 d>1 path. Batched
    # dims own their tiles inside BatchPlan8; dim-1 needs none. Size at construction so the hot path never
    # allocates (Task 5 gate). `maximum(sz[d])` floors it so the buffer is never empty.
    nscratch = maximum(sz[d] for d in dims)
    for d in dims
        (d == 1 || ispow2(sz[d]) || _is_smooth_2a3(sz[d])) && continue   # batched dims own their tiles
        nscratch = max(nscratch, prod(sz[1:d-1]) * sz[d])
    end
    NDPlan{T, D, typeof(plans), N}(dims, plans, sz, Vector{Complex{T}}(undef, nscratch), inverse)
end

# Apply each transformed dim in turn. @generated unrolls over D with LITERAL indices so the
# heterogeneous `plans`/`dims` tuples are never indexed by a runtime variable (CLAUDE.md rule #1).
@generated function apply_unnormalized!(p::NDPlan{T, D, P, N}, x::AbstractArray) where {T, D, P, N}
    body = Expr(:block)
    for i in 1:D
        push!(body.args, :(_apply_dim!(p.plans[$i], x, p.sz, p.scratch)))
    end
    push!(body.args, :(return x))
    body
end

# Flat layout along dim `d`: inner = ∏size[1:d-1], n_d = size[d], outer = ∏size[d+1:N].
@inline function _dim_extents(d::Int, sz)
    inner = 1; @inbounds for i in 1:(d-1); inner *= sz[i]; end
    outer = 1; @inbounds for i in (d+1):length(sz); outer *= sz[i]; end
    return inner, (@inbounds sz[d]), outer
end

# dim 1: each column `x[:, c]` is a unit-stride contiguous run ⇒ apply in place. Must use a CARTESIAN
# colon view (`view(x, :, c)`), NOT a linear `view(x, o*n_d+1:…)`: linear indexing into a multidim array
# reshapes the parent and ALLOCATES an array header (Task 5 zero-alloc gate). Cartesian colon view is
# zero-alloc (verified).
@inline function _apply_dim!(pd::Dim1Plan, x::AbstractArray{Complex{T}}, sz, scratch) where {T}
    @inbounds for c in CartesianIndices(Base.tail(axes(x)))
        apply_unnormalized!(pd.plan, view(x, :, c))
    end
    return x
end

# dim 1, batched across the trailing dims (small F32). The per-column Dim1Plan FFTs ONE length-n1 transform
# at a time, which cannot fill a 512-bit register for small F32 n1 (single-transform width floor → F64 speed;
# task 6n). The width win comes from batching ACROSS transforms — exactly what the strided dims do. Here:
# pack a chunk of m≤M consecutive transforms (n1×m) into the staging buffer as an m×n1 layout (a tuned
# _transpose_block!), run the EXISTING batched kernel (inner=m, outer=1) on the now-contiguous batch, then
# transpose the natural-order results back. The kernel's scalar tail handles the final m%W columns; the
# outer%M last chunk is just a smaller m. Pointer-based + GC.@preserve, zero-alloc.
@inline _batched_apply!(bp::BatchPlan8{T},  pc::Ptr{Complex{T}}, off, inner, outer) where {T} =
    batched_fft8!(bp, pc, off, inner, outer)
@inline _batched_apply!(bp::BatchPlanMR{T}, pc::Ptr{Complex{T}}, off, inner, outer) where {T} =
    batched_fft_mr!(bp, pc, off, inner, outer)

@inline function _apply_dim!(pd::BatchedDim1{T}, x::AbstractArray{Complex{T}}, sz, scratch) where {T}
    n1 = pd.n1; M = pd.M; stage = pd.stage
    _, _, outer = _dim_extents(1, sz)
    es = sizeof(Complex{T})
    GC.@preserve x stage begin
        px = reinterpret(Ptr{Complex{T}}, pointer(x)); ps = pointer(stage)
        t0 = 0
        @inbounds while t0 < outer
            m = min(M, outer - t0)
            xb = px + t0 * n1 * es
            _transpose_block!(ps, xb, n1, m)          # stage[col + m·r] = x[r + n1·col]   (n1×m → m×n1)
            _batched_apply!(pd.bp, ps, 0, m, 1)        # batched length-n1 FFT of the m packed transforms
            _transpose_block!(xb, ps, m, n1)          # x[r + n1·col] = stage[col + m·r]   (m×n1 → n1×m)
            t0 += m
        end
    end
    return x
end

# pow2 d>1: batched radix-8 column kernel, no transpose. Pointer-based + GC.@preserve, zero-alloc.
@inline function _apply_dim!(pd::BatchedDim{T}, x::AbstractArray{Complex{T}}, sz, scratch) where {T}
    inner, n_d, outer = _dim_extents(pd.d, sz)
    GC.@preserve x begin
        pc = reinterpret(Ptr{Complex{T}}, pointer(x))
        batched_fft8!(pd.bp, pc, 0, inner, outer)
    end
    return x
end

# 2^a·3^b d>1: batched mixed-radix column kernel, no transpose. Pointer-based + GC.@preserve, zero-alloc.
@inline function _apply_dim!(pd::BatchedSmoothDim{T}, x::AbstractArray{Complex{T}}, sz, scratch) where {T}
    inner, n_d, outer = _dim_extents(pd.d, sz)
    GC.@preserve x begin
        pc = reinterpret(Ptr{Complex{T}}, pointer(x))
        batched_fft_mr!(pd.bp, pc, 0, inner, outer)
    end
    return x
end

# other non-pow2 d>1 (incl. the rare leading-singleton inner==1, handled as an identity copy): transpose.
@inline function _apply_dim!(pd::TransposeDim, x::AbstractArray{Complex{T}}, sz, scratch) where {T}
    inner, n_d, outer = _dim_extents(pd.d, sz)
    _apply_dim_transpose!(pd.plan, x, inner, n_d, outer, scratch)
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

# Complex AoS block transpose N1×N2 → N2×N1. Correctness reference: dst[k2 + N2*k1] = src[k1 + N1*k2]
# for k1∈0:N1-1, k2∈0:N2-1. NOTE: blocked.jl's `_btranspose!` is SoA (separate real/imag) — can't reuse.
#
# Hot path: cache-blocked loop over a register 4×4-complex micro-kernel. A Vec{8,T} holds 4 packed complex;
# 4 vloads (one per src column j..j+3, each a contiguous 4-complex run down k1) + 8 complex-granular shuffles
# (a 4×4 element transpose, element = complex pair) + 4 vstores move a 4×4 complex block. vs the old 2×2
# kernel this halves the load/store count AND each store is a full 64-byte cache line (Vec{8,Float64}) along
# contiguous dst-k2 instead of a half-line — measured 1.28→0.74 ns/elem (256² F64), ~0.49 (F32). The 16-complex
# tile keeps both buffers L1-resident (tuned: blk=16 beats 8/12/20/32). Generic over T∈{Float32,Float64}
# (Complex{T} = 2 T ⇒ Vec{8,T} = 4 complex for both). Leftover rows/cols (N mod 4) handled scalar.
@inline function _transpose4x4!(ps::Ptr{T}, pd::Ptr{T}, i, j, N1, N2, es) where {T}
    R0 = vload(Vec{8, T}, ps + (i + N1 * j) * es)
    R1 = vload(Vec{8, T}, ps + (i + N1 * (j + 1)) * es)
    R2 = vload(Vec{8, T}, ps + (i + N1 * (j + 2)) * es)
    R3 = vload(Vec{8, T}, ps + (i + N1 * (j + 3)) * es)
    # unpack lo/hi at complex granularity, then merge 128-bit (2-complex) halves → transposed rows.
    t0 = shufflevector(R0, R1, Val((0, 1, 8, 9, 4, 5, 12, 13)))
    t1 = shufflevector(R2, R3, Val((0, 1, 8, 9, 4, 5, 12, 13)))
    t2 = shufflevector(R0, R1, Val((2, 3, 10, 11, 6, 7, 14, 15)))
    t3 = shufflevector(R2, R3, Val((2, 3, 10, 11, 6, 7, 14, 15)))
    vstore(shufflevector(t0, t1, Val((0, 1, 2, 3, 8, 9, 10, 11))),     pd + (j + N2 * i) * es)
    vstore(shufflevector(t2, t3, Val((0, 1, 2, 3, 8, 9, 10, 11))),     pd + (j + N2 * (i + 1)) * es)
    vstore(shufflevector(t0, t1, Val((4, 5, 6, 7, 12, 13, 14, 15))),   pd + (j + N2 * (i + 2)) * es)
    vstore(shufflevector(t2, t3, Val((4, 5, 6, 7, 12, 13, 14, 15))),   pd + (j + N2 * (i + 3)) * es)
    return
end
function _transpose_block!(dst::Ptr{Complex{T}}, src::Ptr{Complex{T}}, N1::Int, N2::Int) where {T}
    ps = reinterpret(Ptr{T}, src); pd = reinterpret(Ptr{T}, dst)
    es = 2 * sizeof(T)                                # bytes per complex (Ptr{T} arithmetic is byte-wise)
    N1e = N1 & ~3; N2e = N2 & ~3                       # mult-of-4 extents covered by the 4×4 SIMD kernel
    blk = 16                                          # complex per tile dim (L1-resident; tuned — beats 8/12/20/32)
    @inbounds for j0 in 0:blk:(N2e-1)
        jhi = min(j0 + blk, N2e)
        for i0 in 0:blk:(N1e-1)
            ihi = min(i0 + blk, N1e)
            j = j0
            while j < jhi
                i = i0
                while i < ihi
                    _transpose4x4!(ps, pd, i, j, N1, N2, es)
                    i += 4
                end
                j += 4
            end
        end
    end
    @inbounds begin                                   # leftover rows [N1e,N1) (full width) + leftover cols
        for i in N1e:(N1 - 1), j in 0:(N2 - 1)        # [N2e,N2) over the already-handled rows [0,N1e)
            unsafe_store!(dst, unsafe_load(src, i + N1 * j + 1), j + N2 * i + 1)
        end
        for j in N2e:(N2 - 1), i in 0:(N1e - 1)
            unsafe_store!(dst, unsafe_load(src, i + N1 * j + 1), j + N2 * i + 1)
        end
    end
    return
end

# AbstractFFTs surface for N-D arrays.
# AbstractVector methods in abstractfft.jl are more specific → a Vector still routes 1-D (no shadowing).
AbstractFFTs.plan_fft(x::AbstractArray{<:Complex}, region; kws...)  = _pure_plan_fft_nd(x, region; inverse=false)
AbstractFFTs.plan_fft!(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=false)
AbstractFFTs.plan_bfft(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=true)
AbstractFFTs.plan_bfft!(x::AbstractArray{<:Complex}, region; kws...)= _pure_plan_fft_nd(x, region; inverse=true)

Base.size(p::NDPlan) = p.sz
AbstractFFTs.fftdims(p::NDPlan) = p.dims

Base.:*(p::NDPlan, x::AbstractArray) = apply_unnormalized!(p, copy(x))
function LinearAlgebra.mul!(y::AbstractArray, p::NDPlan, x::AbstractArray)
    size(y) == p.sz && size(x) == p.sz ||
        throw(DimensionMismatch("NDPlan size $(p.sz) ≠ arrays $(size(y)) / $(size(x))"))
    apply_unnormalized!(p, copyto!(y, x))
end

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
pfft(x::AbstractArray{<:Complex}, dims=1:ndims(x)) = AbstractFFTs.plan_fft(x, dims) * x
