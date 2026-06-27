# N-dimensional FFT — Complex (c2c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add N-dimensional complex FFT (any rank, any `region`) to PureFFT, separable on the existing
≥FFTW 1-D kernels, drop-in via AbstractFFTs, with a trim-safe / dispatch-free hot path.

**Architecture:** New `src/ndim.jl`. One plan `NDPlan{T,D,P,N}` holds an inner 1-D plan per transformed dim
(built once via `plan_pfft(:fast)`). The apply `@generated`-unrolls over `D` (literal plan indices — the
heterogeneous-`NTuple` runtime-index box is the CLAUDE.md rule-#1 trap) and, per dim `d`: if `d==1` runs
batched contiguous 1-D (no transpose); else cache-blocked transpose ↔ dim 1, batched 1-D, transpose back.
All addressing is flat-memory + integer arithmetic — no `selectdim`/`permutedims`/runtime tuple-slicing.

**Tech Stack:** Julia, AbstractFFTs (the c2c interface), the existing `plan_pfft`/`apply_unnormalized!`
1-D kernels and `blocked.jl`'s `_btranspose!`, ReTestItems + FFTW.jl (test), BenchmarkTools/Plots/JSON (bench).

## Global Constraints

- **No Python** anywhere.
- **Hot path dispatch-free + zero-alloc + trim-safe:** the apply `@generated`-unrolls over `D` with LITERAL
  plan indices (`p.plans[1]`, …; NEVER `p.plans[i]` with runtime `i` — boxes, 135×, CLAUDE.md rule #1);
  flat-memory + integer arithmetic; NO `selectdim`/`mapslices`/`permutedims(runtime perm)`/dim-dependent
  `SubArray`. Compute `inner = ∏ size[1:d-1]` / `outer = ∏ size[d+1:N]` with integer loops over `size(x,i)`,
  never `size(x)[1:d-1]` (runtime tuple slice → type-unstable). Verify `@test_opt`, AllocCheck, TrimCheck.
- **`isnothing(x)`**, never `=== nothing`.
- **Commit author** `15278831+el-oso@users.noreply.github.com`; commit body ends
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Work on branch `feat/ndim-fft`.
- **Separable / exact:** N-D = 1-D FFTs along each transformed dim; reuse the 1-D kernels UNCHANGED. Region
  may be reordered for c2c (order-independent).
- **AutoPlan-Union watch:** `plan_pfft(:fast)` may return an `AutoPlan{T,P}` wrapper; store the inner plans
  so each tuple element is concrete (unwrap if `@test_opt` flags a non-concrete `plans` tuple).
- **Perf gate:** `fftw_median / purefft_median ≥ 0.96` per benchmarked shape, **vs FFTW only** (RustFFT has
  no N-D). Median, `taskset -c 2`, in-place, planning excluded. Not "done" until every benchmarked shape
  clears it; below-gate shapes flagged **"below gate — OPEN"** (no softening adjectives).
- **Tolerances:** rel-err ≤ 1e-12 (Float64), ≤ 1e-4 (Float32) vs FFTW.

### Methodology note
Each transform task writes the FFTW bit-exact test FIRST (the gate). The transpose/stride index math is
converged against that test — where a step says "verify the contiguous-view apply / the block transpose
against FFTW", that is the intended TDD workflow, not a placeholder.

---

### Task 1: Region canonicalization + `NDPlan` + plan construction

**Files:**
- Create: `src/ndim.jl`
- Modify: `src/PureFFT.jl` (include + exports)
- Modify: `src/abstractfft.jl` (region helpers — relax `_checkdim1`)
- Test: `test/ndim_tests.jl`

**Interfaces:**
- Consumes: `plan_pfft(Complex{T}, n; inverse, variant=:fast)`, `AbstractFFTPlan`.
- Produces: `struct NDPlan{T,D,P,N}`; `_canon_region(region, N)::NTuple{D,Int}`;
  `_pure_plan_fft_nd(x::AbstractArray{Complex{T},N}, region; inverse) -> NDPlan`.

- [ ] **Step 1: Write the failing test** (`test/ndim_tests.jl`):
```julia
@testitem "NDPlan builds + region canonicalization" begin
    using PureFFT
    x = randn(ComplexF64, 4, 6, 8)
    p = PureFFT._pure_plan_fft_nd(x, (3, 1); inverse=false)   # unsorted, partial region
    @test p isa PureFFT.NDPlan
    @test p.dims == (1, 3)                # sorted, deduped
    @test length(p.plans) == 2
    @test p.sz == (4, 6, 8)
    @test_throws ArgumentError PureFFT._pure_plan_fft_nd(x, (1, 4); inverse=false)  # dim 4 ∉ 1:3
    @test_throws ArgumentError PureFFT._pure_plan_fft_nd(x, (1, 1); inverse=false)  # dup after canon? (see below)
end
```

- [ ] **Step 2: Run it, expect fail** (`NDPlan` undefined).
Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["ndim"])'`. Expected: FAIL.

- [ ] **Step 3: Implement** (`src/ndim.jl`):
```julia
# N-dimensional complex FFT (separable: 1-D FFTs along each transformed dim, reusing the 1-D kernels).
# D = number of transformed dims (a type parameter so the apply @generated-unrolls over it — the inner
# `plans` tuple is heterogeneous, so runtime indexing would box, CLAUDE.md rule #1). N = array rank.
struct NDPlan{T, D, P, N} <: AbstractFFTPlan{T}
    dims::NTuple{D, Int}            # transformed dims, sorted + deduped
    plans::P                       # NTuple{D} of inner 1-D plans
    sz::NTuple{N, Int}             # full array shape
    scratch::Vector{Complex{T}}    # reused transpose/work buffer
    inverse::Bool
end
plan_length(p::NDPlan) = prod(p.sz)
plan_inverse(p::NDPlan) = p.inverse

# canonicalize a region (Int / tuple / range / Colon) over an N-d array → sorted, deduped NTuple{D,Int},
# validated ⊆ 1:N. Order-independent for c2c.
_canon_region(::Colon, N::Int) = ntuple(identity, N)
_canon_region(r::Integer, N::Int) = _canon_region((Int(r),), N)
_canon_region(r, N::Int) = begin
    t = Tuple(sort!(unique(Int.(collect(r)))))
    all(d -> 1 <= d <= N, t) || throw(ArgumentError("region $r out of bounds for a $N-d array"))
    isempty(t) && throw(ArgumentError("empty region"))
    t
end

function _pure_plan_fft_nd(x::AbstractArray{Complex{T}, N}, region; inverse::Bool) where {T, N}
    dims = _canon_region(region, N)
    sz = size(x)
    plans = map(d -> plan_pfft(Complex{T}, sz[d]; inverse, variant=:fast), dims)   # NTuple{D}, one per dim
    NDPlan{T, length(dims), typeof(plans), N}(dims, plans, sz, Vector{Complex{T}}(undef, maximum(sz)), inverse)
end
```
(Note: `(1,1)` canon → `(1,)`, length-1; the brief's "dup" test asserts a length, adjust: replace that line
with `@test PureFFT._pure_plan_fft_nd(x, (1,1); inverse=false).dims == (1,)`.)

- [ ] **Step 4: Wire module** (`src/PureFFT.jl`): add `include("ndim.jl")` after `abstractfft.jl`'s include.

- [ ] **Step 5: Run, expect pass.** Adjust the `(1,1)` assertion per Step 3 note. Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/ndim.jl src/PureFFT.jl test/ndim_tests.jl
git commit -m "feat(ndim): NDPlan + region canonicalization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: c2c apply — dim-1 fast path (no transpose), `@generated` over `D`

**Files:**
- Modify: `src/ndim.jl`
- Test: `test/ndim_tests.jl`

**Interfaces:**
- Consumes: `NDPlan`, `apply_unnormalized!(plan1d, ::AbstractVector)`.
- Produces: `apply_unnormalized!(p::NDPlan, x)`; `_apply_dim!(plan1d, x, d::Int, sz, scratch)` (dim==1 branch;
  the d>1 branch is Task 3).

- [ ] **Step 1: Write the failing test** — bit-exact vs FFTW for region == dim 1 (and the rank-1 vector):
```julia
@testitem "N-D c2c along dim 1 bit-exact vs FFTW" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32)
        for sz in ((8,), (8,5), (6,4,3))
            x = randn(Complex{T}, sz...)
            p = PureFFT._pure_plan_fft_nd(x, (1,); inverse=false)
            y = copy(x); PureFFT.apply_unnormalized!(p, y)
            ref = fft(x, 1)                      # FFTW along dim 1
            @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol(T)
        end
    end
end
```

- [ ] **Step 2: Run it, expect fail** (`apply_unnormalized!(::NDPlan, …)` undefined). Expected: FAIL.

- [ ] **Step 3: Implement** (`src/ndim.jl`). The apply unrolls over D with LITERAL indices; `_apply_dim!`
  for dim 1 runs batched contiguous 1-D FFTs:
```julia
@generated function apply_unnormalized!(p::NDPlan{T, D, P, N}, x::AbstractArray) where {T, D, P, N}
    body = Expr(:block)
    for i in 1:D                                   # literal indices ⇒ no runtime tuple index (rule #1)
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
```
**Verify during TDD:** that `apply_unnormalized!(plan1d, view(x, a:b))` works on a contiguous `SubArray`
(the AVX kernels use `pointer`; a unit-range view of an `Array` is contiguous and supports `pointer`). If a
specific 1-D plan rejects a `SubArray`, copy the chunk into a length-`n_d` work buffer, apply, copy back —
but prefer the in-place view (the FFTW bit-exact test is the correctness gate either way).

- [ ] **Step 4: Run, expect pass.** Expected: PASS (dim-1 sizes; rank-1 vector reduces to the 1-D path).

- [ ] **Step 5: Commit.**
```bash
git add src/ndim.jl test/ndim_tests.jl
git commit -m "feat(ndim): c2c apply dim-1 fast path (@generated over D)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: c2c apply — dim>1 via cache-blocked transpose (full generality)

**Files:**
- Modify: `src/ndim.jl`
- Test: `test/ndim_tests.jl`

**Interfaces:**
- Consumes: `NDPlan`, `_apply_dim!`, `blocked.jl`'s `_btranspose!` (or a block-offset variant).
- Produces: `_apply_dim_transpose!(plan, x, inner, n_d, outer, scratch)`.

- [ ] **Step 1: Write the failing test** — full c2c generality vs FFTW (regions touching dims > 1):
```julia
@testitem "N-D c2c full generality bit-exact vs FFTW" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    cases = (((8,5), 2), ((8,5), (1,2)), ((6,4,5), 3), ((6,4,5), (1,3)), ((6,4,5), (1,2,3)), ((4,4,4,4), (2,4)))
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(Complex{T}, sz...)
        p = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        y = copy(x); PureFFT.apply_unnormalized!(p, y)
        ref = fft(x, region)
        @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol(T)
    end
end
```

- [ ] **Step 2: Run it, expect fail** (`_apply_dim_transpose!` undefined). Expected: FAIL.

- [ ] **Step 3: Implement** (`src/ndim.jl`). For each of `outer` slices, transpose the `inner×n_d` block to
  `n_d×inner` (so `n_d` is contiguous), batched 1-D, transpose back. Read `src/blocked.jl` for `_btranspose!`
  (it transposes an `N1×N2` AoS/SoA buffer); add an offset/AoS-complex block variant if needed:
```julia
@inline function _apply_dim_transpose!(plan, x::AbstractArray{Complex{T}}, inner::Int, n_d::Int, outer::Int, scratch) where {T}
    need = inner * n_d
    length(scratch) >= need || resize!(scratch, need)   # plan-time scratch is maximum(sz); resize is cold-ish
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
```
**TDD:** the inner store `dst[k2 + N2*k1] = src[k1 + N1*k2]` is the correctness reference (1-based:
`unsafe_store!(dst, unsafe_load(src, k1 + N1*k2 + 1), k2 + N2*k1 + 1)`); the tiling is a cache optimization
that must not change the result. The full-generality FFTW test (Task 3 Step 1) pins every index.

- [ ] **Step 4: Run, expect pass** (all `cases`). Also re-run Task 2's dim-1 test (still green). Expected: PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/ndim.jl test/ndim_tests.jl
git commit -m "feat(ndim): c2c dim>1 via cache-blocked transpose — full generality

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: AbstractFFTs c2c API + prefixed `pfft` for arrays + inv/normalization

**Files:**
- Modify: `src/abstractfft.jl` (or `src/ndim.jl`), `src/PureFFT.jl` (exports)
- Test: `test/ndim_tests.jl`

**Interfaces:**
- Consumes: `_pure_plan_fft_nd`, `NDPlan`, `apply_unnormalized!`.
- Produces: `AbstractFFTs.plan_fft`/`plan_fft!`/`plan_bfft`/`plan_bfft!` on `AbstractArray{<:Complex}`;
  `Base.:*`/`mul!`/`inv` for `NDPlan`; `AbstractFFTs.fftdims`/`normalization`; `pfft(x::AbstractArray, dims)`.

- [ ] **Step 1: Write the failing test** — the public API + round-trips vs FFTW:
```julia
@testitem "N-D public API: fft/ifft/bfft/mul!/inv vs FFTW" begin
    using PureFFT, FFTW, LinearAlgebra
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32), (sz, region) in (((8,5), :), ((6,4,5), (1,3)))
        x = randn(Complex{T}, sz...)
        @test maximum(abs.(fft(x, region===(:) ? (1:ndims(x)) : region) .- (region===(:) ? fft(x) : fft(x, region)))) < tol(T)*max(1,maximum(abs.(x)))
        @test maximum(abs.(ifft(fft(x)) .- x)) < tol(T)
        p = plan_fft(x); @test maximum(abs.((p*x) .- fft(x))) < tol(T)*max(1,maximum(abs.(x)))
        y = copy(x); pin = plan_fft!(y); mul!(y, pin, copy(x)); @test maximum(abs.(y .- fft(x))) < tol(T)*max(1,maximum(abs.(x)))
        @test maximum(abs.((inv(plan_fft(x)) * (plan_fft(x)*x)) .- x)) < tol(T)
    end
end
```

- [ ] **Step 2: Run it, expect fail** (`plan_fft(::AbstractArray,…)` routes to AbstractFFTs default / FFTW, not PureFFT). Expected: FAIL.

- [ ] **Step 3: Implement** the AbstractFFTs surface for arrays (mirror `abstractfft.jl`'s 1-D wrappers):
```julia
AbstractFFTs.plan_fft(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=false)
AbstractFFTs.plan_fft!(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=false) # in-place handled by *(!)
AbstractFFTs.plan_bfft(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=true)
AbstractFFTs.plan_bfft!(x::AbstractArray{<:Complex}, region; kws...) = _pure_plan_fft_nd(x, region; inverse=true)
Base.size(p::NDPlan) = p.sz
AbstractFFTs.fftdims(p::NDPlan) = p.dims
Base.:*(p::NDPlan, x::AbstractArray) = apply_unnormalized!(p, copy(x))
LinearAlgebra.mul!(y::AbstractArray, p::NDPlan, x::AbstractArray) = apply_unnormalized!(p, copyto!(y, x))
function AbstractFFTs.plan_inv(p::NDPlan{T}) where {T}
    ip = _pure_plan_fft_nd(Array{Complex{T}}(undef, p.sz...), p.dims; inverse = !p.inverse)
    AbstractFFTs.ScaledPlan(ip, AbstractFFTs.normalization(real(T), p.sz, p.dims))
end
pfft(x::AbstractArray{<:Complex}, dims=1:ndims(x)) = plan_fft(x, dims) * x
```
Note: the 1-D `AbstractVector` methods (`abstractfft.jl`) must NOT be shadowed — `AbstractArray` is more
general, so `AbstractVector` still wins for vectors; verify a rank-1 `fft(::Vector)` still routes 1-D.

- [ ] **Step 4: Export** in `src/PureFFT.jl` if `pfft` array method needs it (already exported).

- [ ] **Step 5: Run, expect pass.** Confirm `fft(::Vector)` (1-D) still works (no ambiguity/regression). Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/abstractfft.jl src/ndim.jl src/PureFFT.jl test/ndim_tests.jl
git commit -m "feat(ndim): AbstractFFTs c2c array API + pfft(array,dims) + inv/normalization

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Hot-path gates — dispatch-free + zero-alloc + trim-safe

**Files:**
- Modify: `test/ndim_tests.jl` (+ `test/strictmode_tests.jl` if the project gates there)

**Interfaces:** Consumes the public `NDPlan` apply.

- [ ] **Step 1: Write the gate test**:
```julia
@testitem "N-D c2c hot path: dispatch-free + zero-alloc" begin
    using PureFFT, FFTW, JET, LinearAlgebra
    for T in (Float64, Float32), (sz, region) in (((8,5),(1,2)), ((6,4,5),(1,3)))
        x = randn(Complex{T}, sz...); y = similar(x); p = plan_fft(x, region)
        mul!(y, p, x)                                   # warmup
        @test (@allocated mul!(y, p, x)) == 0
        @test_opt target_modules=(PureFFT,) mul!(y, p, x)
    end
end
```
Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["ndim"])'`. If `@test_opt` flags the `plans`
tuple as non-concrete, it's the **AutoPlan-Union leak** (Global Constraints): change `_pure_plan_fft_nd` to
store the concrete inner plan — e.g. unwrap `AutoPlan` (`getfield(plan_pfft(...), :inner)` or the project's
accessor) so each `plans[i]` is concrete. If `@allocated != 0`, the per-dim scratch `resize!` or a
`view`/`copy` in `_apply_dim!` is allocating — preallocate to `maximum(sz)` at plan time and avoid `resize!`
in the hot path (size the scratch for the largest `inner*n_d` over the transformed dims at plan construction).
Iterate until both pass. Commit.

- [ ] **Step 2: TrimCheck** — add the N-D apply to `bench/alloccheck.jl` / the TrimCheck `@validate` sweep
  (mirror how the 1-D hot path is validated). Run it; confirm trim-safe (no `Vector{Any}` at runtime). Commit.

---

### Task 6: Perf — bench harness, measure vs FFTW, enforce the gate

**Files:**
- Create: `bench/run_compare_ndim.jl`, `bench/plot_compare_ndim.jl`
- Generate: `bench/results/compare_ndim.json`, `docs/src/assets/comparison_ndim.png`
- Modify: `docs/src/benchmarks.md`, `ROADMAP.md`

**Interfaces:** Consumes `plan_fft`/`mul!`; mirrors `bench/run_compare_f32.jl`.

- [ ] **Step 1: Write the runner** `bench/run_compare_ndim.jl` (FFTW vs PureFFT, ComplexF64+F32, representative
  shapes — 2-D and 3-D, pow2 + non-pow2; **FFTW only**, RustFFT has no N-D). Mirror `run_compare_f32.jl`'s
  records-JSON shape (median + central-68% spread, `gflops(n,t)=5*n*log2(n)/t/1e9` with `n = prod(sz)`):
```julia
using BenchmarkTools, Statistics, Printf, Dates; import FFTW, PureFFT, JSON
gflops(n,t)=5*n*log2(n)/t/1e9; relspread(t)=(quantile(t,0.84)-quantile(t,0.16))/2/median(t)
shapes = [(256,256),(512,512),(384,384),(64,64,64),(96,96,96)]
results=Dict{String,Any}[]
for T in (Float64,Float32), sz in shapes
    x=randn(Complex{T},sz...); n=prod(sz)
    pf=FFTW.plan_fft!(copy(x);flags=FFTW.MEASURE); pp=PureFFT.plan_fft!(copy(x), 1:length(sz))
    using LinearAlgebra
    tf=(@benchmark $pf*y setup=(y=copy($x)) samples=400 seconds=2).times
    tp=(@benchmark mul!(y,$pp,$x) setup=(y=similar($x)) samples=400 seconds=2).times
    g(t)=gflops(n,median(t)/1e9)
    @printf("%s %s  FFTW %6.1f  PureFFT %6.1f  PF/FFTW=%.2f\n", T, sz, g(tf), g(tp), g(tp)/g(tf))
    push!(results, Dict("T"=>string(T),"sz"=>collect(sz),"fftw_gflops"=>g(tf),"purefft_gflops"=>g(tp),"purefft_relspread"=>relspread(tp)))
end
outdir=joinpath(@__DIR__,"results"); isdir(outdir)||mkdir(outdir)
open(joinpath(outdir,"compare_ndim.json"),"w") do io
    JSON.print(io, Dict("meta"=>Dict("cpu"=>Sys.CPU_NAME,"date"=>string(Dates.today()),"note"=>"N-D c2c, FFTW only (no RustFFT N-D)"),"records"=>results),2)
end
```
Run: `taskset -c 2 julia --project=bench bench/run_compare_ndim.jl`. Inspect `PF/FFTW`.

- [ ] **Step 2: Enforce the gate.** For every shape with `PF/FFTW ≥ 0.96`: good. For any below 0.96×:
  do NOT soften. Profile the apply (the transpose is the prime suspect — FFTW's are mature). Options in
  order: (a) reuse/improve the cache-blocked transpose; (b) a batched 1-D kernel to amortize per-call
  overhead. If a shape can't be lifted this session, record it explicitly as **"below gate — OPEN"** in
  ROADMAP (named shape + ratio), and do not call N-D "done". Add an env-conditional perf `@testitem`
  (`Base.JLOptions().check_bounds == 0` → `@test`, else `@test_skip`) like the r2r gate.

- [ ] **Step 3: Plotter** `bench/plot_compare_ndim.jl` (relative-to-FFTW, mirror `plot_compare_f32.jl`) →
  `docs/src/assets/comparison_ndim.png`.

- [ ] **Step 4: Docs.** Add an "N-dimensional FFT" section to `docs/src/benchmarks.md` (plot + the honest
  gate status per shape). Update `ROADMAP.md`: move N-D complex from "none" to its status (DONE only if every
  benchmarked shape ≥ 0.96× vs FFTW; otherwise list the OPEN shapes).

- [ ] **Step 5: Commit.**
```bash
git add bench/run_compare_ndim.jl bench/plot_compare_ndim.jl bench/results/compare_ndim.json docs/src/assets/comparison_ndim.png docs/src/benchmarks.md ROADMAP.md
git commit -m "bench+docs(ndim): c2c vs FFTW + gate status

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Follow-up plan (separate): Real N-D
On the proven c2c engine: `RealNDPlan` (r2c along `first(region)` via `plan_prfft` + c2c rest), `irfft`/
`brfft` with the original-length bookkeeping, `AbstractFFTs.rfft`/`plan_rfft`/`plan_irfft` for arrays,
bit-exact vs FFTW + round-trips, and its own perf gate vs FFTW. Authored after this plan lands.
