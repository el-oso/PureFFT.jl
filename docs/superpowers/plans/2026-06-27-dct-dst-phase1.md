# DCT/DST Phase 1 (foundation + DCT-II/III) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the canonical DCT/IDCT (FFTW `REDFT10`/`REDFT01`) to PureFFT plus all shared r2r
infrastructure, FFTW-verified, behind the exact FFTW API names with a `Result`-first core.

**Architecture:** New module `src/r2r.jl` mirroring `src/rfft.jl`. Each transform is FFTW's reduction
(Makhoul): a same-size real FFT + pre/post twiddle for even `N` (the `prfft`/`pirfft` inner, the ~2× lever
that meets the parity gate); a length-`N` complex `pfft` fallback for odd `N` (correct, documented slower).
Kind is a type parameter so plans are concrete/dispatch-free; buffers are preallocated so the apply path is
zero-allocation.

**Tech Stack:** Julia, SIMD.jl (already used), ErrorTypes.jl (new dep), the existing `plan_pfft` /
`plan_prfft` / `plan_pirfft` kernels, ReTestItems + FFTW.jl (test/bench env), BenchmarkTools + Plots + JSON
(bench env).

## Global Constraints

- **No Python** anywhere (global rule).
- **Hot path dispatch-free + zero-alloc**: kind as a type parameter; preallocated buffers; verify with JET
  `@test_opt` and AllocCheck (CLAUDE.md rule #5).
- **No runtime tuple indexing** in any unrolled/counted loop (CLAUDE.md rule #1).
- **`isnothing(x)`**, never `=== nothing`.
- **Commit author email** `15278831+el-oso@users.noreply.github.com`; end commit messages with
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- **Normalization (FFTW-exact):** `r2r`/`REDFT10`/`REDFT01` unnormalized (`REDFT10` then `REDFT01` =
  `2N·x`); `dct`/`idct` orthonormal (scipy `norm="ortho"`, `idct(dct(x))==x`).
- **API names (exact FFTW, defined independently of FFTW.jl):** kind consts `REDFT00…RODFT11`; `r2r`,
  `r2r!`, `plan_r2r`, `dct`, `dct!`, `idct`, `idct!`, `plan_dct`, `plan_idct` (throwing, drop-in) +
  `try*` Result variants. (Phase 1 implements only `REDFT10`/`REDFT01`; other kinds error cleanly.)
- **Performance gates (both required, CLAUDE.md §6 methodology — median, `taskset -c 2`, in-place reps):**
  (1) **No-regression:** existing `ComplexF64`/`ComplexF32`/`rfft` perf guards stay green.
  (2) **Parity vs FFTW:** `fftw_median/purefft_median ≥ 0.96` per kind, **even `N`** (odd `N` is the
  documented complex-fallback exception, below gate until `prfft` gains odd lengths — a later phase).
- **Tolerances:** rel-err ≤ `1e-12` (Float64), ≤ `1e-4` (Float32) vs FFTW and vs the naive reference.

### Methodology notes (read first)

- **Twiddle/permutation constants are derived by the bit-exact TDD loop, not assumed.** Each transform
  task writes the FFTW comparison test *first* (the gate), then implements the reorder + twiddle until it
  passes. Where a step says "derive the exact … against FFTW", that is the intended workflow (matches the
  project's faithful-port "verify each layer bit-exact" rule and spec §12), **not** a placeholder to skip.
  The inner-plan wiring, buffer shapes, and gating tests in each task are complete; the few lines of
  twiddle math are converged against the test.
- **Confirm the ErrorTypes API surface in Task 1** against the installed version and use it consistently
  everywhere: `Ok` / `Err` constructors, `is_error`, `unwrap`, the error accessor (`unwrap_error` *or*
  `unwrap_err` — check), and `@unwrap_or`. The plan's calls assume those names; adjust verbatim if the
  installed version differs.

---

### Task 1: r2r scaffolding — dependency, kind types, error type, module wiring

**Files:**
- Modify: `Project.toml` (add ErrorTypes to `[deps]` + `[compat]`)
- Create: `src/r2r.jl`
- Modify: `src/PureFFT.jl` (include + exports)
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Produces: `abstract type R2RKind end`; singleton kind types/instances `REDFT00, REDFT01, REDFT10,
  REDFT11, RODFT00, RODFT01, RODFT10, RODFT11` (each `<: R2RKind`); `@enum R2RErrKind` +
  `struct R2RError`; nothing else yet.

- [ ] **Step 1: Add the dependency.** Find ErrorTypes' UUID first.

Run: `julia -e 'using Pkg; Pkg.add(name="ErrorTypes")'` in a scratch env to discover the UUID, OR read it
from the registry. Then add to `Project.toml` under `[deps]`:
```toml
ErrorTypes = "8e4d088d-d59d-4eab-a1f0-4ea1f0e74a8e"
```
and under `[compat]`:
```toml
ErrorTypes = "0.5"
```
(Confirm the installed version with `julia --project=. -e 'using Pkg; Pkg.status("ErrorTypes")'` and set
`[compat]` to that minor series.)

- [ ] **Step 2: Write the failing test** (`test/r2r_tests.jl`):
```julia
@testitem "r2r kinds + error type defined" begin
    using PureFFT
    # the 8 FFTW kind singletons exist and are R2RKind
    for k in (REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11)
        @test k isa PureFFT.R2RKind
    end
    # distinct singletons
    @test REDFT10 !== REDFT01
    # error type constructs
    @test PureFFT.R2RError(PureFFT.ERR_UNSUPPORTED_KIND, "x") isa PureFFT.R2RError
end
```

- [ ] **Step 3: Run it, expect fail.**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'` (or run the single item).
Expected: FAIL — `UndefVarError: REDFT00` / module `PureFFT` has no `R2RKind`.

- [ ] **Step 4: Implement the scaffolding** (`src/r2r.jl`):
```julia
# Real-to-real transforms (DCT / DST) — the 8 FFTW r2r kinds. FFTW's reodft reduction math
# (same-size real FFT + pre/post twiddle for II/III/IV; 2(N∓1) extension for I), implemented with
# Julia specialization (kind as a type parameter ⇒ concrete/dispatch-free plans).
import ErrorTypes: Result, Ok, Err, @unwrap_or

# ---- kind singletons (exact FFTW names) ----
abstract type R2RKind end
struct REDFT00_T <: R2RKind end   # DCT-I
struct REDFT10_T <: R2RKind end   # DCT-II  ("the DCT")
struct REDFT01_T <: R2RKind end   # DCT-III ("the IDCT")
struct REDFT11_T <: R2RKind end   # DCT-IV
struct RODFT00_T <: R2RKind end   # DST-I
struct RODFT10_T <: R2RKind end   # DST-II
struct RODFT01_T <: R2RKind end   # DST-III
struct RODFT11_T <: R2RKind end   # DST-IV
const REDFT00 = REDFT00_T(); const REDFT10 = REDFT10_T(); const REDFT01 = REDFT01_T(); const REDFT11 = REDFT11_T()
const RODFT00 = RODFT00_T(); const RODFT10 = RODFT10_T(); const RODFT01 = RODFT01_T(); const RODFT11 = RODFT11_T()

# ---- error type (Result-first core; throwing shims added in Task 6) ----
@enum R2RErrKind ERR_UNSUPPORTED_KIND ERR_SIZE_TOO_SMALL ERR_BAD_ELTYPE
struct R2RError
    kind::R2RErrKind
    msg::String
end
Base.show(io::IO, e::R2RError) = print(io, "R2RError(", e.kind, "): ", e.msg)
```

- [ ] **Step 5: Wire into the module** (`src/PureFFT.jl`): add after the existing `include("rfft.jl")` (or
near it) `include("r2r.jl")`, and add to the export line:
```julia
export REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11
```

- [ ] **Step 6: Run the test, expect pass.**

Run: the same test command. Expected: PASS.

- [ ] **Step 7: Commit.**
```bash
git add Project.toml src/r2r.jl src/PureFFT.jl test/r2r_tests.jl
git commit -m "feat(r2r): scaffolding — ErrorTypes dep, 8 FFTW kind singletons, R2RError

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `R2RPlan{K,T,P}` + `tryplan_r2r` skeleton (Result plumbing + Err path)

**Files:**
- Modify: `src/r2r.jl`
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Consumes: kind singletons + `R2RError` from Task 1; existing `plan_prfft(::Type{T},n)`,
  `plan_pirfft(::Type{T},n)`, `plan_pfft(Complex{T},n; variant, inverse)`, `apply_rfft!`, `apply_irfft!`,
  `apply_unnormalized!`.
- Produces: `struct R2RPlan{K,T,P}`; `tryplan_r2r(x::AbstractVector{<:Real}, kind::R2RKind) ->
  Result{R2RPlan, R2RError}`; `_natural_size(kind, n)::Int`. Apply is added per-kind in Tasks 3–5.

- [ ] **Step 1: Write the failing test** (append to `test/r2r_tests.jl`):
```julia
@testitem "tryplan_r2r returns Err for unsupported kind / bad size" begin
    using PureFFT, ErrorTypes
    x = randn(8)
    # Phase 1 supports only REDFT10/REDFT01; others are unsupported for now
    @test ErrorTypes.is_error(PureFFT.tryplan_r2r(x, REDFT11))
    # DCT-II of a valid vector returns Ok (a plan)
    r = PureFFT.tryplan_r2r(x, REDFT10)
    @test !ErrorTypes.is_error(r)
    @test ErrorTypes.unwrap(r) isa PureFFT.R2RPlan
end
```

- [ ] **Step 2: Run it, expect fail** (`tryplan_r2r` undefined).

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. Expected: FAIL.

- [ ] **Step 3: Implement the plan struct + constructor skeleton** (`src/r2r.jl`):
```julia
# K = kind singleton type; T = Float64/Float32; P = inner plan type (real FFT, inverse real FFT, or
# complex fallback). Preallocated buffers ⇒ zero-alloc apply. `scale` = 1 for r2r; ortho factor for dct.
struct R2RPlan{K, T, P}
    n::Int
    inner::P
    pre::Vector{Complex{T}}    # pre-twiddles (kind-specific; may be empty)
    post::Vector{Complex{T}}   # post-twiddles
    rbuf::Vector{T}            # real work buffer
    cbuf::Vector{Complex{T}}   # half-spectrum / complex work buffer
    scale::T
end

# natural inner-FFT size per kind (Phase 1: II/III use size n). Type-I (Task of a later phase) is 2(n∓1).
_natural_size(::Union{REDFT10_T, REDFT01_T}, n::Int) = n

# Phase-1 support set. Returns Ok(plan) or Err(R2RError). Per-kind builders are defined in Tasks 3–5;
# this skeleton dispatches and returns the Err for unsupported kinds.
function tryplan_r2r(x::AbstractVector{<:Real}, kind::R2RKind)
    T = float(eltype(x))
    n = length(x)
    return _build_r2r(kind, T, n)
end

# fallthrough: any kind without a concrete _build_r2r method is unsupported (Phase 1)
_build_r2r(kind::R2RKind, ::Type{T}, n::Int) where {T} =
    Err{R2RPlan, R2RError}(R2RError(ERR_UNSUPPORTED_KIND, "kind $(kind) not implemented yet"))
```
(`_build_r2r(::REDFT10_T, …)` / `(::REDFT01_T, …)` are added in Tasks 3–5; until then `REDFT10` would hit
the fallthrough, so this test's `REDFT10` Ok assertion will fail — that's expected and Task 3 fixes it. To
keep Task 2 self-contained, split the test: assert only the **Err** path here, and move the `REDFT10`-Ok
assertion to Task 3.)

- [ ] **Step 4: Trim the Task-2 test to the Err path only** (the Ok assertion belongs to Task 3):
```julia
@testitem "tryplan_r2r returns Err for unsupported kind" begin
    using PureFFT, ErrorTypes
    @test ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(8), REDFT11))
end
```

- [ ] **Step 5: Run it, expect pass.** Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/r2r.jl test/r2r_tests.jl
git commit -m "feat(r2r): R2RPlan{K,T,P} + tryplan_r2r Result skeleton (Err path)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: DCT-II (`REDFT10`) — even-N real-FFT route, bit-exact vs FFTW + naive

**Files:**
- Modify: `src/r2r.jl`
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Consumes: `R2RPlan`, `_build_r2r`, `apply_rfft!`, `plan_prfft`.
- Produces: `_build_r2r(::REDFT10_T, ::Type{T}, n)` (even-N branch); `_apply!(p::R2RPlan{REDFT10_T},
  y, x)`; `tryr2r(x, kind) -> Result{Vector, R2RError}`; an internal naive reference
  `_dct2_naive(x)` for tests.

- [ ] **Step 1: Write the failing test** — bit-exact vs FFTW.jl and vs the naive sum, even N, F64+F32:
```julia
@testitem "DCT-II (REDFT10) bit-exact vs FFTW + naive (even N)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    naive(x) = [2*sum(x[j+1]*cos(pi*(2j+1)*k/(2length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for T in (Float64, Float32), n in (2, 4, 8, 16, 100, 256, 1000)
        x = randn(T, n)
        y = PureFFT.r2r(x, REDFT10)                 # throwing form added in Task 6; use tryr2r here:
        y = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10))
        ref = FFTW.r2r(x, FFTW.REDFT10)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
        @test maximum(abs.(y .- T.(naive(Float64.(x))))) / max(maximum(abs.(ref)), eps(T)) < tol(T)
    end
end
```

- [ ] **Step 2: Run it, expect fail** (`tryr2r` / REDFT10 builder undefined). Expected: FAIL.

- [ ] **Step 3: Implement DCT-II even-N (Makhoul real-FFT route)** (`src/r2r.jl`):
```julia
# DCT-II twiddle: W2N^k = exp(-iπk/2N), k = 0..n-1.
_dct_post_tw(::Type{T}, n) where {T} = Complex{T}[cispi(-T(k) / (2n)) for k in 0:(n - 1)]

function _build_r2r(::REDFT10_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Err{R2RPlan, R2RError}(R2RError(ERR_SIZE_TOO_SMALL, "REDFT10 needs n≥1"))
    if iseven(n)
        inner = plan_prfft(T, n)                       # length-n real FFT
        post  = _dct_post_tw(T, n)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        return Ok{R2RPlan, R2RError}(R2RPlan{REDFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, rbuf, cbuf, one(T)))
    else
        return _build_r2r_dct2_odd(T, n)               # Task 4
    end
end

# Apply (even N): reorder x → rbuf (even samples up, odd samples reversed), real FFT → cbuf half-spectrum,
# y_k = 2·Re(post_k · V_k) with Hermitian extension V_k = conj(V_{n-k}) for k > n/2.
function _apply!(p::R2RPlan{REDFT10_T, T}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T}
    n = p.n; m = n ÷ 2; v = p.rbuf; V = p.cbuf; W = p.post
    @inbounds for j in 0:(m - 1)
        v[j + 1]       = T(x[2j + 1])      # x[2j]
        v[n - j]       = T(x[2j + 2])      # x[2j+1] reversed into the tail
    end
    apply_rfft!(p.inner, v, V)             # V[1..m+1] = half-spectrum
    @inbounds for k in 0:(n - 1)
        Vk = k <= m ? V[k + 1] : conj(V[n - k + 1])
        y[k + 1] = T(2) * real(W[k + 1] * Vk)
    end
    return y
end

_dct2_naive(x) = [2*sum(x[j+1]*cospi((2j+1)*k/(2length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]

# generic apply entry + tryr2r (per-kind _apply! dispatched on the plan's K)
function tryr2r(x::AbstractVector{<:Real}, kind::R2RKind)
    r = tryplan_r2r(x, kind)
    ErrorTypes.is_error(r) && return Err{Vector, R2RError}(ErrorTypes.unwrap_error(r))
    p = ErrorTypes.unwrap(r)
    T = eltype(p.rbuf)
    y = Vector{T}(undef, p.n)
    _apply!(p, y, x)
    return Ok{Vector, R2RError}(y)
end
```

- [ ] **Step 4: Run the (even-N) test, expect pass.**

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. Expected: PASS for the even-N sizes.
(If a constant is off, the FFTW comparison pinpoints it — adjust the `post` sign/scale until bit-exact.)

- [ ] **Step 5: Commit.**
```bash
git add src/r2r.jl test/r2r_tests.jl
git commit -m "feat(r2r): DCT-II (REDFT10) even-N real-FFT route, bit-exact vs FFTW

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: DCT-II odd-N complex fallback

**Files:**
- Modify: `src/r2r.jl`
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Consumes: `R2RPlan`, `plan_pfft`, `apply_unnormalized!`, `_dct_post_tw`.
- Produces: `_build_r2r_dct2_odd(::Type{T}, n)`; `_apply!(p::R2RPlan{REDFT10_T,T,P})` already dispatches —
  add an odd-N branch keyed on whether `p.inner` is a complex plan (store a flag via a distinct kind
  wrapper type, see Step 3).

- [ ] **Step 1: Write the failing test** — odd N, bit-exact vs FFTW (note: odd N is correct but below the
  parity gate; correctness only here):
```julia
@testitem "DCT-II (REDFT10) odd-N bit-exact vs FFTW" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (1, 3, 5, 7, 9, 99, 257)
        x = randn(T, n)
        y = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10))
        ref = FFTW.r2r(x, FFTW.REDFT10)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
    end
end
```

- [ ] **Step 2: Run it, expect fail** (odd N currently returns `Err` / unimplemented). Expected: FAIL.

- [ ] **Step 3: Implement the odd-N complex fallback.** Use a length-`n` complex FFT of the same reordered
  sequence; the post-step is identical (`2·Re(W_k·V_k)`), but `V` is the full complex spectrum (no
  Hermitian shortcut). Distinguish the inner at the type level so `_apply!` stays dispatch-free — store the
  complex plan and branch on `P<:` the complex plan type:
```julia
function _build_r2r_dct2_odd(::Type{T}, n::Int) where {T}
    inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = false)
    post  = _dct_post_tw(T, n)
    cbuf  = Vector{Complex{T}}(undef, n)     # full complex spectrum
    return Ok{R2RPlan, R2RError}(R2RPlan{REDFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, T[], cbuf, one(T)))
end

# odd-N apply: inner is a complex plan ⇒ p.rbuf is empty, p.cbuf length n. Dispatch on the inner type by
# checking isempty(p.rbuf) (set only for the real-FFT route). Keep it branch-free-per-call via @inline.
@inline _is_complex_inner(p::R2RPlan) = isempty(p.rbuf)

function _apply!(p::R2RPlan{REDFT10_T, T}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T}
    n = p.n; W = p.post
    if _is_complex_inner(p)
        V = p.cbuf
        @inbounds for j in 0:(n ÷ 2)            # even/odd reorder into the complex buffer
            V[j + 1] = Complex{T}(T(x[2j + 1]), zero(T))
        end
        # (re-derive the exact odd-N reorder during TDD; the FFTW test pins it)
        apply_unnormalized!(p.inner, V)
        @inbounds for k in 0:(n - 1)
            y[k + 1] = T(2) * real(W[k + 1] * V[k + 1])
        end
        return y
    end
    # even-N real-FFT route (from Task 3):
    m = n ÷ 2; v = p.rbuf; V = p.cbuf
    @inbounds for j in 0:(m - 1)
        v[j + 1] = T(x[2j + 1]); v[n - j] = T(x[2j + 2])
    end
    apply_rfft!(p.inner, v, V)
    @inbounds for k in 0:(n - 1)
        Vk = k <= m ? V[k + 1] : conj(V[n - k + 1])
        y[k + 1] = T(2) * real(W[k + 1] * Vk)
    end
    return y
end
```
(Replace the placeholder reorder comment with the exact odd-N permutation; the bit-exact FFTW test is the
gate. If keeping two `_apply!` methods is cleaner than the `isempty` branch, parameterize `R2RPlan` with a
`Real`/`Complex` route marker instead — choose whichever keeps `@test_opt` clean in Task 7.)

- [ ] **Step 4: Run the odd-N test, expect pass.** Expected: PASS.

- [ ] **Step 5: Run the full r2r test set** to confirm even-N (Task 3) still passes.

Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/r2r.jl test/r2r_tests.jl
git commit -m "feat(r2r): DCT-II odd-N complex fallback (correct; below parity gate, documented)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: DCT-III (`REDFT01`) — inverse, bit-exact + II↔III round-trip

**Files:**
- Modify: `src/r2r.jl`
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Consumes: `R2RPlan`, `plan_pirfft`/`apply_irfft!` (even N), `plan_pfft` (odd N), `tryr2r`.
- Produces: `_build_r2r(::REDFT01_T, ::Type{T}, n)`; `_apply!(p::R2RPlan{REDFT01_T,T}, y, x)`;
  `_natural_size(::REDFT01_T, n)=n`.

- [ ] **Step 1: Write the failing test** — bit-exact vs FFTW + the unnormalized inverse relation
  (`REDFT01(REDFT10(x)) == 2N·x`):
```julia
@testitem "DCT-III (REDFT01) bit-exact vs FFTW + II↔III round-trip" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (2, 4, 8, 16, 100, 256, 3, 5, 99)
        x = randn(T, n)
        y   = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT01))
        ref = FFTW.r2r(x, FFTW.REDFT01)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
        # unnormalized round-trip: REDFT01 ∘ REDFT10 = 2N·identity
        rt = ErrorTypes.unwrap(PureFFT.tryr2r(ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10)), REDFT01))
        @test maximum(abs.(rt ./ (2n) .- x)) / max(maximum(abs.(x)), eps(T)) < tol(T)
    end
end
```

- [ ] **Step 2: Run it, expect fail** (REDFT01 unimplemented). Expected: FAIL.

- [ ] **Step 3: Implement DCT-III** as the structural inverse of DCT-II. Even N: pre-twiddle the input into
  a half-spectrum, `apply_irfft!`, inverse reorder. Odd N: complex `pfft` fallback. The exact pre-twiddle
  is the conjugate/inverse of Task 3's post-step; pin it with the FFTW test:
```julia
_natural_size(::REDFT01_T, n::Int) = n
function _build_r2r(::REDFT01_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Err{R2RPlan, R2RError}(R2RError(ERR_SIZE_TOO_SMALL, "REDFT01 needs n≥1"))
    if iseven(n)
        inner = plan_pirfft(T, n)
        pre   = _dct_post_tw(T, n)                 # reuse W2N^k; III uses its conjugate (apply via conj)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        return Ok{R2RPlan, R2RError}(R2RPlan{REDFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], rbuf, cbuf, one(T)))
    else
        inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = true)
        pre   = _dct_post_tw(T, n)
        cbuf  = Vector{Complex{T}}(undef, n)
        return Ok{R2RPlan, R2RError}(R2RPlan{REDFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], T[], cbuf, one(T)))
    end
end
# _apply!(p::R2RPlan{REDFT01_T,T}, y, x): build half-spectrum U_k from x using conj(pre_k), apply_irfft!,
# inverse-reorder into y (undo the even/odd permutation). Derive the exact U_k boundary terms via TDD
# against FFTW.REDFT01; the round-trip test guards the normalization.
```
(Write the full `_apply!(::R2RPlan{REDFT01_T},…)` here, mirroring Task 3's structure in reverse. The two
tests — direct FFTW match and the `2N` round-trip — together pin every constant.)

- [ ] **Step 4: Run, expect pass.** Expected: PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/r2r.jl test/r2r_tests.jl
git commit -m "feat(r2r): DCT-III (REDFT01) + II↔III unnormalized round-trip, bit-exact vs FFTW

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Throwing shims + `dct`/`idct` orthonormal + `inv`/`\`

**Files:**
- Modify: `src/r2r.jl`, `src/PureFFT.jl` (exports)
- Test: `test/r2r_tests.jl`

**Interfaces:**
- Consumes: `tryplan_r2r`, `tryr2r`, `_apply!`, `R2RPlan`.
- Produces: `plan_r2r`, `r2r`, `r2r!`, `dct`, `dct!`, `idct`, `idct!`, `plan_dct`, `plan_idct`,
  `Base.:*(::R2RPlan, x)`, `LinearAlgebra.mul!`, `Base.inv(::R2RPlan)`, `Base.:\`.

- [ ] **Step 1: Write the failing test** — throwing shims match FFTW; `dct`/`idct` orthonormal:
```julia
@testitem "r2r/dct throwing API + orthonormal dct/idct" begin
    using PureFFT, FFTW, LinearAlgebra
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (4, 8, 16, 100, 7)
        x = randn(T, n)
        @test maximum(abs.(r2r(x, REDFT10) .- FFTW.r2r(x, FFTW.REDFT10))) < tol(T)*max(1, maximum(abs.(x))*n)
        @test maximum(abs.(dct(x) .- FFTW.dct(x))) < tol(T)            # orthonormal, matches FFTW.jl
        @test maximum(abs.(idct(dct(x)) .- x)) < tol(T)               # ortho round-trip
        p = plan_r2r(x, REDFT10); @test maximum(abs.((p*x) .- r2r(x, REDFT10))) < tol(T)*max(1, maximum(abs.(x))*n)
    end
    @test_throws ArgumentError plan_r2r(randn(8), REDFT11)            # unsupported in Phase 1 → throws
end
```

- [ ] **Step 2: Run it, expect fail.** Expected: FAIL.

- [ ] **Step 3: Implement the shims + dct/idct + plan ops** (`src/r2r.jl`):
```julia
import LinearAlgebra

# throwing shims (FFTW drop-in): unwrap-or-throw
plan_r2r(x::AbstractVector{<:Real}, kind::R2RKind) =
    @unwrap_or tryplan_r2r(x, kind) e -> throw(ArgumentError(string(e)))
r2r(x::AbstractVector{<:Real}, kind::R2RKind) =
    @unwrap_or tryr2r(x, kind) e -> throw(ArgumentError(string(e)))
r2r!(x::AbstractVector{<:Real}, kind::R2RKind) = copyto!(x, r2r(x, kind))

Base.:*(p::R2RPlan{K,T}, x::AbstractVector) where {K,T} = _apply!(p, Vector{T}(undef, p.n), x)
LinearAlgebra.mul!(y::AbstractVector, p::R2RPlan, x::AbstractVector) = _apply!(p, y, x)

# orthonormal DCT-II/III (scipy norm="ortho" / FFTW.jl dct). Derived from the unnormalized r2r:
#   ortho DCT-II:  y0 *= sqrt(1/(4n)); yk *= sqrt(1/(2n))  (k>0)   [from REDFT10's factor of 2]
# Pin the exact factors against FFTW.dct in the test; expose dct/idct as scaled r2r.
function dct(x::AbstractVector{<:Real})
    T = float(eltype(x)); n = length(x)
    y = r2r(x, REDFT10)
    s0 = sqrt(T(1) / (4n)); s = sqrt(T(1) / (2n))
    @inbounds y[1] *= s0
    @inbounds for k in 2:n; y[k] *= s; end
    return y
end
function idct(x::AbstractVector{<:Real})
    T = float(eltype(x)); n = length(x)
    # ortho idct = scaled REDFT01 of a pre-scaled input (inverse of dct's scaling); pin vs FFTW.idct
    x2 = copy(x); s0 = sqrt(T(1) / (4n)); s = sqrt(T(1) / (2n))
    @inbounds x2[1] /= s0; @inbounds for k in 2:n; x2[k] /= s; end
    y = r2r(x2, REDFT01)
    @inbounds for k in 1:n; y[k] *= (T(1) / (2n)); end   # exact factor pinned by idct(dct(x))≈x test
    return y
end
dct!(x) = copyto!(x, dct(x)); idct!(x) = copyto!(x, idct(x))
plan_dct(x::AbstractVector{<:Real}) = plan_r2r(x, REDFT10)   # convenience; dct() applies the ortho scale
plan_idct(x::AbstractVector{<:Real}) = plan_r2r(x, REDFT01)

# inverse plan: REDFT10 ↔ REDFT01 with the 1/2n scale (gives \ / ldiv!)
function Base.inv(p::R2RPlan{REDFT10_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), REDFT01)
    return (ip, T(1) / (2 * p.n))
end
```
(Adjust the exact ortho factors until `dct(x) ≈ FFTW.dct(x)` and `idct(dct(x)) ≈ x` — the test pins them.
If `inv` returning a tuple is awkward, wrap in a small `ScaledR2RPlan` holding plan + scale and define
`*`/`\` on it.)

- [ ] **Step 4: Export the throwing + try names** (`src/PureFFT.jl`):
```julia
export r2r, r2r!, plan_r2r, dct, dct!, idct, idct!, plan_dct, plan_idct
export tryr2r, tryplan_r2r   # Result-first variants
```

- [ ] **Step 5: Run, expect pass.** Expected: PASS.

- [ ] **Step 6: Commit.**
```bash
git add src/r2r.jl src/PureFFT.jl test/r2r_tests.jl
git commit -m "feat(r2r): throwing FFTW-drop-in shims + orthonormal dct/idct + inv

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Gates — zero-alloc/dispatch-free + bench harness + perf gates + no-regression

**Files:**
- Modify: `test/r2r_tests.jl`
- Create: `bench/run_compare_r2r.jl`, `bench/plot_compare_r2r.jl`
- Generate: `bench/results/compare_r2r.json`, `docs/src/assets/comparison_r2r*.png`

**Interfaces:**
- Consumes: `plan_r2r`, `mul!`, the existing `bench/run_compare.jl` pattern.
- Produces: an alloc/inference test item; the r2r bench runner + plotter (mirror `run_compare_f32.jl` /
  `plot_compare_f32.jl`).

- [ ] **Step 1: Zero-alloc + dispatch-free test** (append to `test/r2r_tests.jl`):
```julia
@testitem "r2r hot path: zero-alloc + dispatch-free" begin
    using PureFFT, JET, LinearAlgebra
    for T in (Float64, Float32), n in (8, 256)         # even-N real-FFT route
        x = randn(T, n); y = similar(x); p = plan_r2r(x, REDFT10)
        mul!(y, p, x)                                  # warmup
        @test (@allocated mul!(y, p, x)) == 0
        @test_opt target_modules=(PureFFT,) mul!(y, p, x)
    end
end
```
Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. If it fails on allocations, make the
even-N `_apply!` fully buffer-based (it already is) and ensure the odd-N branch isn't selected here; if
`@test_opt` flags the `isempty(p.rbuf)` route branch, switch `R2RPlan` to a route-marker type parameter so
dispatch is static (noted in Task 4 Step 3). Iterate until both pass. Commit.

- [ ] **Step 2: No-regression check.** Re-run the existing perf-regression guards to prove DCT/DST didn't
  perturb the complex/real FFT:

Run: `julia --project=. -e 'using Pkg; Pkg.test()'` (full suite — the relative-perf-regression `@testitem`s
must stay green). Expected: PASS, including the existing performance-regression items.

- [ ] **Step 3: Write the r2r bench runner** `bench/run_compare_r2r.jl` (mirror `bench/run_compare_f32.jl`):
```julia
# DCT/DST comparison runner — FFTW vs PureFFT (r2r), saved to bench/results/compare_r2r.json.
using BenchmarkTools, Statistics, Printf, Dates
import FFTW, PureFFT, JSON
gflops(n, t) = 5 * n * log2(n) / t / 1.0e9   # same model as the FFT bench (comparability)
relspread(t) = (quantile(t, 0.84) - quantile(t, 0.16)) / 2 / median(t)
const SAMPLES = 1000; const SECONDS = 2.0
even_sizes() = [2^e for e in 3:16]
function sample(n, T)
    x = randn(T, n)
    pf = FFTW.plan_r2r(copy(x), FFTW.REDFT10; flags = FFTW.MEASURE)
    pp = PureFFT.plan_r2r(x, REDFT10)
    tf = (@benchmark $pf * y setup=(y=copy($x)) samples=SAMPLES seconds=SECONDS).times
    tp = (@benchmark mul!(y, $pp, $x) setup=(y=similar($x)) samples=SAMPLES seconds=SECONDS).times
    (tf, tp)
end
results = Dict{String,Any}[]
for T in (Float64, Float32), n in even_sizes()
    tf, tp = sample(n, T); g(t) = gflops(n, median(t)/1e9)
    @printf("DCT-II %s n=%-6d FFTW %6.1f  PureFFT %6.1f  PF/FFTW=%.2f\n", T, n, g(tf), g(tp), g(tp)/g(tf))
    push!(results, Dict("kind"=>"REDFT10","T"=>string(T),"n"=>n,
        "fftw_gflops"=>g(tf),"purefft_gflops"=>g(tp),"purefft_relspread"=>relspread(tp)))
end
outdir = joinpath(@__DIR__, "results"); isdir(outdir) || mkdir(outdir)
open(joinpath(outdir, "compare_r2r.json"), "w") do io
    JSON.print(io, Dict("meta"=>Dict("cpu"=>Sys.CPU_NAME,"date"=>string(Dates.today()),
        "note"=>"DCT-II (REDFT10), even N, in-place, planning excluded"), "records"=>results), 2)
end
using LinearAlgebra   # for mul!
```
Run: `taskset -c 2 julia --project=bench bench/run_compare_r2r.jl`. Inspect the `PF/FFTW` column.

- [ ] **Step 4: Enforce the parity gate (0.96×, even N) as a test** (append to `test/r2r_tests.jl`,
  gated so it only runs when explicitly requested, like the existing perf-regression items):
```julia
@testitem "DCT-II parity vs FFTW ≥ 0.96× (even N)" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics, LinearAlgebra
    med(b) = median(b.times)
    for T in (Float64, Float32), n in (256, 1024, 4096)
        x = randn(T, n)
        pf = FFTW.plan_r2r(copy(x), FFTW.REDFT10; flags=FFTW.MEASURE); pp = plan_r2r(x, REDFT10)
        tf = med(@benchmark $pf*y setup=(y=copy($x)))
        tp = med(@benchmark mul!(y,$pp,$x) setup=(y=similar($x)))
        @test tf/tp ≥ 0.96            # PureFFT no more than 4% slower than FFTW
    end
end
```
If a size misses 0.96×, profile the pre/post step (it's the only non-FFT work; the inner FFT is already
≥FFTW); options in priority order: (a) `@generated` the reorder+twiddle to straight-line code (spec §4),
(b) fuse the reorder into `apply_rfft!`'s packing. Re-measure in the full kernel; do not chase below the
±7% floor.

- [ ] **Step 5: Write the plotter** `bench/plot_compare_r2r.jl` (mirror `bench/plot_compare_f32.jl`):
relative-to-FFTW DCT-II throughput, F64+F32, → `docs/src/assets/comparison_r2r.png`. Run:
`julia -O3 --project=bench bench/plot_compare_r2r.jl`.

- [ ] **Step 6: Document** — add a short "DCT / DST (real-to-real)" section to `docs/src/benchmarks.md`
  with the plot + a one-line note that DCT-II/III are live (Phase 1), even-N at FFTW parity, odd-N the
  documented complex fallback. Update `ROADMAP.md`: add a "DCT/DST" line under breadth/type coverage with
  Phase-1 status + a pointer to the spec.

- [ ] **Step 7: Commit.**
```bash
git add test/r2r_tests.jl bench/run_compare_r2r.jl bench/plot_compare_r2r.jl bench/results/compare_r2r.json docs/src/assets/comparison_r2r.png docs/src/benchmarks.md ROADMAP.md
git commit -m "test+bench(r2r): zero-alloc gate, no-regression, 0.96x parity vs FFTW + plots/docs

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Phases 2–4 (separate plans, authored after Phase 1)

Each reuses Phase 1's `R2RPlan`/`Result`/shim/test/bench harness; only per-kind `_pre!`/`_post!` +
`_build_r2r` methods + tests are added, kind-by-kind, each bit-exact vs FFTW before the next:
- **Phase 2:** DCT-IV (`REDFT11`) — dual pre+post twiddle.
- **Phase 3:** DST-II/III/IV (`RODFT10`/`01`/`11`) — sine analogues.
- **Phase 4:** type-I pair (`REDFT00`/`RODFT00`) — the 2(N∓1) real-FFT extension route.
- **Later:** extend `prfft` to odd lengths (lifts odd-N II/III/IV to the parity gate); the v2 `@generated`
  direct small-N codelet + plan-time direct-vs-reduce selection.
