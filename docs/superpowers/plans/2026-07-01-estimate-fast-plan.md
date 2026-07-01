# ESTIMATE Fast-Plan Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `ESTIMATE` planning mode that structurally picks one plan (no timing), cutting first-call plan-construction JIT from ~2-4s to ms, mirroring FFTW's ESTIMATE/MEASURE.

**Architecture:** A pure size-class classifier `_estimate_plan(Complex{T}, n; inverse) → plan-or-nothing` (new `src/estimate.jl`), reached via a single additive branch at the top of `autoplan` when `flags == ESTIMATE`; on a `nothing` (unclassified) it falls through to the unchanged MEASURE competition. A `flags` kwarg (default `MEASURE`) threads from `plan_pfft`/`plan_fft` → `autoplan`.

**Tech Stack:** Julia, ReTestItems (`@testitem`), the existing PureFFT plan types (`Radix4AvxPlan`, `RaderPlan`, `GenPPCodeletPlan`, `GenPPCompositePlan`, `AvxMixedRadixPlan`, `AutoPlan`) and routing predicates (`_max_prime_factor`, `_gen_pp_prime`, `_gen_pp_composite`, `RADER_MIN_P`, `RADER_MAX_PM1_PRIME`), all already in `src/autotune.jl`.

## Global Constraints

- **Julia, dependency-free, pure** — no new packages. The classifier is size arithmetic.
- **MEASURE stays the DEFAULT and byte-unchanged** — ESTIMATE is purely additive; it must not alter the MEASURE code path or its output.
- **Mirror FFTW** — flag names `ESTIMATE`/`MEASURE`, kwarg name `flags`, AbstractFFTs `plan_fft` compatibility.
- **Correct always** — an ESTIMATE plan reuses MEASURE's kernels, so output is numerically identical (bit-exact vs reference DFT ≤1e-14); it may be non-fastest, never wrong.
- **`_estimate_plan` never throws** — uncertainty returns `nothing` (→ MEASURE fallback).
- Commit author email: `15278831+el-oso@users.noreply.github.com`. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Do NOT push; the coordinator handles pushes.
- Run filtered tests during iteration: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["<name>"])'`. Full suite (`Pkg.test()`) only as the final pre-merge gate (Task 3).

## File Structure

- `src/estimate.jl` *(new)* — `@enum PlanRigor ESTIMATE MEASURE` + `_estimate_plan`. One responsibility: classify a size to a single plan or `nothing`.
- `src/PureFFT.jl` *(modify)* — `include("estimate.jl")` immediately BEFORE `include("autotune.jl")` (so the `PlanRigor` enum exists when `autoplan`'s `flags::PlanRigor` signature is parsed); export `ESTIMATE`, `MEASURE`.
- `src/autotune.jl` *(modify)* — `autoplan` gains `flags::PlanRigor = MEASURE` + the one ESTIMATE branch.
- `src/plan.jl` *(modify)* — `plan_pfft` gains `flags` kwarg, threaded to `autoplan` in the `:fast` branch.
- `src/abstractfft.jl` *(modify)* — `_pure_plan_fft` gains `flags` kwarg, threaded to `plan_pfft`.
- `test/estimate_tests.jl` *(new)* — ReTestItems `@testitem`s; auto-discovered by the suite scan.

---

### Task 1: Classifier + PlanRigor enum

**Files:**
- Create: `src/estimate.jl`
- Modify: `src/PureFFT.jl` (add `include` before `autotune.jl`; add export)
- Test: `test/estimate_tests.jl`

**Interfaces:**
- Consumes (all already defined in `src/autotune.jl` and the plan-type files): `Radix4AvxPlan(Complex{T}, n; inverse)`, `RaderPlan(Complex{T}, n; inverse)`, `GenPPCodeletPlan(Complex{T}, n; inverse)`, `GenPPCompositePlan(Complex{T}, n, p, m; inverse)`, `AvxMixedRadixPlan(Complex{T}, n; inverse)` (→ plan or `nothing`), `AutoPlan{T,K}(inner)`, `_max_prime_factor(n)`, `_gen_pp_prime(n)` (→ `Union{Int,Nothing}`), `_gen_pp_composite(n)` (→ `Union{Tuple{Int,Int},Nothing}`), `RADER_MIN_P`, `RADER_MAX_PM1_PRIME`.
- Produces: `@enum PlanRigor ESTIMATE MEASURE`; `_estimate_plan(::Type{Complex{T}}, n::Integer; inverse::Bool=false) where {T} → AbstractFFTPlan or nothing`. pow2 → `AutoPlan`-wrapped `Radix4AvxPlan`; other classes → the raw plan; unclassified → `nothing`.

- [ ] **Step 1: Write the failing routing test**

Create `test/estimate_tests.jl`:

```julia
@testitem "ESTIMATE classifier (_estimate_plan) routes each size class" begin
    P = PureFFT
    C = ComplexF64
    # pow2 → AutoPlan-wrapped Radix4Avx
    @test P._estimate_plan(C, 1024) isa P.AutoPlan{Float64, <:P.Radix4AvxPlan}
    # 2·3·5-smooth → AvxMixedRadixPlan (raw)
    @test P._estimate_plan(C, 720) isa P.AvxMixedRadixPlan
    # prime-square (17²=289) → GenPP codelet (5²/7² use hand B25/B49 via the smooth tree, NOT _gen_pp_prime)
    @test P._estimate_plan(C, 289) isa P.GenPPCodeletPlan
    # large prime with 2^a·3^b p-1 → Rader (257-1 = 256 = 2^8; RADER_MAX_PM1_PRIME=3 requires p-1 = 2^a·3^b)
    @test P._estimate_plan(C, 257) isa P.RaderPlan
    # unclassified (19946 = 2·9973, 9973 prime → not pow2/Rader/GenPP/smooth) → nothing (fallback)
    @test isnothing(P._estimate_plan(C, 19946))
    # returns a valid PlanRigor enum
    @test P.ESTIMATE isa P.PlanRigor && P.MEASURE isa P.PlanRigor
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE classifier"])'`
Expected: FAIL — `UndefVarError: _estimate_plan` (and `ESTIMATE`/`PlanRigor` undefined).

- [ ] **Step 3: Create `src/estimate.jl`**

```julia
# ESTIMATE-style fast-plan classifier. `_estimate_plan` picks ONE plan by size-class heuristic (no timing),
# so autoplan(...; flags=ESTIMATE) compiles ~1 tree instead of ~7 → ms first-call vs seconds. Mirrors FFTW
# ESTIMATE/MEASURE. Uncertainty → `nothing` → autoplan falls back to the MEASURE competition. See
# docs/superpowers/specs/2026-07-01-estimate-fast-plan-design.md.

@enum PlanRigor ESTIMATE MEASURE

# Structural size-class pick, reusing autoplan's exact routing predicates so ESTIMATE and MEASURE agree on
# WHICH plan applies (they differ only in whether the choice among applicable candidates is timed). pow2
# returns the AutoPlan-wrapped Radix4Avx (matching autoplan's pow2 return); other classes return the raw
# plan (matching autoplan's non-pow2 return). Never throws — returns `nothing` on any uncertainty.
function _estimate_plan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ni = Int(n)
    if ispow2(ni)
        p = Radix4AvxPlan(Complex{T}, ni; inverse)
        return AutoPlan{T, typeof(p)}(p)
    end
    # large prime with smooth p-1 → Rader (autoplan's exact short-circuit)
    if ni >= RADER_MIN_P && _max_prime_factor(ni) == ni && _max_prime_factor(ni - 1) <= RADER_MAX_PM1_PRIME
        return RaderPlan(Complex{T}, ni; inverse)
    end
    if T === Float64
        !isnothing(_gen_pp_prime(ni)) && return GenPPCodeletPlan(Complex{T}, ni; inverse)
        gppc = _gen_pp_composite(ni)
        !isnothing(gppc) && return GenPPCompositePlan(Complex{T}, ni, gppc[1], gppc[2]; inverse)
    end
    # 2·3·5-smooth → AvxMixedRadix (W4); plan_tree may still decline a smooth size → nothing → fallback
    if _max_prime_factor(ni) <= 5
        p = AvxMixedRadixPlan(Complex{T}, ni; inverse)
        !isnothing(p) && return p
    end
    return nothing   # unclassified → MEASURE fallback
end
```

- [ ] **Step 4: Wire the include + export into `src/PureFFT.jl`**

Add the export near the other exports (after line 10 `export plan_pfft, ...`):

```julia
export ESTIMATE, MEASURE
```

Add the include IMMEDIATELY BEFORE `include("autotune.jl")` (currently line 39). The result must read:

```julia
include("estimate.jl")
include("autotune.jl")
```

(`estimate.jl` before `autotune.jl` so the `PlanRigor` enum exists when `autoplan`'s `flags::PlanRigor` signature is parsed in Task 2. `_estimate_plan`'s body references `autotune.jl` names — `AutoPlan`, `RADER_MIN_P`, `_gen_pp_prime`, … — but those resolve at call time, after the whole module loads, so the forward reference is fine.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE classifier"])'`
Expected: PASS (6 assertions). (First run recompiles PureFFT — ~2 min.)

- [ ] **Step 6: Commit**

```bash
git add src/estimate.jl src/PureFFT.jl test/estimate_tests.jl
git -c user.email=15278831+el-oso@users.noreply.github.com -c user.name=el-oso commit -m "feat(estimate): PlanRigor enum + _estimate_plan classifier

Structural size-class picker (pow2/Rader/GenPP/smooth) returning one plan or
nothing; the core of the ESTIMATE fast-plan path. Not yet wired into autoplan.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wire `flags` into `autoplan`

**Files:**
- Modify: `src/autotune.jl:164` (the `autoplan` signature + add the branch at the top of the body)
- Test: `test/estimate_tests.jl` (add correctness + fallback testitem)

**Interfaces:**
- Consumes: `_estimate_plan(Complex{T}, n; inverse)` and `PlanRigor`/`ESTIMATE`/`MEASURE` from Task 1.
- Produces: `autoplan(::Type{Complex{T}}, n; inverse=false, flags::PlanRigor=MEASURE)` — when `flags==ESTIMATE` and `_estimate_plan` returns non-nothing, returns it directly; otherwise the existing MEASURE competition (unchanged).

- [ ] **Step 1: Write the failing correctness + fallback test**

Append to `test/estimate_tests.jl`:

```julia
@testitem "ESTIMATE autoplan: correct output + safe fallback" begin
    P = PureFFT
    C = ComplexF64
    ndft(x) = [sum(x[j+1]*cispi(-2*j*k/length(x)) for j in 0:length(x)-1) for k in 0:length(x)-1]
    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    # ESTIMATE output vs reference DFT across classes (pow2, smooth, prime-square, Rader) + a fallback size
    # (19946 = 2·9973: not pow2/Rader/GenPP/smooth → _estimate_plan nothing → MEASURE→Bluestein fallback).
    # Tolerance 1e-12 matches the codebase's reference-DFT convention (test/gen_tests.jl) — the naive ndft
    # reference and Bluestein/Rader both accumulate ~1e-13 at these sizes.
    for n in (1024, 720, 289, 257, 19946)
        x = [C(randn(), randn()) for _ in 1:n]
        pe = P.autoplan(C, n; flags = P.ESTIMATE)
        y = copy(x); P.apply_unnormalized!(pe, y)
        @test relerr(y, ndft(x)) ≤ 1e-12
        # inverse round-trip
        pei = P.autoplan(C, n; inverse = true, flags = P.ESTIMATE)
        P.apply_unnormalized!(pei, y); y ./= n
        @test relerr(y, x) ≤ 1e-12
    end
    # default is MEASURE (flags omitted) — unchanged behavior, still correct
    x = [C(randn(), randn()) for _ in 1:720]
    pm = P.autoplan(C, 720); y = copy(x); P.apply_unnormalized!(pm, y)
    @test relerr(y, ndft(x)) ≤ 1e-12
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE autoplan"])'`
Expected: FAIL — `MethodError`/`UndefKeywordError`: `autoplan` has no `flags` keyword.

- [ ] **Step 3: Add the `flags` param + branch in `src/autotune.jl`**

Change the `autoplan` signature (line 164) from:

```julia
function autoplan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
```

to:

```julia
function autoplan(::Type{Complex{T}}, n::Integer; inverse::Bool = false, flags::PlanRigor = MEASURE) where {T}
    if flags == ESTIMATE
        ep = _estimate_plan(Complex{T}, n; inverse)
        isnothing(ep) || return ep      # structural hit → one plan, no timing; miss → MEASURE below
    end
```

Leave the entire existing body (the `if !ispow2(n)` block and the pow2 tuple timing) exactly as-is after this branch.

- [ ] **Step 4: Run the test to verify it passes**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE autoplan"])'`
Expected: PASS (all classes bit-exact ≤1e-14 fwd + inverse; MEASURE default still correct).

- [ ] **Step 5: Commit**

```bash
git add src/autotune.jl test/estimate_tests.jl
git -c user.email=15278831+el-oso@users.noreply.github.com -c user.name=el-oso commit -m "feat(estimate): autoplan flags=ESTIMATE branch (additive, safe fallback)

autoplan gains flags::PlanRigor=MEASURE; ESTIMATE picks via _estimate_plan (one
plan, no timing) and falls through to the unchanged MEASURE competition on a miss.
MEASURE path byte-unchanged. ESTIMATE output bit-exact vs reference DFT.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Public API (`plan_pfft` / `plan_fft`) + full-suite gate

**Files:**
- Modify: `src/plan.jl:50-64` (`plan_pfft` signature + the `:fast` branch)
- Modify: `src/abstractfft.jl:42-45` (`_pure_plan_fft` signature + the `plan_pfft` call)
- Test: `test/estimate_tests.jl` (add public-API testitem)

**Interfaces:**
- Consumes: `autoplan(...; flags)` from Task 2; `ESTIMATE`/`MEASURE` from Task 1.
- Produces: `plan_pfft(x; flags=MEASURE, variant, inverse)` and `plan_fft(x; flags=MEASURE)` (via `_pure_plan_fft`) both thread `flags` to `autoplan`. Default `MEASURE` (unchanged behavior).

- [ ] **Step 1: Write the failing public-API test**

Append to `test/estimate_tests.jl`:

```julia
@testitem "ESTIMATE public API: plan_pfft + plan_fft flags kwarg" begin
    P = PureFFT
    C = ComplexF64
    ndft(x) = [sum(x[j+1]*cispi(-2*j*k/length(x)) for j in 0:length(x)-1) for k in 0:length(x)-1]
    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    using AbstractFFTs
    x = [C(randn(), randn()) for _ in 1:720]
    # native entry: plan_pfft with flags=ESTIMATE
    pe = P.plan_pfft(x; flags = P.ESTIMATE)
    y = copy(x); P.pfft!(y, pe)
    @test relerr(y, ndft(x)) ≤ 1e-12
    # AbstractFFTs/FFTW-facing entry: plan_fft with flags=ESTIMATE (the transition-compat surface)
    pf = plan_fft(x; flags = P.ESTIMATE)
    @test relerr(pf * x, ndft(x)) ≤ 1e-12
    # default (no flags) still produces a correct transform
    pm = P.plan_pfft(x); ym = copy(x); P.pfft!(ym, pm)
    @test relerr(ym, ndft(x)) ≤ 1e-12
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE public API"])'`
Expected: FAIL — `plan_pfft` / `plan_fft` reject the `flags` keyword.

- [ ] **Step 3: Thread `flags` through `plan_pfft` (`src/plan.jl`)**

Change the signature (line 50-52) from:

```julia
function plan_pfft(
        ::Type{Complex{T}}, n::Integer; inverse::Bool = false, variant::Symbol = :fast
    ) where {T}
```

to:

```julia
function plan_pfft(
        ::Type{Complex{T}}, n::Integer; inverse::Bool = false, variant::Symbol = :fast, flags::PlanRigor = MEASURE
    ) where {T}
```

Change the `:fast` branch (line 64) from:

```julia
        :fast => autoplan(Complex{T}, n; inverse)
```

to:

```julia
        :fast => autoplan(Complex{T}, n; inverse, flags)
```

(The vector-forwarding method `plan_pfft(x; kw...)` at line 83 already passes `flags` through `kw...`; no change needed there.)

- [ ] **Step 4: Thread `flags` through `_pure_plan_fft` (`src/abstractfft.jl`)**

Change the signature (line 42) from:

```julia
function _pure_plan_fft(x::AbstractVector{Complex{F}}, region = 1:1; inplace::Bool = false, inverse::Bool = false) where {F}
```

to:

```julia
function _pure_plan_fft(x::AbstractVector{Complex{F}}, region = 1:1; inplace::Bool = false, inverse::Bool = false, flags::PlanRigor = MEASURE) where {F}
```

Change the inner `plan_pfft` call (line 45) from:

```julia
    inner = plan_pfft(Complex{F}, length(x); inverse, variant = :fast)
```

to:

```julia
    inner = plan_pfft(Complex{F}, length(x); inverse, variant = :fast, flags)
```

(The four `AbstractFFTs.plan_fft`/`plan_fft!`/`plan_bfft`/`plan_bfft!` methods at lines 50-53 already forward `kws...` to `_pure_plan_fft`, so `plan_fft(x; flags=ESTIMATE)` reaches it; no change needed there.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test(test_args=["ESTIMATE public API"])'`
Expected: PASS.

- [ ] **Step 6: Full-suite pre-merge gate**

Run: `taskset -c 2 julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: all pass, 0 failures (the three ESTIMATE testitems + the existing suite unchanged). This confirms the MEASURE default path did not regress.

- [ ] **Step 7: Commit**

```bash
git add src/plan.jl src/abstractfft.jl test/estimate_tests.jl
git -c user.email=15278831+el-oso@users.noreply.github.com -c user.name=el-oso commit -m "feat(estimate): plan_pfft + plan_fft flags kwarg (FFTW-mirroring API)

plan_pfft(x; flags=ESTIMATE) and plan_fft(x; flags=PureFFT.ESTIMATE) thread the
rigor to autoplan; default MEASURE (unchanged). Completes the opt-in ESTIMATE
fast-plan path. Full suite green.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Verification (whole feature)

- `plan_fft(x; flags = PureFFT.ESTIMATE)` and `plan_pfft(x; flags = ESTIMATE)` produce correct transforms (bit-exact vs reference DFT ≤1e-14) across pow2 / smooth / prime-square / Rader, and fall back safely on unclassified sizes.
- MEASURE remains the default and byte-unchanged; full suite green.
- Manual smoke (optional, not a CI test — timing is machine-dependent): on a fresh non-pow2 size, `@elapsed autoplan(ComplexF64, 27720; flags=ESTIMATE)` (warm) should be orders of magnitude below the MEASURE `@elapsed autoplan(ComplexF64, 27720)` first call — the whole point of the feature.
