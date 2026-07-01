# ESTIMATE-style fast-plan path — design

**Date:** 2026-07-01
**Status:** design approved (brainstorm) — ready for implementation plan

## Goal

Add an opt-in `ESTIMATE` planning mode to PureFFT that picks a plan **structurally** (by size-class
heuristic, no timing), cutting the first-call plan-construction JIT cost from seconds to milliseconds, at
some runtime-optimality cost — mirroring FFTW's `ESTIMATE` vs `MEASURE`.

## Motivation (measured)

PureFFT's `autoplan` is effectively **always-MEASURE**: for each size it constructs *all* 6–7 candidate plan
types (codelet, GenPP, GenPP-composite, Bluestein, smooth-tree, AvxMixedRadix W4, W8) and times each, keeping
the fastest. Excellent runtime, but the **first call to a new size JIT-compiles every candidate's `@generated`
codelets** — measured **2.4 s (n=4620), 4.0 s (n=27720)** on a warm session; the *second* call to the same
size is ~10 ms (the timing itself is cheap — the cost is compilation of all candidates). For a general-purpose
library this first-call latency is a real UX wart vs FFTW's near-instant `ESTIMATE`.

ESTIMATE builds **exactly one** plan (the structurally-chosen one), so first-call compiles ~1 tree instead of
~7 → fast planning.

## Constraints & design tenets

- **Mirror FFTW for frictionless transition.** Flag names (`ESTIMATE`/`MEASURE`), the `flags` kwarg, and
  AbstractFFTs `plan_fft` compatibility match FFTW.jl so an FFTW user's call sites work verbatim. Diverge
  ONLY where PureFFT is strictly better *and* the divergence is itself a smooth transition (see: fallback).
- **Opt-in first; MEASURE stays the default.** Protects the strict 0.96×-vs-FFTW-MEASURE parity culture and
  the default runtime while PureFFT's (new, unproven) heuristic earns trust. Flip the default to ESTIMATE
  later, once the heuristic is measured near-MEASURE on the common classes (see Future work).
- **Dependency-free, pure Julia** (project vision) — the classifier is plain size arithmetic; no new deps.
- **MEASURE path byte-unchanged** — ESTIMATE is a purely additive branch; it cannot regress the default.
- **Correct always, optimal best-effort** — an ESTIMATE plan uses the same kernels as MEASURE, so it is
  numerically identical; it may just be a non-fastest (but valid) choice. Never a *wrong* plan.

## Design

### API surface (FFTW-mirroring)

- `@enum PlanRigor ESTIMATE MEASURE`, exported as `PureFFT.ESTIMATE` / `PureFFT.MEASURE`.
- `flags` kwarg (FFTW's name), default `MEASURE`, on both entry points:
  - native: `plan_pfft(x; flags = MEASURE, variant, inverse)`
  - AbstractFFTs / FFTW-facing: `plan_fft(x; flags = MEASURE, …)` — so `plan_fft(x; flags = PureFFT.ESTIMATE)`
    works exactly like the FFTW.jl call site.
- `flags` threads down to `autoplan(Complex{T}, n; inverse, flags = MEASURE)`.

### Data flow — one additive branch in `autoplan`

```julia
function autoplan(::Type{Complex{T}}, n; inverse = false, flags = MEASURE) where {T}
    if flags == ESTIMATE
        p = _estimate_plan(Complex{T}, n; inverse)
        !isnothing(p) && return p          # structural hit → build ONE plan, no timing
    end                                     # miss → fall through to the unchanged MEASURE competition
    <existing MEASURE body, verbatim>
end
```

MEASURE is untouched (default, parity-safe). ESTIMATE is additive with a safe fallback.

### The classifier — `_estimate_plan(Complex{T}, n; inverse) → AbstractFFTPlan or nothing`

Cheap predicates classify first (no construction); then build the ONE chosen plan. Order matters (a
prime-square is also non-pow2, so GenPP is checked before smooth):

| # | condition (cheap check)                          | build                                  |
|---|--------------------------------------------------|----------------------------------------|
| 1 | `ispow2(n)`                                       | `Radix4AvxPlan` (wrapped `AutoPlan`)   |
| 2 | prime `n` ≥ `RADER_MIN_P` and smooth `n−1` (Rader) | `RaderPlan`                          |
| 3 | `_gen_pp_prime(n)` ≠ nothing (Float64 only)      | `GenPPCodeletPlan`                     |
| 4 | `_gen_pp_composite(n)` ≠ nothing (Float64 only)  | `GenPPCompositePlan`                   |
| 5 | `n` is 2·3·5-smooth                               | `AvxMixedRadixPlan` (W4); `nothing` if `plan_tree` declines |
| 6 | else                                             | `nothing` → MEASURE fallback           |

Reuses `autoplan`'s existing routing predicates verbatim (`_gen_pp_prime`, `_gen_pp_composite`, the Rader
condition, `AvxMixedRadixPlan`'s `plan_tree`), so ESTIMATE and MEASURE agree on *which* plan applies — they
differ only in whether the choice among applicable candidates is timed.

### Deliberate, documented tradeoffs (correct, not always fastest)

- **Smooth sizes → always W4.** ESTIMATE skips the W4-vs-W8 timing call, so it forfeits W8's ~1.05–1.2×
  win on the sizes where W8 wins. W4 is correct and the reliable general choice.
- **All pow2 → Radix4Avx.** Forfeits the `B256/B512` odd-power win on 512/2048 (~0.9× vs the measured best).
  Radix4Avx is correct and competitive elsewhere.

Both are exactly the ambiguous, timing-resolved calls; predicting them is Future work (cost model).

### Error handling / safety

- `_estimate_plan` never throws — uncertainty → `nothing` → MEASURE fallback.
- Non-`ESTIMATE`/`MEASURE` `flags` is a clean `@enum` type error at the call site.
- ESTIMATE reuses MEASURE's kernels → numerically identical output.

## Testing (`test/estimate_tests.jl`, ReTestItems)

1. **Classifier routing** — `_estimate_plan(n)` returns the expected plan *type* per class: 1024→Radix4Avx,
   a Rader prime→Rader, a prime-square→GenPP, 720→AvxMixedRadix W4, and an unclassified size → `nothing`.
2. **Correctness** — across a spread of all classes, `flags = ESTIMATE` output == reference DFT (exact-kernel
   classes reach ≤1e-14; the Bluestein/Rader **fallback** sizes measure ~3e-12 against a naive O(n²) reference,
   so the test gate is ≤1e-11 — still 5 orders below any real error, diagnostic of a correct transform),
   forward + inverse round-trip.
3. **Fallback safety** — an unclassified size under ESTIMATE still yields a correct plan (via MEASURE).
4. **No regression** — MEASURE default byte-unchanged; full suite green.

Planning *speed* is asserted structurally (the ESTIMATE plan's concrete type equals the classifier's single
pick ⇒ one plan built), NOT via a flaky wall-clock test in CI; the first-call speedup gets a `bench/` note.

## Files (clean unit boundaries — `_estimate_plan` is a pure classifier; the rest is thin wiring)

- `src/estimate.jl` *(new)* — `@enum PlanRigor` + `_estimate_plan(Complex{T}, n; inverse)`
- `src/autotune.jl` — `flags` param on `autoplan` + the single ESTIMATE branch
- `src/plan.jl` / `src/abstractfft.jl` — `flags` kwarg (default `MEASURE`) threaded to `autoplan`
- `src/PureFFT.jl` — include `estimate.jl`; export `ESTIMATE` / `MEASURE`
- `test/estimate_tests.jl` *(new)*

## Out of scope / future work

- **Flop-cost model (FFTW-style), scoring all candidates without timing** — the route to "structural for
  every size" (no fallback) and to **flipping the default to ESTIMATE** to fully match FFTW. Bigger effort;
  this session showed how easily such predictions miss (the W8 sizes). Deliberately deferred.
- **A smarter W4/W8 and pow2 sub-range heuristic** — folds into the cost-model work.
- **Plan cache / wisdom** — NOT this project: the 2nd call to a size is already ~10 ms; the wart is the
  first compile, which a cache does not fix.

## Success criteria

- `plan_fft(x; flags = PureFFT.ESTIMATE)` / `plan_pfft(x; flags = ESTIMATE)` build in ≪ MEASURE first-call
  time (one plan, not seven) on the common classes (pow2, smooth, prime-square, large-prime).
- ESTIMATE output correct vs reference DFT on all covered classes + fallback sizes (≤1e-14 for exact-kernel
  classes; ≤1e-11 for the Bluestein/Rader fallback, which accumulates ~3e-12 vs a naive reference).
- MEASURE remains the default and byte-unchanged; full test suite green; no parity regression.
