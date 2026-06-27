# N-dimensional FFT (2-D/3-D/…) — design

**Date:** 2026-06-27
**Status:** approved design, pre-implementation
**Goal:** add N-dimensional complex and real FFTs to PureFFT — full FFTW generality (`fft(x, dims)` for any
rank, any `region`), drop-in via AbstractFFTs — built separably on the existing (≥FFTW) 1-D kernels, with a
trim-safe / dispatch-free hot path.

## 1. Scope

**In scope:**
- **Complex N-D** (c2c): `AbstractArray{Complex{T}}` of any rank `N`, transforming any subset `region ⊆ 1:N`.
- **Real N-D** (r2c / c2r): `rfft`/`irfft`/`brfft` over real arrays.
- `Float64` and `Float32` (the 1-D kernels are generic over both).
- Forward + inverse, out-of-place + (c2c) in-place.
- Drop-in via AbstractFFTs (`fft`/`ifft`/`bfft`/`rfft`/`irfft`/`brfft` + `plan_*`, `mul!`, `\`, `inv`) plus
  the prefixed `pfft`/`prfft` extended to arrays.

**Out of scope (separate tracks):**
- N-D real-to-real (DCT/DST over arrays) — rides on the 1-D r2r work.
- Multi-threading — single-thread (its own ROADMAP track; N-D is the first thing that will *want* it).
- Distinct twiddle/codelet work — N-D reuses the 1-D kernels unchanged.

## 2. The separable principle

The DFT is separable: an N-D transform over `region` = a sequence of **1-D FFTs along each dim in
`region`**, in any order, mathematically exact. This is also FFTW's strategy. So N-D is an *engine that
applies the existing fast 1-D kernels along each transformed dim* — no new butterfly/twiddle code.

## 3. Architecture & components

New file **`src/ndim.jl`**, `include`d from `PureFFT.jl`. One plan type:
```julia
struct NDPlan{T, D, P, N} <: AbstractFFTPlan{T}
    dims::NTuple{D, Int}           # transformed dims (sorted, deduped runtime region)
    plans::P                       # NTuple{D} of inner 1-D plans (autoplan-selected per transformed-dim size)
    sz::NTuple{N, Int}             # full array shape
    scratch::Vector{Complex{T}}    # one reused transpose/work buffer
    inverse::Bool
end
```
The inner 1-D plans are the existing autotuned kernels (already ≥FFTW), built **once per transformed-dim
size** at plan time (cold path). The hot path reuses them — the entire point of the separable design.

## 4. The apply — hybrid C, trim-safe

For each transformed dim `d` (in `dims`), view the column-major layout as
`inner = ∏ size[1:d-1]`, `n_d = size[d]`, `outer = ∏ size[d+1:N]`, computed with **integer loops over
`size(x, i)`** — never `size(x)[1:d-1]` (runtime tuple slice → type-unstable). Then:
- **`inner == 1` (d == 1):** the `n_d`-runs are already contiguous → apply the dim's 1-D plan to each of
  `outer` contiguous chunks (a `@view` of a *unit-stride* linear range — concrete `SubArray` type). **No
  transpose.**
- **`d > 1`:** cache-blocked transpose the `inner×n_d` block ↔ `n_d×inner` (reuse `blocked.jl`'s
  `_btranspose!`) so `d` becomes contiguous → batched contiguous 1-D → transpose back.

All addressing is **flat memory via `pointer`/linear offsets + integer arithmetic**. Forbidden (trim/
type-stability hazards with a runtime `dim`): `selectdim`, `mapslices`, `permutedims` with a runtime perm,
any dim-dependent `SubArray`/`ReshapedArray` type.

### 4.1 The `NTuple` risk (the #1 implementation hazard)
`plans::NTuple{D}` is **heterogeneous** — each dim's `autoplan` picks a different concrete plan type.
Indexing it with a runtime loop variable (`plans[i]` while looping the dims) is the **CLAUDE.md rule-#1
trap** (runtime tuple indexing boxes — 135× measured). Therefore the apply **`@generated`-unrolls over `D`**,
emitting straight-line code with *literal* indices (`plans[1]`, `plans[2]`, …) — same discipline as the
codelets. `D` is a type parameter of `NDPlan`. Watch-points (concretize if they bite):
- **Plan-type proliferation:** every distinct inner-plan combination is a distinct `NDPlan` type. Acceptable
  (plan construction is cold; apply is specialized) — but monitor compile cost.
- **`autoplan` returns a `Union`/`AutoPlan{T,…}`** (open ROADMAP item) — if that leaks into `plans`, the
  tuple isn't fully concrete and `@test_opt` will flag it. If so, store the concrete `best` plan (unwrap the
  `AutoPlan` wrapper) so each element is concrete.

## 5. API surface (full FFTW generality, drop-in)

- **Complex:** `AbstractFFTs.plan_fft(x::AbstractArray{<:Complex}, region; …)` + `plan_fft!`/`plan_bfft`/
  `plan_bfft!` → `NDPlan`. `fft(x, dims)`/`ifft`/`bfft`/`mul!`/`\`/`inv` route through. Region
  canonicalization replaces `_checkdim1`: accept any `region` (Int / tuple / range / `:`), sort, dedup,
  validate `⊆ 1:N` (else `ArgumentError`). The 1-D `AbstractVector` methods remain (a vector is rank-1).
- **Real:** `AbstractFFTs.rfft(x::AbstractArray{<:Real}, region)` + `plan_rfft`; `irfft(X, d, region)` /
  `brfft` / `plan_irfft`.
- **Prefixed:** `pfft`/`pfft!`/`ipfft`/`prfft`/`pirfft` extended to accept `AbstractArray` + `dims` (matches
  the project convention; thin wrappers over the AbstractFFTs path).
- **Normalization:** via AbstractFFTs — `ifft`/`irfft` use `normalization(real(T), sz, region)` =
  `1/∏(sz[r] for r in region)`; `inv`/`\` return the opposite-direction `NDPlan` scaled accordingly.

## 6. Real N-D

`rfft` over `region`: **r2c along `first(region)`** — the halved dim, matching FFTW/AbstractFFTs convention,
so order *matters here* (unlike c2c, where §4 may sort freely). Existing `plan_prfft` → half-spectrum, that
dim `n → n÷2+1`, real→complex; then **c2c along the remaining region dims** via the complex engine. The
output shape differs from the input only in `first(region)`. `irfft(X, d, region)` reverses it: c2c⁻¹ along
the remaining dims, then `pirfft` (length `d`) along `first(region)` — `d` (the original real length) is
required since it can't be recovered from `n÷2+1` alone. `brfft` = unnormalized `irfft`. `prfft` requires
even length on the r2c dim (1-D limitation); odd → documented error (same as 1-D today).

Real N-D uses its **own plan** (`RealNDPlan{T,…}`, parallel to `NDPlan`) carrying: the r2c dim
(`first(region)`) + its `prfft`/`pirfft` plan + original length, and the c2c dims + their plans. (It does
*not* subtype the c2c `NDPlan`; the apply is r2c-then-c2c, a different shape.)

## 7. Trim / dispatch gates

- `@generated` apply over `D` (literal plan indices, no runtime tuple index — §4.1).
- Flat-memory + integer arithmetic; reuse `_btranspose!` for the d>1 transpose.
- Verify on the hot path: **dispatch-free** (`@test_opt target_modules=(PureFFT,)`), **zero-alloc**
  (AllocCheck / `@allocated == 0` after warmup — the scratch is preallocated in the plan), **trim-safe**
  (TrimCheck `@validate`). One `@verify`-style assertion per routing path (c2c dim-1, c2c dim>1, r2c).

## 8. Testing

`test/ndim_tests.jl` (ReTestItems), FFTW.jl as the golden reference:
- **Bit-exact vs FFTW** for c2c and r2c/c2r across: ranks 2/3/4-D; regions = all-dims AND partial (e.g.
  `fft(x, 2)` on a matrix, `fft(x, (1,3))` on a 3-D array); sizes pow2 + non-pow2 + mixed-per-dim; F64+F32.
  Rel-err ≤ `tol(T)` (1e-12 F64, 1e-4 F32).
- **Round-trips:** `ifft(fft(x)) ≈ x`, `irfft(rfft(x), d) ≈ x`, `inv(p)*(p*x) ≈ x`.
- **API edges:** in-place `fft!`; `region` as Int/tuple/range/`:`; invalid region → `ArgumentError`; rank-1
  array still routes to the 1-D path; non-even r2c dim → documented error.
- **Hot-path gates** (§7): `@test_opt`, AllocCheck, TrimCheck on one plan per routing path.

## 9. Performance gate

- **Reference: FFTW only.** RustFFT has **no N-D** transforms, so it is not a reference here (unlike the
  1-D gate which is vs both). FFTW's single-thread N-D is the bar and is the *same shape* as ours (separable
  1-D + cache-blocked transposes), so parity is plausible — but FFTW's transposes are mature, so the
  **transpose efficiency is where the gate is won or lost**.
- **Gate: `fftw_median / purefft_median ≥ 0.96`** per benchmarked shape (representative 2-D/3-D, pow2 +
  non-pow2, F64+F32), single-thread, in-place, planning excluded, `taskset -c 2`, median + central-68%.
- Honest expectation: this is the hard part. Per the `enforce-parity-gate` rule, N-D is **not "done"** until
  every benchmarked shape clears 0.96× vs FFTW; any that don't are flagged **"below gate — OPEN"** (not
  softened), and closing them (better transpose / batched-1-D kernel) is the explicit follow-up. Reproducible
  pipeline: `bench/run_compare_ndim.jl` → `bench/results/compare_ndim.json` → `bench/plot_compare_ndim.jl`.

## 10. Phasing (incremental, bit-exact before perf)
1. **Complex N-D, dim-1 fast path** (the no-transpose case) — engine + plan + AbstractFFTs c2c + the
   `@generated`-over-`D` apply + trim/dispatch gates. Bit-exact for regions that only touch dim 1
   (and the rank-1 vector path).
2. **Complex N-D, d>1 via transpose** (`_btranspose!` integration) — full c2c generality, bit-exact.
3. **Real N-D** (r2c first dim + c2c rest; `irfft`/`brfft`) — bit-exact, round-trips.
4. **Perf + publish** — `run_compare_ndim.jl`, measure vs FFTW, close/flag per the gate, ROADMAP + docs.

## 11. Risks
- §4.1 `NTuple` heterogeneity / `AutoPlan`-`Union` leak — the #1 hazard; resolved by `@generated`-unroll +
  unwrapping to concrete inner plans. If plan-type proliferation hurts compile, revisit.
- Perf vs FFTW's mature transposes — may not reach 0.96× on the first cut; surfaced, not hidden.
- `prfft`'s even-length-only limit propagates to r2c dim-1 (odd → error); documented.
