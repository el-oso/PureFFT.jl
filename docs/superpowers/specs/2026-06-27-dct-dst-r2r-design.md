# DCT / DST (real-to-real transforms) — design

**Date:** 2026-06-27
**Status:** approved design, pre-implementation
**Goal:** add the eight FFTW real-to-real (r2r) transforms — DCT-I/II/III/IV and DST-I/II/III/IV — to
PureFFT, reaching FFTW r2r feature parity for 1-D, with FFTW as the golden reference and Julia's
runtime-specialization strengths exploited where they beat FFTW's fixed codelet set.

## 1. Scope

**In scope (this spec):**
- All 8 r2r kinds, 1-D, `Float64` and `Float32`, any length `N` (no even/pow2 restriction).
- FFTW-exact API names + semantics (unnormalized `r2r`; orthonormal `dct`/`idct`), defined
  independently of FFTW.jl (no FFTW.jl dependency).
- `Result`-first error handling (ErrorTypes.jl) with thin throwing shims for drop-in compatibility.
- Zero-allocation, dispatch-free hot path (project requirement).

**Out of scope (separate tracks / future):**
- N-dimensional r2r (rides on the future N-D FFT track).
- The Hartley transform (`DHT`) and half-complex (`R2HC`/`HC2R`) FFTW kinds — DCT/DST only here.
- Extending `prfft` to odd lengths (odd-`N` type-II/III/IV uses a complex-FFT fallback in v1; see §9).
- `@generated` direct small-`N` r2r codelets (designed-for as v2; see §9).

## 2. Background — the FFTW reference

FFTW exposes r2r transforms via kind flags; PureFFT mirrors the exact names. FFTW's numbering:

| FFTW kind | transform | logical length |
|---|---|---|
| `REDFT00` | DCT-I  | 2(N−1) |
| `REDFT10` | DCT-II ("the DCT") | 2N |
| `REDFT01` | DCT-III ("the IDCT") | 2N |
| `REDFT11` | DCT-IV | 4N |
| `RODFT00` | DST-I  | 2(N+1) |
| `RODFT10` | DST-II | 2N |
| `RODFT01` | DST-III | 2N |
| `RODFT11` | DST-IV | 4N |

FFTW implements these in its `reodft` layer by **reducing each to a real DFT** plus pre/post-processing
(`reodft00e-r2hc`, `reodft010e-r2hc`, `reodft11e-r2hc`), and additionally ships hardcoded direct codelets
for small sizes selected by its timing planner. We take the reduction math as the reference and improve on
the *mechanism* (§4, §9).

Unnormalized FFTW conventions (the `r2r` API reproduces these exactly):
- `REDFT10` then `REDFT01` = `2N · x`; `REDFT11` is self-inverse up to `2N`; `REDFT00` self-inverse up
  to `2(N−1)`.
- `RODFT10` then `RODFT01` = `2N · x`; `RODFT11` self-inverse up to `2N`; `RODFT00` self-inverse up to
  `2(N+1)`.

## 3. Architecture overview

A single new module file **`src/r2r.jl`** (mirroring `src/rfft.jl`), `include`d from `PureFFT.jl`:

1. **Kind singleton types** — `REDFT00 … RODFT11` (exported with the FFTW names). Used as a type
   parameter so plans are concrete and dispatch-free.
2. **`R2RPlan{K,T,P}`** — kind `K`, element type `T`, inner plan type `P`. Holds the inner FFT plan
   (built at the kind's natural size), precomputed pre/post twiddle tables, preallocated work buffers,
   and the normalization scale. Zero-allocation hot path.
3. **Constructors** — `tryplan_r2r(x, kind) → Result{R2RPlan,R2RError}` (and `tryplan_dct`/`tryplan_idct`)
   plus throwing shims `plan_r2r`/`plan_dct`/`plan_idct`.
4. **Apply** — `r2r`/`r2r!`/`dct`/`idct` (+ `try*` variants), `p * x`, `mul!(y,p,x)`, `inv(p)`.
5. **Per-kind `_pre!` / `_post!`** — the only per-type code; `@generated`, specialized on `(K, N)`.

Reduction strategy = FFTW's (approach **C**): type-II/III/IV via a **same-size** FFT + pre/post twiddle
(Makhoul); type-I via a real FFT of the natural **2(N∓1)** extension.

## 4. Julia strengths over FFTW's mechanism

FFTW ships a *fixed, build-time* set of codelets; Julia specializes at *runtime* for any size/type:

1. **`@generated` per-`(kind, N)` pre/post.** The reorder + twiddle emits straight-line code specialized
   to the exact kind and size (the `src/codelets.jl` idiom) — no runtime tuple indexing (project rule
   #1), no per-element branch.
2. **`@generated` direct small-`N` codelet for *any* `N` (v2).** FFTW's direct r2r codelets exist only
   for a bounded built-in size set; Julia generates one on demand for whatever `N` is asked. A plan-time
   selector (reusing `autoplan`'s timing machinery) times direct-codelet vs reduce-to-FFT and keeps the
   faster — mirroring FFTW's planner but unbounded in size. This is a potential *beat*, not just parity.
3. **Kind as a type parameter** → the whole plan is concrete/dispatch-free, specialized end-to-end.
4. **One generic-over-`T` implementation** → `Float32` for free, with the AVX path following from `T`
   (as just done for the complex FFT). FFTW compiles separate codelets per precision.

## 5. API surface

Exact FFTW names, defined independently (no FFTW.jl dep). `using PureFFT` makes `dct(x)` a drop-in; with
both packages loaded, Julia requires `PureFFT.dct` / `FFTW.dct` (normal disambiguation, no silent error).

**Kind constants:** `REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11`.

**Result-first core (ErrorTypes.jl) — the preferred, throw-free path:**
- `tryplan_r2r(x, kind) → Result{R2RPlan, R2RError}`
- `tryplan_dct(x) / tryplan_idct(x) → Result{R2RPlan, R2RError}`
- `tryr2r(x, kind) / trydct(x) / tryidct(x) → Result{Vector, R2RError}`

**Throwing shims (FFTW drop-in) — thin `@unwrap_or`-throw wrappers over the above:**
- `plan_r2r(x, kind)`, `plan_dct(x)`, `plan_idct(x)` → plan or `throw(ArgumentError)`
- `r2r(x, kind)`, `r2r!(x, kind)`, `dct(x)`, `dct!(x)`, `idct(x)`, `idct!(x)`

**Normalization:**
- `r2r`/`plan_r2r` — **unnormalized** (exact FFTW C convention).
- `dct`/`idct` — **orthonormal** DCT-II/III (scipy `norm="ortho"` / FFTW.jl `dct`): `idct(dct(x)) == x`,
  orthogonal. The scale is derived from the unnormalized REDFT10/REDFT01 and verified bit-exact against
  FFTW.jl's `dct`.

`inv(plan_r2r(x, REDFT10))` returns the `REDFT01` plan scaled by `1/2N`; self-inverse kinds (IV, and the
type-I pair) return themselves scaled by their FFTW factor — giving `\` / `ldiv!` for free.

## 6. Per-kind reduction algorithms

`Wc(a) = cos(πa)`, twiddles precomputed in the plan. All output is real.

| kind | inner FFT | method |
|---|---|---|
| `REDFT10` (DCT-II) | real, size N (even) | even/odd input reorder `v=[x₀,x₂,…,x_{N-1},…,x₃,x₁]` → real FFT → `yₖ = 2·Re(e^{−iπk/2N} · V̂ₖ)` |
| `REDFT01` (DCT-III) | real, size N (even) | inverse of II: pre-twiddle the half-spectrum, real-inverse FFT, inverse reorder |
| `REDFT11` (DCT-IV) | real/complex, size N | pre-twiddle (`e^{−iπj/2N}`-class) → FFT → post-twiddle (`e^{−iπ(2k+1)/4N}`), `Re` |
| `RODFT10` (DST-II) | real, size N | reflected/sign-flipped DCT-II (`yₖ` from the imaginary combination) |
| `RODFT01` (DST-III) | real, size N | inverse of DST-II |
| `RODFT11` (DST-IV) | real/complex, size N | sine variant of DCT-IV (pre+post twiddle) |
| `REDFT00` (DCT-I) | real, size 2(N−1) | even extension `e=[x₀,…,x_{N-1},x_{N-2},…,x₁]` → `prfft` → `yₖ = Re(Êₖ)` |
| `RODFT00` (DST-I) | real, size 2(N+1) | odd extension `o=[0,x₀,…,x_{N-1},0,−x_{N-1},…,−x₀]` → `prfft` → `yₖ = −Im(Ôₖ₊₁)` |

The exact twiddle/boundary terms for each kind are derived from the FFTW definitions during
implementation and pinned by the bit-exact tests (§10). Type-I extension lengths 2(N∓1) are always even,
so type-I always uses the real-FFT (`prfft`) inner. Type-II/III/IV with even `N` use `prfft(N)`; odd `N`
uses a complex-FFT fallback (§9).

## 7. Plan struct & data flow

```julia
struct R2RPlan{K, T, P}        # K = kind singleton type; T = Float64/Float32; P = inner plan type
    n::Int
    inner::P                   # forward/inverse real FFT (plan_prfft / plan_pirfft) at the kind's natural
                               # size [even/real], or plan_pfft(N) [odd-N complex fallback]. Direction
                               # follows the kind (e.g. DCT-II forward, DCT-III inverse).
    pre::Vector{Complex{T}}    # precomputed pre-twiddles (kind-specific; may be empty)
    post::Vector{Complex{T}}   # precomputed post-twiddles
    rbuf::Vector{T}            # real work buffer (extension / reordered input)
    cbuf::Vector{Complex{T}}   # half-spectrum work buffer
    scale::T                   # 1 for r2r; orthonormal factor for dct/idct
end
```

Hot path `mul!(y, p::R2RPlan{K}, x)`:
1. `_pre!(K, p, x)` — `@generated`, specialized on `(K, n)`: fill `rbuf` (extension for type-I; reorder
   ± pre-twiddle for II/III/IV).
2. inner FFT (`prfft`/`pfft` already autotuned and zero-alloc).
3. `_post!(K, p, y)` — `@generated`: combine the half-spectrum with `post` twiddles, apply boundary/sign
   terms and `scale`, write real `y`.

Preallocated buffers ⇒ zero heap allocation, exactly like `RealFFTPlan`. `r2r(x,kind) = mul!(similar(x),
plan_r2r(x,kind), x)`; the `!` forms reuse `x`'s storage where shapes permit.

## 8. Error handling

`R2RError` (ErrorTypes) enumerates: unsupported kind, size-too-small for the kind, non-float element
type that cannot promote. Validation lives in `tryplan_*` (cold path) and returns `Result`:
- `REDFT00` (DCT-I) requires `N ≥ 2` (logical 2(N−1)); `RODFT00` requires `N ≥ 1` — matching FFTW.
- All other kinds accept `N ≥ 1`.
- Real input (`AbstractVector{<:Real}`); integer input promoted to `float`. Strided/views are copied by
  the pre-step, so they are handled transparently.

The throwing shims are one-liners: `plan_r2r(x,k) = @unwrap_or tryplan_r2r(x,k) e -> throw(ArgumentError(string(e)))`.
A dimension mismatch in `*`/`mul!` is a cheap length check that throws `DimensionMismatch` (apply-time,
not hot-loop) — kept as a throw because it is a programmer error and FFTW does the same.

## 9. Performance

- **Real-FFT inner is primary** (FFTW's ~2× lever — the input is real): even-`N` II/III/IV and all
  type-I use the real FFT (`prfft`/`pirfft` per direction). Odd-`N` II/III/IV fall back to a length-`N`
  complex `pfft` (correct, ~2× slower for those sizes); closing this = extend `prfft` to odd lengths
  (future, noted in §1 out-of-scope).
- **`@generated` pre/post** keeps the reduction overhead straight-line and dispatch-free.
- **Inner FFT is `autoplan`-tuned**, so the reduced transform inherits at/above-FFTW inner speed.
- **v2 — `@generated` direct small-`N` codelet + plan-time selection.** The plan structure is designed so
  a direct r2r codelet is an alternative "inner strategy" the constructor can time against reduce-to-FFT
  (Julia strength #2). Not required for functional parity; the explicit beat-FFTW-on-small-`N` play.
- Known limitation (acceptable, matches the existing complex-FFT behaviour): for *tiny* `N` the pre/post
  overhead is relatively larger than FFTW's direct codelet until v2 lands.

## 10. Testing strategy (FFTW = golden reference)

`test/r2r_tests.jl` (ReTestItems), using FFTW.jl in the test env:
- **Bit-exact vs FFTW.jl `r2r`** for all 8 kinds, `Float64`+`Float32`, across `N` covering
  even/odd/prime/pow2/non-pow2 and small edge sizes — rel-err ≤ `tol(T)` (`1e-12` F64, `1e-4` F32).
- **Independent ground truth:** the naive O(N²) DCT/DST summation for small `N` (not only FFTW).
- **`dct`/`idct`:** match FFTW.jl `dct`/`idct`; orthonormal round-trip `idct(dct(x)) ≈ x`.
- **Inverse/scale relations:** II↔III, IV self-inverse, type-I self-inverse, with the FFTW scale factors;
  `inv(p)`/`\` round-trips.
- **Error paths:** `REDFT00` with `N=1` → `Result` err / `ArgumentError`; invalid kind; the `try*`
  variants return the right `Err`.
- **Zero-alloc + dispatch-free** hot path via `@test_opt` / AllocCheck (project requirement).

### 10.1 Performance gates

Two gates, both required before a kind is considered done. Measured with the standing methodology
(median, central-68% spread, `taskset -c 2`, in-place reps; CLAUDE.md §6).

1. **No-regression gate (the strict rule — protect what we already achieved).** Adding DCT/DST must not
   regress *any* existing transform. The existing complex-FFT and real-FFT perf-regression guards
   (`test/` relative-perf tests + `bench/run_compare.jl` → `compare.json`) must stay green: the
   `ComplexF64`/`ComplexF32` and `rfft` medians after the change are within run-to-run noise (≥ their
   pre-change values, modulo the documented ±7% ratio noise). Since DCT/DST is additive (`src/r2r.jl`,
   building on the existing plans), this is primarily a guard against accidentally perturbing shared hot
   paths — but it is a hard gate, not a hope.

2. **Parity gate vs FFTW (the 0.96× rule).** Per kind, `Float64`+`Float32`: `fftw_median /
   purefft_median ≥ 0.96` (the project parity rule, relative to the golden reference). Because the inner
   FFT is *already* at/above FFTW and DCT reduces to it, a kind that lands materially below 0.96× signals
   avoidable pre/post overhead and must be investigated (re-measured in the full kernel — not chased
   below the ±7% floor), not waved through. Reproducible via a new `bench/run_compare_r2r.jl` →
   `bench/results/compare_r2r.json` → `bench/plot_compare_r2r.jl`, mirroring the F32 pipeline.

## 11. Phasing (incremental rollout — reduce risk)

The error-handling shape (`Result` core + throwing shims) is present from the first kind; risk is in the
transform math, so kinds land one group at a time, each bit-exact-verified before the next:

1. **DCT-II + DCT-III** (`REDFT10`/`REDFT01`) — the canonical pair, the `dct`/`idct` wrappers, the plan
   plumbing, the `Result`/shim layer, the test harness.
2. **DCT-IV** (`REDFT11`) — the dual-twiddle pattern.
3. **DST-II/III/IV** (`RODFT10`/`01`/`11`) — the sine analogues of 1–2.
4. **Type-I pair** (`REDFT00`/`RODFT00`) — the 2(N∓1) extension route.
5. (v2, optional) `@generated` direct small-`N` codelet + plan-time selection.

## 12. Open questions

None blocking. The odd-`N` `prfft` extension and the v2 direct codelet are explicitly deferred; the exact
per-kind twiddle constants are derived during step 1–4 and pinned by the bit-exact tests.
