# DCT/DST Phases 2–4 (the remaining 6 r2r kinds) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** add the remaining 6 FFTW real-to-real transforms — DCT-IV, DST-IV, DST-II, DST-III, DCT-I, DST-I — to PureFFT, reaching full FFTW 1-D r2r parity (Phase 1 shipped DCT-II/III).

**Architecture:** Each kind is one `_build_r2r(::KIND_T, T, n)` (precompute pre/post twiddle tables + the inner FFT plan at the kind's natural size) + one `_apply!(::R2RPlan{KIND_T,T,P}, y, x)` (the FFTW reduction), routed in `tryplan_r2r`, plus `inv`. All on the EXISTING shared `R2RPlan{K,T,P}` machinery in `src/r2r.jl` — no new infrastructure. Bit-exact vs FFTW.jl's `r2r` is the gate for every kind.

**Tech Stack:** Julia; the existing `R2RPlan` + 1-D complex/real FFT plans (`plan_pfft`, `plan_prfft`, `plan_pirfft`); ErrorTypes.jl (`Result`/`tryplan_r2r`); ReTestItems + FFTW.jl (golden reference, test-only).

## Global Constraints

- **No Python.** **`isnothing(x)`**, never `=== nothing`. Commit author `15278831+el-oso@users.noreply.github.com`; commit body ends `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Branch `feat/dct-dst-phase2`.
- **Mirror Phase 1 exactly** (`src/r2r.jl` REDFT10/REDFT01): per-kind `_build_r2r` returns `Result{R2RPlan,R2RError}`; `_apply!` is monomorphic + `@test_opt`-clean; pre/post twiddle tables precomputed in the plan (zero-alloc hot path); even-N uses the real-FFT route (`plan_prfft`/`plan_pirfft`), odd-N (and type-IV/I where the natural size needs it) uses a complex-FFT route — follow how REDFT10 has both a `RealFFTPlan` and an `AbstractFFTPlan` `_apply!` method.
- **FFTW unnormalized conventions are the contract** (spec §2): REDFT11/RODFT11 self-inverse up to 2N; REDFT00 self-inverse up to 2(N−1); RODFT00 up to 2(N+1); RODFT10∘RODFT01 = 2N.
- **Bit-exact gate:** every kind tested against `FFTW.r2r(x, FFTW.<KIND>)`, rel-err ≤ 1e-12 (F64) / 1e-4 (F32), for a range of N (even **and** odd) — FFTW.jl is the golden reference.
- **Test scope (per [[test-scope-filtered]]):** iterate with `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. `src/r2r.jl` is self-contained (not 1-D-planner core), so the filtered `["r2r"]` run is the gate; full `Pkg.test()` only as the final pre-merge gate.

### Reference: the spec's per-kind reduction (spec §2 table + lines 113–121)
| kind | FFTW unnormalized formula | reduction |
|---|---|---|
| REDFT11 (DCT-IV) | `y_k = 2·Σ_j x_j cos(π(2j+1)(2k+1)/(4N))` | pre-twiddle `e^{−iπj/2N}` → size-N FFT → post-twiddle `e^{−iπ(2k+1)/4N}`, take Re |
| RODFT11 (DST-IV) | `y_k = 2·Σ_j x_j sin(π(2j+1)(2k+1)/(4N))` | sine variant of DCT-IV (same twiddles, Im/sign combination) |
| RODFT10 (DST-II) | `y_k = 2·Σ_j x_j sin(π(2j+1)(k+1)/(2N))` | reflected/sign-flipped DCT-II |
| RODFT01 (DST-III) | `y_k = (−1)^k x_{N−1} + 2·Σ_{j} x_j sin(π(j+1)(2k+1)/(2N))` | structural inverse of DST-II |
| REDFT00 (DCT-I) | `y_k = x_0 + (−1)^k x_{N−1} + 2·Σ_{j=1}^{N−2} x_j cos(πjk/(N−1))` | even extension `e=[x_0..x_{N−1},x_{N−2}..x_1]` (size 2(N−1)) → `prfft` → `y_k = Re(Ê_k)` |
| RODFT00 (DST-I) | `y_k = 2·Σ_j x_j sin(π(j+1)(k+1)/(N+1))` | odd extension `o=[0,x_0..x_{N−1},0,−x_{N−1}..−x_0]` (size 2(N+1)) → `prfft` → `y_k = −Im(Ô_{k+1})` |

**Methodology (how to get the twiddles right):** write the `FFTW.r2r` bit-exact test FIRST, then converge the exact pre/post twiddle factors + reorder against it — exactly how Phase 1's REDFT10/01 were built. A naive direct-formula reference (`y_k = Σ ...`) in the test, computed independently, guards against copying FFTW's bug-for-bug; assert against BOTH FFTW and the naive sum.

---

### Task 1: DCT-IV (REDFT11)

**Files:** Modify `src/r2r.jl` (add `_build_r2r(::REDFT11_T,…)` + `_apply!(::R2RPlan{REDFT11_T},…)` + `inv`; route in `tryplan_r2r`). Test: `test/r2r_tests.jl`.

**Interfaces:**
- Consumes: `R2RPlan{K,T,P}`, `tryplan_r2r`, `plan_pfft`, the `_apply!`/`_build_r2r` idiom from REDFT10.
- Produces: `_build_r2r(::REDFT11_T, ::Type{T}, n)`, `_apply!(::R2RPlan{REDFT11_T,T,P},…)`, `Base.inv(::R2RPlan{REDFT11_T})` (self-inverse: `inv = (1/2N)·REDFT11`).

- [ ] **Step 1: Write the failing test** (`test/r2r_tests.jl`):
```julia
@testitem "DCT-IV (REDFT11) bit-exact vs FFTW + self-inverse" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12
    naive_dct4(x) = [2*sum(x[j+1]*cos(pi*(2j+1)*(2k+1)/(4length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for n in (1,2,3,4,5,8,9,16,17,32)
        x = randn(n)
        y = unwrap(PureFFT.tryr2r(x, REDFT11))
        @test maximum(abs.(y .- FFTW.r2r(x, FFTW.REDFT11)))/max(1,maximum(abs.(x))*n) < tol(Float64)
        @test maximum(abs.(y .- naive_dct4(x)))/max(1,maximum(abs.(x))*n) < tol(Float64)   # independent ref
        # REDFT11 self-inverse up to 2N
        @test maximum(abs.(unwrap(PureFFT.tryr2r(y, REDFT11)) .- 2n .* x))/max(1,maximum(abs.(x))*n) < tol(Float64)
    end
end
```

- [ ] **Step 2: Run it, expect fail** — `tryr2r(x,REDFT11)` returns `Err` ("unsupported kind") so `unwrap` errors.
Run: `julia --project=. -e 'using Pkg; Pkg.test(test_args=["r2r"])'`. Expected: FAIL.

- [ ] **Step 3: Implement.** Add to `src/r2r.jl` (mirror REDFT10's structure). DCT-IV: a length-N complex FFT with a pre-twiddle on input and post-twiddle on output (Makhoul type-IV). Skeleton (converge the exact constants against the test):
```julia
# ── DCT-IV (REDFT11): pre-twiddle → size-N FFT → post-twiddle, take Re ──
function _build_r2r(::REDFT11_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT11 needs n≥1")))
    inner = plan_pfft(Complex{T}, n; variant = :fast)
    pre  = [cispi(-T(j) / T(2n))        for j in 0:n-1]   # e^{−iπ j /2N}
    post = [cispi(-T(2k + 1) / T(4n))   for k in 0:n-1]   # e^{−iπ(2k+1)/4N}
    plan = R2RPlan{REDFT11_T, T, typeof(inner)}(n, inner, pre, post, T[], Vector{Complex{T}}(undef, n))
    return Result{R2RPlan, R2RError}(Ok(plan))
end
function _apply!(p::R2RPlan{REDFT11_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; c = p.cbuf
    @inbounds for j in 1:n; c[j] = p.pre[j] * x[j]; end       # pre-twiddle (real → complex)
    apply_unnormalized!(p.inner, c)                            # size-N FFT
    @inbounds for k in 1:n; y[k] = 2 * real(p.post[k] * c[k]); end  # post-twiddle, Re
    return y
end
```
**Converge against the test:** the exact pre/post phases (and any reorder) follow FFTW's REDFT11 reduction — adjust the constants/sign until the bit-exact + naive test passes. The `R2RPlan` field names (`pre`/`post`/`rbuf`/`cbuf`) must match Phase 1's struct (read it).
Route it: in `tryplan_r2r`, dispatch `REDFT11_T` to `_build_r2r(REDFT11, …)` (remove the "unsupported" path for this kind). Add `Base.inv(p::R2RPlan{REDFT11_T,T}) = ScaledR2RPlan(plan_r2r(Vector{T}(undef,p.n), REDFT11), one(T)/(2p.n))` (mirror REDFT10's inv).

- [ ] **Step 4: Run, expect pass** (all N, even+odd; self-inverse). Expected: PASS.

- [ ] **Step 5: Commit.**
```bash
git add src/r2r.jl test/r2r_tests.jl
git commit -m "feat(r2r): DCT-IV (REDFT11) — pre/post-twiddle reduction, bit-exact vs FFTW

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: DST-IV (RODFT11)

**Files/Interfaces:** as Task 1 but `RODFT11_T`. Produces `_build_r2r(::RODFT11_T,…)`, `_apply!`, `inv` (self-inverse).

- [ ] **Step 1: Failing test** — same shape as Task 1 with `naive_dst4(x)=[2*sum(x[j+1]*sin(pi*(2j+1)*(2k+1)/(4length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]`, vs `FFTW.RODFT11`, self-inverse up to 2N.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — DST-IV is the sine sibling of DCT-IV: same pre/post twiddle magnitudes, but the output takes the imaginary combination (and a reversed/sign-flipped index per FFTW's RODFT11). Mirror Task 1's `_build_r2r`/`_apply!`; converge the Im/sign against the test. Route `RODFT11_T` in `tryplan_r2r`; add `inv` (self-inverse, `1/2N`).
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** (`feat(r2r): DST-IV (RODFT11)…`).

---

### Task 3: DST-II (RODFT10)

**Files/Interfaces:** `RODFT10_T`. Produces `_build_r2r`, `_apply!`. (Inverse is RODFT01 — Task 4.)

- [ ] **Step 1: Failing test** — `naive_dst2(x)=[2*sum(x[j+1]*sin(pi*(2j+1)*(k+1)/(2length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]`, vs `FFTW.RODFT10`, even+odd N.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — DST-II mirrors DCT-II (REDFT10) with a sine combination: the same even/odd input reorder, real FFT, but the output is `−2·Im(W_k·V̂_k)` reversed (`y_k` from the imaginary part), per FFTW RODFT10. Mirror REDFT10's two `_apply!` methods (RealFFTPlan even-N + complex-FFT odd-N fallback). Route `RODFT10_T`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** (`feat(r2r): DST-II (RODFT10)…`).

---

### Task 4: DST-III (RODFT01)

**Files/Interfaces:** `RODFT01_T`. Produces `_build_r2r`, `_apply!`, `inv` (= `(1/2N)·RODFT10`).

- [ ] **Step 1: Failing test** — `naive_dst3(x)=[((-1)^k)*x[end] + 2*sum(x[j+1]*sin(pi*(j+1)*(2k+1)/(2length(x))) for j in 0:length(x)-2) for k in 0:length(x)-1]`, vs `FFTW.RODFT01`; plus **RODFT01∘RODFT10 = 2N·identity** round-trip.
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — structural inverse of DST-II (mirror REDFT01's relationship to REDFT10): pre-twiddle the half-spectrum with the sine combination, real-inverse FFT, inverse reorder. Route `RODFT01_T`; add `inv(::RODFT10_T)` returning the scaled RODFT01 and `inv(::RODFT01_T)` the scaled RODFT10 (the II↔III pair, like REDFT10/01).
- [ ] **Step 4: Run, expect pass** (incl. round-trip).
- [ ] **Step 5: Commit** (`feat(r2r): DST-III (RODFT01) + DST-II/III inv pair…`).

---

### Task 5: DCT-I (REDFT00)

**Files/Interfaces:** `REDFT00_T`. Produces `_build_r2r`, `_apply!`, `inv` (self-inverse up to 2(N−1)).

- [ ] **Step 1: Failing test** — `naive_dct1(x)=(N=length(x); [x[1]+((-1)^k)*x[N]+2*sum(x[j+1]*cos(pi*j*k/(N-1)) for j in 1:N-2) for k in 0:N-1])`, vs `FFTW.REDFT00` (n≥2), self-inverse up to 2(N−1). (FFTW requires N≥2 for REDFT00.)
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — even (symmetric) extension `e=[x_0,…,x_{N−1},x_{N−2},…,x_1]` of length `2(N−1)`, run `plan_prfft` on it (length 2(N−1)), take `y_k = Re(Ê_k)` for k=0..N−1. Precompute nothing twiddle-wise (pure real FFT of the extension); the plan holds the length-`2(N−1)` `prfft` plan + a real work buffer for the extension. Validate `n≥2` (else `Err`). Route `REDFT00_T`; `inv` self-inverse `1/(2(N−1))`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** (`feat(r2r): DCT-I (REDFT00) — even-extension real-FFT…`).

---

### Task 6: DST-I (RODFT00)

**Files/Interfaces:** `RODFT00_T`. Produces `_build_r2r`, `_apply!`, `inv` (self-inverse up to 2(N+1)).

- [ ] **Step 1: Failing test** — `naive_dst1(x)=(N=length(x); [2*sum(x[j+1]*sin(pi*(j+1)*(k+1)/(N+1)) for j in 0:N-1) for k in 0:N-1])`, vs `FFTW.RODFT00`, self-inverse up to 2(N+1).
- [ ] **Step 2: Run, expect fail.**
- [ ] **Step 3: Implement** — odd (antisymmetric) extension `o=[0,x_0,…,x_{N−1},0,−x_{N−1},…,−x_0]` of length `2(N+1)`, run `plan_prfft`, take `y_k = −Im(Ô_{k+1})` for k=0..N−1. Plan holds the length-`2(N+1)` `prfft` plan + the extension buffer. Route `RODFT00_T`; `inv` self-inverse `1/(2(N+1))`.
- [ ] **Step 4: Run, expect pass.**
- [ ] **Step 5: Commit** (`feat(r2r): DST-I (RODFT00) — odd-extension real-FFT…`).

---

### Task 7: Final integration — generic `r2r`/`plan_r2r`/`mul!`/docs

**Files:** `src/r2r.jl` (confirm generic entry covers all 8), `docs/src/` (r2r section / guide), ROADMAP.

- [ ] **Step 1: Test** — a parametric `@testitem` looping ALL 8 kinds through the public `r2r`/`plan_r2r`/`p*x`/`mul!` + `try*` API, each bit-exact vs `FFTW.r2r`, F64 + F32, even+odd N; confirm `tryplan_r2r` returns `Ok` for all 8 (no "unsupported" remains) and the throwing/Result variants agree.
- [ ] **Step 2: Run** `Pkg.test(test_args=["r2r"])` — all green.
- [ ] **Step 3: Docs** — update `docs/src/benchmarks.md`/guide r2r section (now all 8 kinds) and ROADMAP ("DCT/DST — Phase 1 DONE" → "all 8 r2r kinds DONE"). Note any perf measurement is the existing `bench/run_compare_r2r.jl` (extend it to the new kinds if cheap; the spec deferred perf-tuning to v2, so correctness/parity is the bar here).
- [ ] **Step 4: Full suite (pre-merge gate)** `julia --project=. -e 'using Pkg; Pkg.test()'` — green.
- [ ] **Step 5: Commit** (`feat(r2r): all 8 FFTW r2r kinds complete + docs`).

---

## Self-review notes
- **Spec coverage:** Tasks 1–6 cover the 6 remaining kinds (REDFT11/RODFT11/RODFT10/RODFT01/REDFT00/RODFT00); Task 7 the generic API + docs. Phase 1 (REDFT10/01) already shipped.
- **The twiddle constants in the skeletons are starting points** — the bit-exact-vs-FFTW + independent-naive-sum gate is the source of truth (Phase 1 methodology). This is intentional, not a placeholder: r2r reductions are converged against the reference, never copied blind.
- **inv pairing:** type-IV and type-I are self-inverse (scaled); II↔III pair invert to each other (Task 4 wires both directions) — mirrors REDFT10/01.
