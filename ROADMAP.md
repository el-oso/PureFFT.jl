# PureFFT.jl — Roadmap

Status + planned work. This is the canonical, checked-in roadmap (human- and agent-readable). For the
"why/how" of finished work see git history, `docs/src/performance.md`, and `docs/src/benchmarks.md`.

## Where it stands today

- **Power-of-two**: AVX-512 (`Vec{8,Float64}`) radix-4 engine for the bulk; the **odd-power gap** (512=4⁴·2,
  2048=4⁵·2 sat ~0.88–0.90× of FFTW) is now **closed** by faithful **Butterfly256/512** monolith bases + an
  8xn radix chain (rustfft's scheme), timed in `autoplan`. PureFFT now at/above FFTW & RustFFT across the
  range (256/512 → ~1.3×, 2048/4096 → ~1.05×). `src/radix4_avx.jl`, `src/avxradix/{kernels,recursive,planner}.jl`.
- **Non-power-of-two**: a faithful mechanical port of RustFFT's AVX mixed-radix (`src/avxradix/`),
  routed by `autoplan` (`src/autotune.jl`) and exposed as `AvxMixedRadixPlan`. Bases are now B36 (2²·3²)
  **and B18 (2·3², a port of `Butterfly18Avx64`)** — B18 **closed the 2^odd·3²·5 oscillation** (90 → 2.0×,
  360 → 1.1×). The rest fall back to codelet / four-step / recursive / Rader / Bluestein. `autoplan` ranks
  candidates by **median** (was min — a lucky-outlier trap) and times them via a static **Tuple + `map`**
  (no abstract-`eltype` `Vector` — the old dynamic-dispatch / trim-hostile container). No size cliff.
- **AVX-512 for non-pow2 (the differentiator — RustFFT is AVX2-only)**: width-generic compute layer +
  W=8 kernels (`src/avxradix/width8.jl`, `AvxMixedRadixPlanW8`). `autoplan` times W=8 vs W=4 and keeps
  it only where it wins — so it's a strict improvement. W=8 beats W=4 **and** RustFFT on small
  compute-bound non-pow2 sizes (e.g. n=9216: ~43 vs ~37 GF/s).
- Tooling: AbstractFFTs plan interface, JET dispatch-free + TrimCheck trim-safe hot path, ReTestItems
  suite, relative perf-regression guards, σ-ribbon comparison plots.

## Open / planned

### AVX-512 (W=8) — extend the win beyond small compute-bound sizes
- **Large-size regression — RESOLVED (it was a codegen bug, not memory bandwidth).** The W=8 store loops
  used runtime tuple indexing (`for k in 1:N; t[k]`), a CLAUDE.md rule-#1 violation that boxed/slowed the
  larger sizes. Unrolling them with `@nexprs` (literal indices, matching the V4f kernels) eliminated it:
  same-tree W=8 now beats W=4 **1.03–1.11×** and is at/above RustFFT parity (**0.96–1.07×**) across L1→L3
  (768/9216/110592). The "memory-bandwidth-bound" hypothesis was wrong. Plus radix-9 at W=8 (`MR9W8`)
  enables radix-9-dominant sizes (576/5184/…). Remaining W=8 items below.
- **radix-9 below rust (~0.86×)** — radix-9 W=8 is correct + the best PureFFT option for low-2-count sizes
  (576 beats FFTW), but stays under rust — the intrinsic radix-9 shuffle-bound floor (transpose9 = 2×
  transpose4 + bridging shuffles). Only the `vpermt2pd` redesign (below) would lift it.
- **5-smooth coverage — DONE.** `transpose5`/`transpose9` derived at W=8 (each: one+ `transpose4` block +
  leftover rows + bridging shuffles, bit-exact-verified); `MR5W8`/`MR9W8` added and `plan_tree_w8` extended
  to 2·3·5-smooth. autoplan now routes 5-smooth sizes (2880/23040/46080) to W=8 — each **beats FFTW** and
  approaches rust (0.88–1.00×, 46080 at parity). radix-5/9 stay just under rust (shuffle-bound floor); only
  the `vpermt2pd` redesign (below) would close that.
- **Proper AVX-512 CPU detection** — currently `_HAS_AVX512` is read from `/proc/cpuinfo` at precompile
  (Linux). Replace with a portable feature query (e.g. HostCPUFeatures) so non-Linux + cross-machine
  precompile are handled cleanly; also skip *building* W=8 plans entirely when unavailable (today
  `_besttime` just never selects them).
- **AVX-512-native shuffle reduction (research)** — the non-pow2 kernels are shuffle-bound (Zen5's
  512-bit shuffle throughput doesn't double), so width alone gives only ~1.2× on the best radixes. A
  native redesign using `vpermt2pd` to cut shuffle count could lift this materially. High-risk research.

### Non-pow2 coverage / parity
- **More packed bases** — `Butterfly18` **DONE** (B18 = 2·3², closed 2^odd·3²·5); pow2 `Butterfly256/512`
  **DONE** (closed the odd-power gap). Still to port: `Butterfly{24,27,32}` (dual-width packed path) + the
  `avx_planner` base-selection (`base_fn`) so PureFFT's decompositions match RustFFT's per size. Lifts the
  remaining 3-heavy laggard sizes (~0.85–0.92×).
- **radix-9/12 floor ~0.90× of rust — it's ALGORITHMIC, not a Julia compiler issue.** An MWE comparing a
  matched radix-9 butterfly *and* a full radix-9 step (butterfly+twiddle+transpose) in Julia (SIMD.jl) vs
  Rust (`core::arch`) — see the standalone `julia-sched-mwe/` reproducer — found Julia compiles to identical
  asm and runs **at least as fast** at both levels (and LLVM 18/19/21 lower the same IR identically). So the
  gap is **rustfft's implementation** being more optimized, not Julia/LLVM scheduling. To close it: diff
  PureFFT's MR9/MR12 pass against rustfft's `Butterfly9`/mixed-radix source (decomposition, in-place /
  transpose / memory strategy) and adopt what's better — a PureFFT optimization, not a compiler chase.
- **MR2 / MR16** — currently sizes needing a radix-2 or radix-16 step fall back; add them for fuller
  smooth-size coverage.

### Breadth / type coverage (gaps for a *general* library vs the 1-D complex-`Float64` investigation)
- **DCT/DST (real-to-real) — Phase 1 DONE.** DCT-II (`REDFT10`) + DCT-III (`REDFT01`) are live
  for all N: even N uses the Makhoul real-FFT reduction (zero-alloc, dispatch-free); odd N falls
  back to a complex FFT (correctness, ~2× slower). API: `plan_r2r`, `r2r`, `mul!`, `dct`/`idct`
  (orthonormal, FFTW.jl drop-in), `plan_r2r \ x` (inverse). Bench harness:
  `bench/run_compare_r2r.jl` → `bench/results/compare_r2r.json` → `bench/plot_compare_r2r.jl`.
  Bench (no bounds checking): even-N PF/FFTW **1.45–2.71× for F64, 1.23–2.37× for F32**.
  Parity gate skips under `Pkg.test --check-bounds=yes` (overrides @inbounds in Julia loops
  but not FFTW's C — an artificial ~3× handicap); asserts in a fair env (check_bounds==0).
  The bench is the authoritative measurement; gate catches genuine regressions.
  Phases 2–4: DCT-IV, DST-II/III/IV, type-I pair — each adds `_pre!`/`_post!` per kind.
  See `docs/superpowers/specs/2026-06-27-dct-dst-r2r-design.md` (design spec) and
  `docs/superpowers/plans/2026-06-27-dct-dst-phase1.md` (Phase-1 plan; Phases 2–4 noted at its end).
- **Float32 — DONE (≥ 0.96× of FFTW AND RustFFT at every benchmarked size).** The AVX path is `Float32`-
  capable by genericizing the 4-complex kernels over `Vec{8,T}` (the hardware register follows from `T`;
  only the explicit FMA `llvmcall` is per-(N,T)). **Non-pow2** routes through the `V8f32 = Vec{8,Float32}`
  (256-bit AVX2) W=8 tree — beats FFTW & RustFFT (PF/FFTW 0.99–1.41). **Pow2** clears the gate at every size
  (256→65536: PF/FFTW & PF/Rust **1.00–1.49×**; F32 ≈ **1.3–1.8× the F64 GFLOP/s**), via three fixes layered
  in this order:
  1. **vectorized scratch transpose for all F32 sizes** (the n≤2048 cap was a Float64-tuned false premise;
     half-width data stays cache-friendly) → closed the even-power sizes;
  2. **radix-4 W8 kernel** (`MR4W8`) so the W=8 tree spans all pow2;
  3. **the decisive one — `B256W8`/`B512W8`**, faithful generic-`Vec{8,T}` ports of RustFFT's *f32*
     `Butterfly256Avx`/`Butterfly512Avx` (4-complex), the F32 equivalent of the `V4f` `B256`/`B512` monoliths
     that close the F64 odd-power gap. `plan_tree_w8` uses them as the pow2 base (rustfft's 8xn scheme);
     `autoplan` times the W=8 plan in the pow2 branch. This lifted the laggards (256: 0.99/0.88→1.37/1.20;
     512: 0.86/0.80→1.33/1.27; 8192: 0.83→1.02; 32768: 0.84→1.06) and F64 improved too (no regression).
  Bit-exact end-to-end (≤1.2e-8 F32). Null result kept for the record (`docs/src/performance.md` §17): a
  512-bit `Vec{16,Float32}` (8-complex) *base codelet* is bit-exact but measures identical to the 256-bit
  one (digit-reversed gather/scatter cancels the width gain) — the monolith port, not a wider base, was the
  answer. Minor follow-up: the n=64/128 fused in-register kernels are still Float64-only (those F32 sizes
  already clear the gate via the W=8 tree / radix4-AVX).
- **N-dimensional (2-D/3-D) FFT — DONE for complex (c2c); 14/16 benchmarked shapes ≥ 0.96× FFTW** (branch
  `feat/ndim-fft`). Full FFTW generality (any rank, any `region`), drop-in via AbstractFFTs + prefixed `pfft`.
  Separable on the ≥FFTW 1-D kernels, but fast via: (1) a **batched-strided kernel** (`src/ndim_batched.jl`) —
  each strided dim FFTs by vectorizing *across the contiguous batch*, **no transpose** (the transpose-per-dim
  path sat at ~0.25×; this is the big lever); (2) **`BatchedDim1`** gather-pack to fill the SIMD width on the
  contiguous dim for small Float32 shapes; (3) **F32 512-bit batch widening** (`Vec{16,Float32}`); (4) a 1-D
  **planner fix** admitting single-factor-of-3 sizes (48/96/384 = 2ᵏ·3 — a `p3≥2` guard was dumping them on a
  slow generic path; now routed to the fast mixed-radix kernels, beating FFTW — this also lifts 1-D
  non-pow2). Hot path dispatch-free + zero-alloc + trim-safe (`@generated`-over-rank apply, concrete per-dim
  descriptors). Bit-exact vs FFTW; full suite green.
  - **OPEN — `128×128` small-square** (F64 0.78, F32 ~0.92): a 256 KB L2-resident compute-bound square where
    FFTW's hand-tuned fused 2-D codelet wins; PureFFT's batched length-128 codelet is ~2× slower per pass and
    fused/transpose alternatives are slower (verified, tasks 6m/6t). Needs a dedicated FFTW-class length-128
    contiguous-batched codelet — niche; the only sub-gate shape.
  - **Remaining (not blocking c2c):** real N-D (`rfft`/`irfft` over arrays) per the spec follow-up; batched
    non-pow2 radix-5/7 + Rader for 5/7-smooth & prime dims (the `2^a·3` smooth case is done); `512²`-class
    large-inner strided could use FFTW's buffered/tiled transpose (currently passes at ~0.98, near gate).
  - Spec: `docs/superpowers/specs/2026-06-27-ndim-fft-design.md`; plan:
    `docs/superpowers/plans/2026-06-27-ndim-fft-complex.md`.
- **Multi-threading** — single-thread only (deliberate for the kernel investigation; a real library wants
  threads).
- **`autoplan` returns a 7-member `Union`, not a concrete type** — runtime kernel selection (the
  plan-constructor exception to "concrete returns"; one dispatch per `apply`, amortized over the transform).
  The pow2 `AutoPlan{T, typeof(best)}` wrapper widens its `T`, putting a bare `AutoPlan` in the union —
  dropping or concretely-parameterizing it makes the union fully `{Float64}`. (Found via StrictMode
  dogfooding; StrictMode now flags abstract-`eltype` containers — F34 — and its empirical `@assert_noalloc`
  `gc_num`-artifact false-fail was fixed — F33.)
- **Registration** — `0.1.0`, not registered; needs a stable API + (at least) Float32 before General.

### Beat / match rustfft where Julia has a structural edge (grounded by the `julia-sched-mwe` MWEs)
The MWEs proved there is **no compiler barrier**: identical-algorithm Julia compiles ≥ Rust. So aim to:
- **Beat** where Julia has an edge rustfft lacks: **AVX-512 (W=8)** (rustfft is AVX2-only — already beats it
  1.05–1.15× on W=8-clean non-pow2; extend coverage) and **runtime per-size codegen** (`@generated`/JIT
  specialization vs rustfft's fixed shipped codelets — exploit for sizes rustfft handles less well).
- **Parity** on rustfft's mature pow2 / radix-8 path (match the algorithm; no compiler reason it can't).
- Caveat / not a goal: "uniformly faster than rustfft" — it's a well-tuned library; the realistic target is
  *beat on the structural-edge regimes, parity elsewhere*. The MWE's "Julia ≥ matched Rust" is not "≥ the
  hand-tuned rustfft library."

### Tooling / infrastructure
- **StrictMode.jl dogfooding — DONE (test dep + feedback).** Adopted StrictMode (declarable perf
  guarantees = AllocCheck + JET + `@inferred` unified) as a test dep: `test/strictmode_tests.jl`
  (`@assert_typestable`/`@assert_noalloc` on one plan per routing path; gated on `checks_enabled()`),
  `bench/strictmode_audit.jl` (broader hot-path `check` sweep, mirrors `bench/alloccheck.jl`),
  `test/LocalPreferences.toml` ships checks enabled. StrictMode's verdicts agree with PureFFT's existing
  JET/AllocCheck on the whole hot path (26/26 checks pass). Findings + 2 direct fixes sent upstream (see
  the StrictMode clone's `FEEDBACK.md`): F2 `@assert_typestable` mislabeled unresolved names as
  instability (fixed), F4 `format_findings` had no String method (fixed); F1 enable-needs-restart, F3
  `audit` return-type varies by kwarg, F5 whole-module audit isn't scoped to declared guarantees
  (reported). Future: optionally add `@strict_function` to a couple `src/` kernels (compile-time-gated
  off) once StrictMode stabilizes; keep TrimCheck (orthogonal — StrictMode doesn't cover trim-safety).
- **Nightly CI `@testsetup`** — the nightly job fails on a ReTestItems-vs-nightly `@testsetup`
  incompatibility (currently `continue-on-error`, so it doesn't red the pipeline). Bump/patch when
  convenient.
- **FixedSizeArrays** — tried, null result on the pointer-based hot path (not pursued); revisit only if
  the scalar-indexed buffers become hot.

## Done (for context)
Pow2 radix-4 AVX-512 engine; non-pow2 codelet (Stage 9) / four-step (Stage 10) / recursive mixed-radix
(Stage 12); Rader (Stage 11) + Bluestein (Stage 8); CPU-generic cache tuning (CPUSummary); AbstractFFTs
integration; ReTestItems + perf-regression tests; the faithful RustFFT-AVX non-pow2 port + integration;
the width-generic AVX-512 (W=8) compute layer, W=8 transposes, and targeted W=8 routing; faithful
**Butterfly18** (non-pow2 2^odd·3²·5 oscillation) + **Butterfly256/512** (pow2 odd-power gap) bases;
`autoplan` **median** ranking (was min) + static-**Tuple/`map`** timing (replacing the abstract-`eltype`
`Vector` dynamic-dispatch container); **Float32 AVX parity** (genericize the 4-complex kernels over
`Vec{8,T}`: `V8f32` non-pow2 W=8 tree + `T`-generic pow2 base codelet & vectorized transpose; at/above
FFTW & RustFFT, 1.3–1.8× the Float64 throughput).
