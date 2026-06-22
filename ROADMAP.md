# PureFFT.jl — Roadmap

Status + planned work. This is the canonical, checked-in roadmap (human- and agent-readable). For the
"why/how" of finished work see git history, `docs/src/performance.md`, and `docs/src/benchmarks.md`.

## Where it stands today

- **Power-of-two**: AVX-512 (`Vec{8,Float64}`) radix-4 engine, 40–48 GFLOP/s — matches/beats FFTW &
  RustFFT across most of the range (`src/radix4_avx.jl`).
- **Non-power-of-two**: a faithful mechanical port of RustFFT's AVX mixed-radix (`src/avxradix/`),
  routed by `autoplan` (`src/autotune.jl`) and exposed as `AvxMixedRadixPlan`. radix-8-dominated smooth
  sizes are at FFTW/RustFFT parity; the rest fall back to the existing codelet / four-step / recursive /
  Rader / Bluestein paths. No size cliff.
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
- **Packed bases + full planner** — port `Butterfly{18,24,27,32}` (the dual-width packed path) + the
  `avx_planner` base-selection (`base_fn`) so PureFFT's decompositions match RustFFT's per size (today
  the planner uses a Butterfly36 base only). Lifts the 3-heavy laggard sizes (~0.85–0.92×).
- **radix-9/12 floor ~0.90× of rust — it's ALGORITHMIC, not a Julia compiler issue.** An MWE comparing a
  matched radix-9 butterfly *and* a full radix-9 step (butterfly+twiddle+transpose) in Julia (SIMD.jl) vs
  Rust (`core::arch`) — see the standalone `julia-sched-mwe/` reproducer — found Julia compiles to identical
  asm and runs **at least as fast** at both levels (and LLVM 18/19/21 lower the same IR identically). So the
  gap is **rustfft's implementation** being more optimized, not Julia/LLVM scheduling. To close it: diff
  PureFFT's MR9/MR12 pass against rustfft's `Butterfly9`/mixed-radix source (decomposition, in-place /
  transpose / memory strategy) and adopt what's better — a PureFFT optimization, not a compiler chase.
- **MR2 / MR16** — currently sizes needing a radix-2 or radix-16 step fall back; add them for fuller
  smooth-size coverage.

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
the width-generic AVX-512 (W=8) compute layer, W=8 transposes, and targeted W=8 routing.
