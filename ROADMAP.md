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
- **5-smooth coverage** — the main comparison plots use 5-smooth sizes, which W=8 doesn't cover yet
  (needs `transpose5`/`transpose9` derived at W=8, like `transpose4/8/12` already are in
  `src/avxradix/avxport.jl`). Doing this lets the *main* plots show W=8, not just the 2·3-smooth demo
  (`docs/src/assets/avx512_nonpow2.png`).
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
- **radix-9/12 codegen floor** — radix-9/12 levels sit ~0.90× of rust (intrinsically ~3× more
  shuffle/FMA-heavy than radix-8). Diffuse Julia-LLVM-vs-rustc scheduling; only worth chasing with the
  measurement floor pinned (see `performance.md` §15: ±7% run-to-run on the ratio — pin CPU frequency).
- **MR2 / MR16** — currently sizes needing a radix-2 or radix-16 step fall back; add them for fuller
  smooth-size coverage.

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
