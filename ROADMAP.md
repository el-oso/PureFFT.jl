# PureFFT.jl — Roadmap

Status + planned work. This is the canonical, checked-in roadmap (human- and agent-readable). For the
"why/how" of finished work see git history, `docs/src/performance.md`, and `docs/src/benchmarks.md`.

## Vision / positioning

PureFFT started as an investigation (*can pure Julia match FFTW/rustfft?* — **yes**). The goal now:
**mature it into a general-purpose FFT library that also showcases Julia.** This is the filter for every
roadmap decision below — *does this serve general-purpose use, the Julia showcase, or (ideally) both?*

> A **general-purpose, pure-Julia, MIT-licensed, dependency-free** FFT that **matches FFTW and RustFFT**
> while doing what an ahead-of-time C/Rust library structurally can't: **on-demand codelet
> specialization, type-generic kernels, `AbstractFFTs`-native composability.**

- **Practical differentiator (true today):** pure Julia, no binary dependency, **MIT** vs FFTW's **GPL**
  (`FFTW.jl` wraps a GPL binary) — a real reason to choose PureFFT regardless of the showcase.
- **Showcase ≈ library:** the codelet generator, type-generic kernels, and the AbstractFFTs-native API
  are *both* Julia flexes *and* general-purpose capabilities — mostly one investment, not two.
- **The one separable piece is multi-threading** — required for "general-purpose," but a byproduct of no
  showcase feature, so it's a deliberate scheduled milestone (below), after the generator prototype.
- **Honest cost:** general-purpose is a forever commitment (the full input space becomes a permanent
  test/maintenance surface) and a multi-year arc; the strict parity-gate culture is what makes it tractable.

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

### ⭐ Codelet generator — FIRST DELIVERY SHIPPED (2026-06-30); genfft-*optimiser* dead vs LLVM

The flagship "Julia-native genfft analogue" was investigated and partly delivered. Two clear results:

**1. The genfft *optimising-compiler* framing is DEAD vs modern LLVM (measured, decisive).** A make-or-break
gate built a minimal DFT IR + CSE pass and compared *compiled* `@code_native`: naive 1008 insns vs IR+CSE
**1024** (worse) — LLVM already does genfft's CSE. A scheduler gate (register-pressure list scheduling)
likewise didn't beat LLVM's allocator (the DFT DAG's antichain makes peak liveness irreducible). So
CSE / sign-prop / network-transposition / scheduling are **redundant against LLVM 18+** (consistent with
`julia-sched-mwe`). *Do not rebuild them.* (Evidence in git history: the CSE gate commit + the scheduler
gate commit on the merged `feat/codelet-generator` arc; the experiment files were pruned after the verdict.)

**2. Generating the proven-fast COLUMN-PACKED structure WON — and ships.** The pivot (after wrongly
benchmarking the slow SoA four-step — the reinterpretation plateau) was to generate PureFFT's *column-packed*
`avxradix` structure, not a generic one. Validated at parity: `src/gen/transpose.jl` reproduces hand
`avx_transpose{5,7,9}_packed` at **Δ=0** instructions; `src/gen/colgen.jl` (`gen_pp_codelet!`) reproduces hand
B25/B49 at **identical** arith+shuffle counts. **In production now** (additive `autoplan` candidates,
invariant-hole fixed — Bluestein is timed when a generated candidate competes, so "cannot-regress" truly
holds): **7 prime-squares P²** (121…961, 1.76–5.49× FFTW) + **10 composites M·P²** (M∈{2,4}, P∈{17..31},
1.43–3.27×), all beating Bluestein. Honestly dropped as measured non-wins: all P³ cubes, 2^a·11²/13².
PrecompileTools `@compile_workload` (`genpp_precompile_max_p` Preference, default 31, ≤~60 s first-use → 0).

**Open continuation (the real remaining value — *not* the optimiser):**
- **Systematise the hand-written `avxradix` SIMD functions** into the generator — **largely DONE.** Phase 0
  (prime cluster: cb5/7/13, transposes, MR5/7/13→MRPrime, B25/B49→gen_pp, W=8 variants; master a23d275) +
  Phase 1 (`src/gen/composite.jl` `avx_colbf_composite`, a strictly-2-factor `@generated` with a quadrant
  twiddle classifier: **cb8/cb9 replaced by forwarders**, par ALL PAR vs hand; master 3638cf7). Generator
  **proven for every composite shape** (cb16/24/32 register forms, ref-DFT). cb12 stays hand (Good-Thomas).
- **The register composite ports B24/B27/B32 are DEAD CODE — do NOT wire (measured 2026-07-01).** A
  monolithic R-point butterfly needs R live registers (24>16 ymm) → spill-bound (~100× slower than the
  staged `MR` route: B24=5970ns vs `MR3{8,B8}`=60ns, 66 spill moves), AND every port size already has a
  fast native route (24→MR3{8,B8}, 27→MR3{9,B9}, 32→Radix4Avx). They'd lose autoplan's timed competition.
  This is *why* the design stages into small MR passes + memory transposes. (Supersedes the old
  "port Butterfly{24,27,32}" / "add MR16" items in the Non-pow2 section below.)
- **More winning size classes** (measure-then-wire, like the composites): other prime-power composites.
- *Not* a fix for the 2^a·5³ architectural floor (that's the column-packed structure's own ceiling).


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
- **1-D non-pow2 vs FFTW — geomean 0.84 → ~1.1 (DONE; was a cluster of routing bugs, not floors).**
  An external benchmark prompt (`docs/superpowers/plans/2026-06-28-nonpow2-1d-perf.md`) flagged a ~16% gap;
  the slow sizes were each falling to the slow `RecursiveMixedRadixPlan` because the fast `AvxMixedRadixPlan`
  `plan_tree` returned `nothing`. Fixed: **single-factor-3** (96: 0.63→1.3), **2ᵃ·5ᵐ** (1000/10000: 0.7→1.0,
  B8 leaf + radix-5 chain), **lone large-prime-13** (65520: 0.69→1.13, new SIMD `avx_column_butterfly13`+MR13
  + a smooth-tree-carries-7/13 refactor), **Bluestein smooth-M** (99991: 0.68→0.98 — convolution size is now
  the smallest 2·3·5-smooth ≥2n−1, not nextpow2), and **pure-3ⁿ radix-9** (6561: 0.84→**1.40**) — the latter
  via a **partial-`V2f` odd-column tail** on MR9/MR3 (the SIMD radix-9 stage needs even column counts; pure-3ⁿ
  is always odd and was dropping the last column → it was routed to recursive). All bit-exact (≤1.6e-16), no
  pow2 regression. Remaining sub-gate: the genuine **radix-5/9 shuffle floor vs *rustfft*** (still ~0.90 on a
  few high-5-power sizes) and **radix-11** sizes (no codelet — Bluestein, niche).
- **More packed bases** — `Butterfly18` **DONE** (B18 = 2·3², closed 2^odd·3²·5); pow2 `Butterfly256/512`
  **DONE** (closed the odd-power gap). `Butterfly{24,27,32}` **WON'T DO — measured dead code** (register
  composite butterfly R≥16 spills, ~100× slower; sizes already fast via MR trees — see the codelet-generator
  section above). The remaining 3-heavy laggard sizes (~0.85–0.92× vs rust) are the radix-9/12 *algorithmic*
  gap below (diff rustfft's pass), not a missing base.
- **radix-9/12 vs rust — CLOSED (measured 2026-07-01, LOCKED clock ~2 GHz, boost off — clean).** A targeted
  probe (`bench/measure_radix912.jl` → `bench/results/radix912.json`) found **every pure radix-9/12 size at
  or above rust**: 81=1.88×, 729=1.51×, 1728=1.03×, 5184(**MR12**)=1.03×, 6561=1.35×, 20736=1.07×, 15552=
  1.13×, 3888=1.09× (PF÷Rust). The old "~0.90× floor" was stale — closed by the intervening non-pow2 work
  (B18/B36/W8/coverage-gap fixes); no MR9/MR12 rust-diff needed. **The genuine remaining sub-rust sizes are
  radix-5/9 at large sizes, not core radix-9/12:** (a) the **2^a·5³+ architectural floor** (2000=0.95×,
  50000=2⁴·5⁵=**0.81×**) — the batch-vectorized-codelet item above, not a tweak; and (b) a **W8 radix-9
  shuffle floor on large 3-smooth sizes** (110592=2¹²·3³=**0.83×**, 46080=**0.89×**, 55296=0.95×; but
  23040=1.23×, 20736=1.07× fine) — the `vpermt2pd` redesign below. NOTE: the first runs this session used
  `pin 4500`, which does NOT hold on amd-pstate-epp (boost overrides the clamp, clock drifts 1.4–4.5 GHz;
  `pin_check`'s min==max test false-positives) — a drifting-clock ratio misled both ways (110592 read 0.85×
  then 1.026× then the locked truth 0.83×). **Use `lock` (boost off) for parity ratios, not `pin`.** The
  historical MWE (`julia-sched-mwe/`) still stands: Julia/LLVM is not the bottleneck; these two are
  architectural/shuffle floors, both already tracked as hard items.
- **MR16** — deferred (same additive-slot dead-code risk as the ports: 16-smooth sizes already route to
  radix-4/8 trees, so a radix-16 pass would ride the timed slot mostly unused). Register cb16 is proven
  generatable (ref-DFT) but unwired; only build MR16 if a specific 16-smooth size shows a measured gap.
  (**MR2 DONE** — the F64 radix-2 pass was added with the radix-5 base work below.)
- **radix-5/7 packed bases (B25/B49) — DONE.** `B25` (5²) + `B49` (7²) `@generated` register codelets now
  ROOT the radix-5/7 trees (25→B25, 49→B49, 125=MR5(B25), 625=MR5²(B25), 343=MR7(B49)), plus an F64 `MR2`
  radix-2 pass fixing the 2·5³/4·5³ routing gap (250/500 were falling to slow recursive, 0.54/0.59). Closed
  the radix-5/7 **pure-power floor**: 25=2.35, 49=2.07, 125=1.52, 625=1.41, 1000=1.20× FFTW (pinned, all green).
- **The 2^a·5³ floor (250≈0.93× / 500≈0.81× / 2000≈0.83× FFTW) is ARCHITECTURAL — verified, not a tweak.**
  Four codelet levers built+measured+disproven: the odd-M radix-5 padding trick (tail = 3% of cost), a
  monolithic B125 (LLVM spills), B50 on AVX2 (spills harder), and B50 on **AVX-512** (went register-resident
  but the ratio didn't move → **shuffle/permute-bound, not spill-bound**). Root cause (FFTW.flops + plan dump
  + `@code_native`): **FFTW is not monolithic** — it's Cooley-Tukey with small SIMD codelets (`n1fv_10/25`,
  `dft-vrank`) that **vectorize across the sub-transform batch** (transpose-free). PureFFT **column-packs**
  (W=2) ⇒ intra-transform transpose shuffles (radix-5 butterfly shuf/arith=0.38). Same flop class
  (near-parity rules out a big op-count gap), different SIMD axis. vs RustFFT, 250/500 already pass.
- **Batch-vectorized codelets for small non-pow2 (the open lever for the 2^a·5³ floor) — research/rearchitecture.**
  The one verified way to close the above: mirror FFTW's strategy — vectorize a `vrank`-style **batch of
  sub-transforms** (SIMD lanes = different transforms, transpose-free) instead of column-packing within one
  transform. Large effort and RISKY: column-packing wins on the *many* sizes where it's used, so this is a
  deliberate separate project (brainstorm/scope first), not a grind for three sizes. Would also help other
  shuffle-bound non-pow2 (radix-9/12 vs rust).
- **Non-pow2 fast-path coverage gaps (from the slow-backend audit) — DONE.** `plan_tree` rejected some
  smooth sizes, so `autoplan` fell to the best slow option (0.2–0.6× — "fell to slow generic," NOT an
  architectural floor). All closed (pinned PF/FFTW before→after): **2^a·7²** 98 0.33→1.40 (MR2(B49)),
  196 0.39→1.18, 294 0.57→1.19, 588 →1.10 (relaxed the `p7≤1` gate + an odd-safe even-izer + `_carry_even`
  helper, reusing the B49 7² codelet); **2^a·13** 26 0.28→1.45, 52 0.40→1.11, 78 0.21→1.19 (a new trivial
  **B2** leaf so the 13 rides the fast `avx_column_butterfly13` via MR13(B2) — the prescribed BP13+MR2
  route measured below gate, so per "floors are often bugs" a faster mechanism was substituted);
  **13²** 169 0.30→1.11, 338 →1.07 (new radix-13 odd-M tail `_colbf13_oddtail!`, analogous to radix-5/7,
  `isodd(M)`-guarded so even-M MR13 stays byte-identical — 65520 unchanged at 1.09). Bit-exact, 0 test fail,
  no regression. PF now beats FFTW *and* RustFFT on every former gap size.
- **Slow-backend audit verdict — `RecursiveMixedRadixPlan` STAYS; no guard test.** `autoplan` is a *timed
  competition* (it times `codelet` / four-step-or-recursive / Avx / W8 and keeps the `argmin`), so Recursive
  is selected only when it genuinely times fastest — never silently misrouted. Dropping it would force a
  slower fallback; the timing already *is* the slow-path guard (the perf gate surfaces any slow winner). The
  only real issue is coverage gaps (above), fixed by adding kernels.
- **Slow first-call planning — ESTIMATE fast-plan path SHIPPED (2026-07-01, opt-in).** The ~2–4 s cost is
  *first-call JIT compilation* (measured: autoplan(4620) 2449 ms first / 9 ms warm), because `autoplan` is
  always-MEASURE — it constructs + times ALL ~7 candidate plan types per size, compiling each. `ESTIMATE`
  (`src/estimate.jl` `_estimate_plan`, mirrors FFTW ESTIMATE/MEASURE) picks ONE plan structurally by size-class
  (pow2→Radix4Avx, large-prime→Rader, prime-square→GenPP, smooth→AvxMixedRadix, else→fall back to MEASURE),
  compiling ~1 tree → fast planning. Opt-in via `plan_pfft(x; flags=ESTIMATE)` / `plan_fft(x; flags=PureFFT.
  ESTIMATE)`; **MEASURE stays the default** (parity-safe). Spec/plan in `docs/superpowers/`. FOLLOWUPS (task):
  thread `flags` through `plan_inv` (inverse of an ESTIMATE plan currently re-times MEASURE); and the bigger
  future step — a **flop-cost model** to pick optimally without timing → then flip the default to ESTIMATE to
  fully match FFTW (deferred; the W8 measurements this session show how easily such predictions miss). A
  process-wide plan cache is NOT the lever (the 2nd call to a size is already ~10 ms — the wart is first compile).

### Breadth / type coverage (gaps for a *general* library vs the 1-D complex-`Float64` investigation)
- **DCT/DST (real-to-real) — all 8 r2r kinds DONE.** DCT-I/II/III/IV (`REDFT00/10/01/11`) and
  DST-I/II/III/IV (`RODFT00/10/01/11`), bit-exact vs `FFTW.r2r` for F64 and F32, any N.
  DCT-II (`REDFT10`) + DCT-III (`REDFT01`) even-N: Makhoul real-FFT reduction (zero-alloc,
  dispatch-free); odd N + remaining 6 kinds: extension or complex-FFT reductions (correctness path).
  API: `plan_r2r`, `r2r`, `mul!`, `dct`/`idct` (orthonormal, FFTW.jl drop-in), `plan_r2r \ x` (inverse).
  Bench harness (all 8 kinds, small + mid N): `bench/run_compare_r2r.jl` → `bench/results/compare_r2r.json`
  → `bench/plot_compare_r2r.jl`. Bench: mid-N PF/FFTW **1.4–3× (F64+F32)**.
  **Small-N `@generated` codelets (DONE).** The slow small-N kinds (DCT/DST II/III/I) now route to a
  fully-unrolled `@generated` r2r codelet (`src/r2r.jl`): input reorder + straight-line **half-size**
  real-packed DFT (reuses `_gen_dft_soa_mixed!`) + baked pre/post twiddles — branch/loop/dispatch-free,
  zero-alloc. Forward kinds (II/DST-II, I/DST-I) half-size real pack for n ≤ 64; inverse (III/DST-III)
  full-complex for n ≤ 32. Small-N PF/FFTW lifted from ~0.2–0.9× (wrap) to ~0.9–1.5× (codelet vs wrap
  at the same size = 1.1–4.9×). FFTW's hardcoded n=8 codelets still edge a couple of kinds (DCT-II
  n=8 ≈ 0.75×) — honest partial. See `docs/superpowers/specs/2026-06-27-dct-dst-r2r-design.md`.
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
- **N-dimensional (2-D/3-D) FFT — DONE (complex + real); 23/24 benchmarked complex shapes ≥ 0.96× FFTW**
  (branch `feat/ndim-fft`, ~32 commits, full suite green). Full FFTW generality (any rank, any `region`),
  drop-in via AbstractFFTs + prefixed `pfft`/`prfft`. A pure **"beat rustfft"** win — rustfft has no N-D.
  Separable on the ≥FFTW 1-D kernels, but fast via:
  1. a **batched-strided kernel** (`src/ndim_batched.jl`) — each strided dim FFTs by vectorizing *across the
     contiguous batch*, **no transpose** (the transpose-per-dim path sat at ~0.25×; the big lever);
  2. **`BatchedDim1`** gather-pack — fills the SIMD width on the contiguous dim for small shapes;
  3. **F32 512-bit batch widening** (`Vec{16,Float32}`);
  4. **batched radix-3/5/7** + a **1-D planner fix** (single-factor-of-3/5/7 sizes were on a slow generic
     path; admitting them — plus new **B16/B32 leaves, `MR7` radix-7, `avx_transpose7`/`_rbf15`/`_rbf16`** —
     routes them to the fast kernels, *beating* FFTW; this also lifted **1-D non-pow2**: n=96 0.65→1.28,
     112/160/224/240 → 1.19–1.46);
  5. **dedicated fused codelets** for the small compute-bound squares: length-128 (8×16, F32) and length-240
     (16×15, both precisions).
  Hot path dispatch-free + zero-alloc + trim-safe (`@generated`-over-rank apply, concrete per-dim descriptors).
  Bit-exact vs FFTW.
  - **Real N-D (`rfft`/`irfft`/`brfft`)** — done (`src/ndim_real.jl`): r2c along `first(region)` + c2c rest on
    the proven complex engine, batched r2c (forward zero-alloc). F32 pow2 + F64 512² clear the gate; F64
    small/non-pow2 rfft floor on the narrow-F64 real-codelet (below).
  - **OPEN — the one genuine floor: F64 `128×128`** (0.78–0.81): a 256 KB L2-resident *compute-bound* square
    where FFTW's hand-tuned `n1fv_128` wins at narrow (4-wide) F64 AVX. **Five** measured structural NO-GOs
    (6m/6t/6u/6x/7b: amortized dim-1, fused/transpose sandwich, radix-16, dedicated 8×16 codelet, radix4avx
    transpose). The F32 version of the 8×16 codelet *does* clear F32 128² (pays off at AVX-512 width). Same
    ceiling caps F64 rfft small shapes (256²/64³ ≈ 0.78–0.84) — confirmed by a no-gain recombine opt (7e).
    Beating it needs a hand-written FFTW-class length-128 F64 codelet — niche.
  - **At-gate within noise:** F64 64³ (≈0.93–0.99) and 512² (≈0.96–1.01) — bandwidth-bound; a register
    codelet is infeasible at n=512 (sub-DFTs exceed the zmm file).
  - **Batched Rader for prime dims — DONE** (branch `feat/ndim-rader`): a `BatchedRaderDim` runs a prime
    strided dim p (smooth p−1) as a batched length-(p−1) cyclic convolution on the batched kernel, no
    transpose. All prime shapes clear (127² 1.9×, 251² 1.7×, 113³ 1.95/3.3×, 256×127 2.3×) and beat the old
    transpose path 1.2–1.46×. So **non-pow2 N-D coverage is complete**: pow2 + 2·3·5·7-smooth + Rader-primes
    all batched. **27/28 benchmarked shapes ≥ 0.96×** (only F64 128² floors).
  - **Remaining (not blocking):** a fused real N-D pass for F64 rfft small shapes (narrow-F64 real-codelet
    floor, same class as F64 128²).
  - Spec: `docs/superpowers/specs/2026-06-27-ndim-fft-design.md`; bench: `bench/run_compare_ndim.jl` +
    `bench/run_compare_rndim.jl`.
- **Multi-threading — a REQUIRED general-purpose milestone, scheduled after the generator prototype.**
  Single-thread only today (deliberate for the kernel investigation). MT is the one general-purpose
  requirement that is *not* a showcase byproduct (FFTW/Rust already have it; it's relatively mechanical —
  thread the outer/batch loops, handle plan reuse / false sharing / thread safety). It is on the critical
  path to "general-purpose," but sequenced after the ⭐ flagship codelet generator (which is foundational
  and the Julia differentiator) — or pulled forward if a concrete large-transform workload demands it.
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
