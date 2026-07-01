# Spec: dedicated small-pow2 complex monolith (n=64/128) — close the vs-RustFFT gap

**Date:** 2026-07-01  **Status:** spec only (not implemented) — measured, scoped for later

## Problem (measured, locked clock)
At the smallest pow2 sizes PureFFT trails RustFFT while still beating FFTW:
- n=64: FFTW 142 ns, **Rust 105.6 ns, PF 113.7 ns** → PF/Rust = **0.93×** (a gate miss vs Rust), PF/FFTW = 1.25×.
- PF also shows higher variability here: σ ≈ 5–7% vs FFTW/Rust ≈ 1.5%.

## Root cause (measured, NOT the AutoPlan wrapper)
- **The `AutoPlan` wrapper is free** — head-to-head, raw `Radix4Avx` vs `AutoPlan`-wrapped is 71 ns vs 71 ns, σ7% vs σ7% (0 ns overhead, 0 extra jitter). The earlier "AutoPlan dispatch" hypothesis is **debunked**. (Separately, `AutoPlan`'s non-concrete return-Union is a real *type-stability* item — see ROADMAP line ~263 — but it is not a perf/jitter cause.)
- The real cause: at n=64 PF routes to `Radix4Avx`, a **multi-pass** engine (bit-reversal transpose → base butterflies → log₄ cross-passes). The memory passes cost ~8 ns more than Rust's **single in-register `Butterfly64` monolith** and are more cache/schedule-jitter-prone at this tiny size. No existing PF path is better: `Codelet` = 310 ns (4× slower, but steady σ1.8%); W8 = 70 ns but σ14%.

## Approach
Port rustfft's `Butterfly64Avx` (and `Butterfly128Avx`) as **staged register monoliths** — 8×8 (and 8×16) with an in-register transpose, fitting the register file (NOT a flat 64-register butterfly, which would spill catastrophically — see [[ports-are-dead-code]]). PureFFT already has a buffer/stride `B64` used inside trees; this is a standalone-n=64 register form tuned as an `autoplan` pow2 candidate for n=64 (and n=128), timed additively so it's kept only where it wins (cannot regress).

## Feasibility / risk
- Medium. The staged 8×8 structure is proven (Rust does it; PF's B64 exists). The register transpose + tuning to actually beat `Radix4Avx`'s 70 ns is the uncertain part; the additive-autoplan slot means a non-win is simply not selected (safe).
- Gate: bit-exact vs reference DFT; par-or-faster vs current `Radix4Avx` at n=64/128 (locked clock, `lock` boost-off); no pow2 regression; full suite + TrimCheck.

## ROI
**Moderate.** n=64/128 are *common* real-world sizes (audio frames, small batched FFTs), and the goal is matching RustFFT everywhere — so closing a 0.93× vs-Rust gate miss at the two smallest pow2 has real value. But it's isolated kernel work (one careful codelet family) with an uncertain margin (~8 ns to recover). Recommend as a dedicated post-MT effort, or bundled with any future small-codelet pass.

## Explicitly out of scope
The variability alone is not worth chasing (7% jitter on a 70 ns transform is cosmetic); it rides along if the monolith lands (a single in-register codelet is inherently steadier).
