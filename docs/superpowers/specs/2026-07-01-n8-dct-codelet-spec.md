# Spec: hand-tuned n=8 DCT-II/III codelet — close the vs-FFTW gate miss

**Date:** 2026-07-01  **Status:** spec only (not implemented) — measured, scoped for later

## Problem (measured, locked clock)
At n=8, two r2r kinds fall below the 0.96× gate vs FFTW (PureFFT beats FFTW almost everywhere else in r2r):
- **DCT-II n=8: PF 4.26 vs FFTW 5.72 GF/s → 0.745×** (route `codelet`).
- **DCT-III n=8: PF 5.31 vs FFTW 5.94 GF/s → 0.894×** (route `codelet`).
Every other r2r size PF wins 1.1–2×; these two tiny sizes are the exception.

## Root cause
FFTW ships **hardcoded, hand-tuned n=8 DCT codelets** (assembly-grade, like its n=8 complex codelets). PureFFT's small-N `@generated` r2r codelet (`R2RCodeletPlan`, `src/r2r.jl`) carries input-reorder + real-pack + baked-twiddle overhead that FFTW's fused n=8 avoids. This is the ROADMAP's documented "honest partial" ("FFTW's hardcoded n=8 codelets still edge a couple of kinds"). The alternative PF route (the FFT-`wrap`) is *slower* still at n=8 (the wrap wins only at n≥~48/128 depending on kind), so re-routing does not help — a genuinely better n=8 codelet is required.

## Approach
Hand-write a fused straight-line n=8 DCT-II and DCT-III codelet (in the spirit of FFTW's), avoiding the generic reorder/pack path: fold the n=8 DCT symmetry directly into a fully-unrolled 8-point real kernel with pre-baked pre/post rotations, dispatch-free and zero-alloc. Gate it into `R2RCodeletPlan` for exactly `{DCT-II, DCT-III} × n=8` (leave the general codelet for other kinds/sizes).

## Feasibility / risk
- Medium–hard. Matching FFTW's hand-tuned n=8 is precisely the kind of micro-kernel it excels at; recovering 34%/11% is uncertain. Bit-exactness is easy (vs `FFTW.r2r`); *beating* FFTW's throughput at n=8 is the risk.

## ROI
**Low.** A single tiny size (n=8) per kind, niche real-world use. It *is* a strict-gate miss (0.745×), so it matters for a "≥0.96× everywhere" claim, but the effort (a hand-tuned micro-kernel with uncertain margin) is disproportionate to the impact. Recommend deferring to a dedicated r2r-polish pass (e.g. before registration), not before MT.

## Explicitly out of scope
DCT-IV/DST-IV (already handled — codelet wins ≤n=12, documented); DCT-I (a separate story — its ratio "peaks" are FFTW cratering on prime-(N−1) sizes, not a PF weakness).
