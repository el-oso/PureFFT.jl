# 1-D non-power-of-two performance — close the remaining gap

**Status:** planning / partially addressed. Source: an external task prompt
(`/home/el_oso/Documents/claude/nonpow2_fix_prompt.md`) to bring 1-D non-pow2 PureFFT-`:fast`
to **≥ 0.95× FFTW-MEASURE and RustFFT (geomean)**, without regressing pow2 (geomean ~1.11–1.14×) or
accuracy (rel-err ≤ ~1.4e-15).

## What the N-D session ALREADY fixed (measured znver5, PF/FFTW-MEASURE)
The N-D work's 1-D side-fixes (single-factor-of-3/5/7 routing + B16/B32/MR7 + radix-7/2ᵏ·5 — commits
`65cf783`, `3016960`) closed several of the prompt's offenders **as a byproduct**:
- **96 (2⁵·3): 0.63 → 1.18** ✅ — the prompt's worst small-mixed size; fixed by admitting single-factor-of-3
  to the fast mixed-radix path (`p3≥2` guard removed; B16 leaf).
- **768 (2⁸·3): 1.23** ✅ already fine · **2187 (3⁷): 0.96** ~at gate.
- Single factors of 5 and 7 (e.g. 160=2⁵·5, 224=2⁵·7) now route fast (radix-5/7 + 2ᵏ·5 1-D bases).

## What REMAINS (the real non-pow2 floor — measured znver5)
| N | factor | now | target | bucket |
|---|---|---|---|---|
| 99991 | prime | 0.73 | ≥0.95 | **(1) Bluestein/prime path** |
| 1000 | 2³·5³ | 0.69 | ≥0.95 | **(2) high-power-of-5** |
| 10000 | 2⁴·5⁴ | 0.65 | ≥0.95 | **(2) high-power-of-5** |
| 65520 | 2⁴·3²·5·7·13 | 0.69 | ≥0.95 | **(3) large prime factor (13) in a mixed size** |
| 6561 | 3⁸ | 0.88 | ≥0.95 | **(4) 3-heavy (radix-9/27) floor** — ROADMAP §15 |

The prompt's priority order: (1) Bluestein/prime → (2) 5-heavy → (3) small mixed (now done) — so the live
priorities are **(1) Bluestein, (2) high-power-5, (3) large-prime-factor mixed, (4) 3-heavy**.

## Work breakdown (to become execution tasks)
1. **Bluestein/prime (99991, and large prime factors generally — feeds (3)).** Profile `src/bluestein.jl`:
   is the chirp + its FFT precomputed in the plan (cold) or rebuilt per apply? Is the inner pow2 convolution
   size M minimal (≥2n−1, next pow2 — or could a smooth non-pow2 M via the now-faster mixed-radix be
   cheaper)? Are twiddles/chirp laid out for SIMD? The N-D **batched Rader** (just built, `src/ndim_batched.jl`)
   proves Rader is fast — for 1-D primes with smooth p−1, ensure `autoplan`'s Rader gate is as wide as it
   should be (the N-D work widened it to ≤7-smooth p−1; check 1-D matches). 99991−1 = 99990 = 2·3³·5·7·53 →
   has a 53 factor, so Rader's inner FFT is itself non-smooth ⇒ Bluestein is the honest path here; focus on
   making *Bluestein* fast (precompute, M-choice, layout).
2. **High-power-of-5 (1000 = 2³·5³, 10000 = 2⁴·5⁴).** The session added single-factor-of-5 routing, not
   5²/5³/5⁴. Add a **radix-25 (or radix-5 composed deeper) fast path** / admit 5-heavy sizes to the
   mixed-radix kernels (mirror the single-3 → B16 fix and the N-D batched radix-5). Check `plan_tree`/
   `autoplan` for a `5`-power guard dumping these on a slow generic/Bluestein path.
3. **Large prime factor in a mixed size (65520 = …·13).** A size with a single 13 factor: needs a radix-13
   codelet (or Rader-on-the-13-factor inside the mixed-radix), else the 13 step is slow. Check whether
   `:fast` falls back to a generic path for the 13.
4. **3-heavy (6561 = 3⁸).** The documented ROADMAP §15 radix-9/12 ~0.85–0.92× floor (algorithmic, proven not
   a compiler issue via the `julia-sched-mwe`). Lift via the standing "diff vs rustfft's Butterfly9 /
   adopt a better radix-9/27 decomposition" item — hardest, lowest priority per the prompt.

## Gates (per the prompt + project rules)
- **No pow2 regression** (pow2 geomean must stay ~1.11–1.14× — re-run `bench/run_compare.jl` after each change).
- Rel-err ≤ ~1.4e-15 (machine precision) on all sizes.
- Per-change before/after ratio tables on BOTH sweeps (the prompt's deliverable format).
- Respect CLAUDE.md / the 0.96× contract; [[enforce-parity-gate]]; [[test-scope-filtered]] (1-D planner
  changes touch shared core → full `Pkg.test()` is the pre-commit gate for those).
- Reproduce via the prompt's driver (copy `bench/compare.jl` → `:fast` only + the size list) before/after.

## Relation to existing ROADMAP items
This subsumes/extends the ROADMAP "Non-pow2 coverage / parity" items (radix-9/12 floor, MR2/MR16, Bluestein)
with concrete measured targets. The N-D session knocked out the small-mixed (96) and single-5/7 cases; this
plan tracks the high-power-5 / large-prime-factor / Bluestein / 3-heavy remainder.
