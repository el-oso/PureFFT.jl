# N-D batched non-pow2 FFT — the "big prize" for non-power-of-two dims

**Date:** 2026-06-28
**Status:** design / planning (pow2 batched kernel already validated + integrated; this extends it)
**Goal:** eliminate the transpose for **non-pow2** strided dims in the N-D FFT, the same way the pow2 batched
radix-8 kernel did for pow2 dims — by vectorizing the FFT **across the contiguous batch** (no transpose) for
non-pow2 lengths. Closes the remaining transpose-path shapes (384², 512×384, 96³, 48³, and any region with a
non-pow2 transformed dim), which currently sit at ~0.4–0.7× FFTW on the transpose path.

## 1. Context (what already exists)
- N-D c2c apply (`src/ndim.jl`): per transformed dim, dim-1 = contiguous batched 1-D; dim>1 = **pow2 →
  batched radix-8 kernel** (`src/ndim_batched.jl`, no transpose), **non-pow2 → transpose path**
  (`_apply_dim_transpose!`).
- The batched kernel principle (validated, bit-exact, zero-alloc): for a strided dim, the `inner` dimension
  is contiguous → process W consecutive batch elements as one `Vec` (one element-position across W
  transforms); FFT butterflies run SIMD across the W lanes; twiddles are **scalar-broadcast** (every lane is
  at the same butterfly position); NO transpose. Per-dim descriptors are concrete types
  (`Dim1Plan`/`BatchedDim`/`TransposeDim`) dispatched by separate `_apply_dim!` methods; the `@generated`
  apply indexes the heterogeneous tuple with literal indices.

## 2. The principle carries over unchanged
A batched FFT across the batch is **independent of the transform length's factorization** — only the
*butterfly stages* differ. The pow2 kernel uses radix-8/4/2 stages on the row-vectors. Non-pow2 just needs
non-pow2 stages (radix-3/5/7) and/or Bluestein. Everything else (W-wide complex `Vec` across the contiguous
batch, scalar twiddles, cache-blocked inner-chunk loop, scalar `inner%W` tail) is **reused as-is**.

## 3. Two regimes (mirror the 1-D `autoplan`)
### 3a. Smooth non-pow2 → batched mixed-radix
For `n_d` that is 2·3·5·7-smooth (covers the current misses: 384 = 2⁷·3, 96 = 2⁵·3, 48 = 2⁴·3): compose the
existing batched radix-8/4/2 stages with **new batched radix-3 / radix-5 / radix-7 stages**. A batched
radix-`r` butterfly = the size-`r` DFT applied to `r` row-vectors (each W complex across the batch), with
scalar-broadcast twiddles — the *same* algebra as the codebase's `avx_column_butterfly3/5` but re-expressed
on `r` separate batch-row-vectors instead of packed-within-a-vector. Mixed-radix decomposition per `n_d`
(reuse the 1-D factorizer's logic, `_recursive_factors`-style).

**Bonus — likely reuse, not build-from-scratch:** PureFFT *already* has **batched SoA codelets** in
`FourStepCodeletPlan` (the four-step's inner pass does batched mixed-radix FFTs across a batch — `guide.md:65`,
`autotune.jl`). That is almost exactly the primitive needed here (a batched FFT across a batch dimension).
**First investigate reusing/adapting that batched SoA codelet** for the N-D contiguous-batch layout before
hand-writing new radix-3/5/7 stages — it may already cover smooth `n_d`. Build new batched stages only for
what it doesn't cover.

### 3b. Smooth-p-1 primes → batched Rader (the prime *parity* path)
**Bluestein is NOT a parity solution — it is the project's known non-par floor (~3–5 GFLOP/s).** The 1-D
path uses it only as a last resort and built codelets/four-step/Rader precisely to avoid it
(`autotune.jl:40-48`, `benchmarks.md:65`). So the prime *parity* path is **batched Rader** (for primes with
smooth `p−1 = 2ᵃ·3ᵇ`, `p ≥ RADER_MIN_P`): Rader turns the length-`p` DFT into a length-`(p−1)` cyclic
**convolution**, which runs on a fast inner FFT (here a *batched* pow2/smooth FFT — already built/being
built) plus elementwise pre/post (batchable). Mirror `autoplan`'s Rader gate (`largest-prime(p−1) ≤ 3`).

### 3c. Hard primes → batched Bluestein (the known floor, NOT a gate-clearer)
For primes where Rader loses (a 5/7 factor in `p−1`) or genuinely large primes, fall back to **batched
Bluestein**: chirp pre/post + pointwise are elementwise (batchable), the two length-`M` FFTs reuse the
**batched pow2 kernel**. This is worth building only because it still beats *transpose + Bluestein* (no
transpose) — but it **inherits the 1-D Bluestein sub-gate floor**; a prime dim routed here will NOT clear
0.96× vs FFTW, the same known limitation as 1-D. Flag such shapes OPEN honestly; do not pretend Bluestein
clears the gate. Lowest priority (prime dims are rare in real N-D shapes — images/volumes are pow2 or smooth).

## 4. Routing (per non-pow2 transformed dim, d>1)
Mirror `autoplan`'s non-pow2 decision: `n_d` 2·3·5·7-smooth and ≤ a size cap → batched mixed-radix (3a);
else (prime / large prime factor) → batched Bluestein (3b). dim-1 and pow2 dims unchanged. Add a
`BatchedSmoothDim` / `BatchedBluesteinDim` concrete descriptor (parallel to `BatchedDim`) so `_apply_dim!`
dispatches by type and the `@generated` apply stays literal-indexed + dispatch-free.

## 5. Phasing (bit-exact before perf, each layer gated vs FFTW)
1. **Batched radix-3 stage** + mixed-radix composition for `2^a·3` sizes — covers the immediate misses
   (384², 96³, 48³). Prototype → bit-exact vs FFTW for the strided dim → measure batched-vs-transpose →
   integrate + route → full-ND PF/FFTW. (Highest value: these are the current non-pow2 misses.)
2. **Batched radix-5 / radix-7** stages — full 2·3·5·7-smooth coverage.
3. **Batched Bluestein** (chirp/pointwise batched + reuse batched pow2) — prime / large-factor dims.
4. **dim-1 for non-pow2** if the dim-1 grind's approach generalizes (separate track).

## 6. Gates (every layer)
- **Bit-exact vs FFTW** for the new strided non-pow2 dim, all ranks/regions, F64+F32, incl. the `inner%W`
  tail and inverse. The existing `test/ndim_tests.jl` non-pow2 cases must stay green (they currently exercise
  the transpose path; after routing they exercise the batched path — both must be bit-exact).
- **Zero-alloc + dispatch-free + trim-safe** (the Task-5 gate testitem) for a batched-non-pow2-routed plan.
- **0.96× vs FFTW** per shape — the contract. Not "done" until the non-pow2 shapes clear it; grind the next
  lever (radix coverage, cache-blocking, Bluestein M-choice) until they do.

## 7. Risks
- Batched radix-3/5/7 butterfly algebra correctness (the row-vector re-expression) — gate bit-exact per
  stage before composing.
- Bluestein M-choice + chirp precompute per dim (cold path) — must stay zero-alloc on the hot path.
- Mixed-radix stage ordering / digit-reversal across the batch (the pow2 kernel already solved a
  digit-reversal-vs-stage-order subtlety — carry that lesson).
- Descriptor proliferation (more per-dim concrete types) — keep `@test_opt` green.
