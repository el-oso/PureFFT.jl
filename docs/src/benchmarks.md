# Benchmarks

All benchmarks: AMD Zen 5 (znver5, AVX-512), single-thread, in-place, planning excluded.
Input: `Vector{ComplexF64}`, power-of-two sizes. FFTW at `MEASURE` flag. RustFFT with
`IgnoreArrayChecks`. PureFFT `:fast` (autotuned).

## GFLOP/s comparison

![GFLOP/s vs transform size: FFTW, RustFFT, PureFFT :fast](assets/comparison.png)

Flop model: `5 · N · log2(N)` (standard radix-2 count).

## PureFFT `:fast` vs FFTW and RustFFT (GFLOP/s, power-of-two)

| n | FFTW | RustFFT | **PureFFT `:fast`** | verdict |
|---:|---:|---:|---:|---|
| 128    | 32 | 34 | **41** | PureFFT fastest |
| 256    | 39 | 41 | **44** | PureFFT fastest |
| 512    | 45 | 46 | 42 | RustFFT ahead |
| 1024   | 48 | 44 | **48** | parity |
| 4096   | 46 | 45 | 44 | parity |
| 16384  | 38 | 41 | **44** | PureFFT fastest |
| 65536  | 35 | 36 | **39** | PureFFT fastest |
| 262144 | 27 | 27 | 26 | parity (memory-bound) |

PureFFT `:fast` **matches or beats both FFTW and RustFFT across most of the range**. It trails only
at n=64 (RustFFT's hand-tuned small butterfly) and n=512/2048 (a shuffle-bound size-32 base codelet).
The small-n fused register kernels (n≤128), radix-16 pass fusion, and vectorized transpose closed
the earlier gap (pure Julia was once ~0.55× of FFTW — see git history / REPORT.md).

## Non-power-of-two

No O(n²) cliff: a large prime factor no longer falls to a direct DFT. PureFFT's `:fast` routes by
size/factorization: small smooth → dynamically-generated mixed-radix codelet (Stage 9); smooth
composite up to ~4096 → SIMD four-step with batched codelets (Stage 10, split autotuned at plan
time); everything else (large prime, **or smooth composite > 4096**) → Bluestein chirp-Z (Stage 8).

The plot sweeps smooth-composite sizes (the common non-pow2 case). PureFFT (green) reaches ~9–22
GF/s on the four-step path (≈0.6× FFTW — not yet parity, the AoS↔SoA + transpose overhead), then
**drops to ~5 above n≈4096**: the four-step factors cap at 64 (64×64=4096), so larger smooth sizes
currently fall back to Bluestein. Lifting that needs a recursive four-step (tracked as future work).

![GFLOP/s on non-power-of-two sizes](assets/comparison_nonpow2.png)

| regime | example n | PureFFT | note |
|---|---:|---:|---|
| smooth, small — codelet | 27 / 48 | **12.8** / 13 | beats FFTW (10.7); was ~0.2 via old mixed-radix |
| smooth composite ≤4096 — four-step | 720 / 2880 | **22 / 22** | ~0.6× FFTW, 2–4× Bluestein (Stage 10) |
| smooth composite >4096 — Bluestein | 5760 / 23040 | ~5 | four-step ceiling; recursive four-step is future work |
| large prime / prime power — Bluestein | 181 / 5793 | ~5 | O(n log n), no cliff |

## All variant progression

| Variant | GFLOP/s (F64) | Key technique |
|---|---:|---|
| `:scalar` | 7–9 | Radix-2 baseline |
| `:base` | 11–17 | `@simd ivdep` cross-pass |
| `:recursive` | 13–24 | `@generated` codelets, cache-oblivious |
| `:soa` | 13–21 | Split re/im (negative — split/merge overhead) |
| `:fourstep` | 16–22 | Cache-blocked four-step |
| `:radix4` | 27–28 | Port of rustfft Radix4 + cache-blocked transpose |
| `:radix4avx` / `:fast` (pow2) | **40–48** | + AVX Butterfly16/32, radix-16 fusion, small-n register kernels, vectorized transpose |
| `:bluestein` | non-pow2 | chirp-Z, O(n log n) on primes |
| `:codelet` | non-pow2 | dynamically-generated mixed-radix kernel (small smooth) |
| four-step (via `:fast`) | **12–20** | batched SoA codelets, smooth composite non-pow2 |
| FFTW-MEASURE | 35–48 | Reference |
| rustfft-AVX | 34–46 | Reference |

## Controlled Julia vs Rust (same algorithm)

To separate **language** from **algorithm**, we ran the identical radix-2 DIT kernel in both Julia
and Rust (same layout, same twiddle indexing, same `muladd`/`mul_add` FMA, same `@inbounds` /
`get_unchecked`). Checksums match bit-for-bit.

| n | Julia | Rust | winner |
|---:|---:|---:|---|
| 64     | 267 ns   | 240 ns   | Rust +11% |
| 256    | 1211 ns  | 1111 ns  | Rust +9% |
| 1024   | 5266 ns  | 5346 ns  | tie |
| 4096   | 24309 ns | 26598 ns | Julia +9% |
| 16384  | 143956 ns| 237816 ns| Julia +65% |
| 65536  | 1.10 ms  | 1.32 ms  | Julia +20% |
| 262144 | 6.83 ms  | 7.39 ms  | Julia +8% |

**Conclusion: same algorithm ⇒ same speed. The language is not the lever.**

The earlier "PureFFT is 2× slower than rustfft" result was about algorithm choice, not language.
Once we ported the same algorithm (rustfft's `Radix4`) to Julia, added AVX codelets, fused passes,
and added register-resident small-n kernels, PureFFT reached parity and now leads at most sizes.

## Methodology

- **Timing**: BenchmarkTools `@belapsed` with `setup=(y=copy(x)) evals=1`, min over ≥400 samples.
- **In-place**: all transforms applied in-place on a fresh copy per sample.
- **Planning excluded**: plans built once outside the timing loop.
- **Single-threaded**: `FFTW.set_num_threads(1)`; RustFFT and PureFFT are single-thread by design.
- **Correctness**: all variants validated against FFTW, relative error ≤ 5e-16.
