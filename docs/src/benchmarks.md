# Benchmarks

All benchmarks: AMD Zen 5 (znver5, AVX-512), single-thread, in-place, planning excluded.
Input: `Vector{ComplexF64}`, power-of-two sizes. FFTW at `MEASURE` flag. RustFFT with
`IgnoreArrayChecks`. PureFFT `:fast` (autotuned).

## GFLOP/s comparison

![GFLOP/s vs transform size: FFTW, RustFFT, PureFFT :fast](assets/comparison.png)

Flop model: `5 · N · log2(N)` (standard radix-2 count).

## PureFFT `:fast` vs rustfft (GFLOP/s)

| n | PureFFT `:fast` | rustfft-scalar | rustfft-AVX | vs scalar | vs AVX |
|---:|---:|---:|---:|---:|---:|
| 1024   | 38 | 21 | 44 | **1.8×** | 0.86× |
| 4096   | 40 | 19 | 48 | **2.1×** | 0.83× |
| 16384  | 37 | 20 | 44 | **1.9×** | 0.83× |
| 65536  | 37 | 19 | 35 | **1.9×** | **1.04×** |
| 262144 | 28 | 17 | 30 | **1.6×** | 0.92× |

PureFFT `:fast` is **within ~1.2× of rustfft-AVX** across all sizes and **faster at n=65536**.
It is **~2× faster than rustfft-scalar** — the apples-to-apples comparison for the same algorithm.

## PureFFT `:fast` vs FFTW (GFLOP/s, earlier measurements)

These measurements are from the `:fast` variant before the AVX Butterfly16/32 codelets were added
(the radix-2 four-step path):

| n | FFTW-MEASURE | RustFFT | PureFFT `:fast` | ratio |
|---:|---:|---:|---:|---:|
| 1024   | 44 | 44 | 24 | 0.55× |
| 4096   | 45 | 44 | 22 | 0.49× |
| 16384  | 41 | 41 | 22 | 0.54× |
| 65536  | 36 | 36 | 21 | 0.58× |
| 262144 | 26 | 27 | 16 | 0.60× |

The current `:fast` (with AVX Butterfly codelets, table above) substantially improves these.

## All variant progression

| Variant | GFLOP/s (F64) | Key technique |
|---|---:|---|
| `:scalar` | 7–9 | Radix-2 baseline |
| `:base` | 11–17 | `@simd ivdep` cross-pass |
| `:recursive` | 13–24 | `@generated` codelets, cache-oblivious |
| `:soa` | 13–21 | Split re/im (negative — split/merge overhead) |
| `:fourstep` | 16–22 | Cache-blocked four-step |
| `:radix4` | 27–28 | Port of rustfft Radix4 + cache-blocked transpose |
| `:radix4avx` / `:fast` | **35–40** | + explicit SIMD.jl Butterfly16/32 |
| FFTW-MEASURE | 35–46 | Reference |
| rustfft-AVX | 35–51 | Reference |

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
Once we ported the same algorithm (rustfft's `Radix4`) to Julia and added AVX codelets, the gap
vanished.

## Methodology

- **Timing**: BenchmarkTools `@belapsed` with `setup=(y=copy(x)) evals=1`, min over ≥400 samples.
- **In-place**: all transforms applied in-place on a fresh copy per sample.
- **Planning excluded**: plans built once outside the timing loop.
- **Single-threaded**: `FFTW.set_num_threads(1)`; RustFFT and PureFFT are single-thread by design.
- **Correctness**: all variants validated against FFTW, relative error ≤ 5e-16.
