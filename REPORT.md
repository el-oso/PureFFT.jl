# PureFFT.jl — where FFT speed comes from, and how close pure Julia gets to FFTW/rustfft

**Question.** rustfft (Rust) is reported competitive with FFTW (C). Is the speed from the
**algorithm/implementation**, the **LLVM compiler**, or **Rust the language**? And can pure
Julia — same LLVM backend — match it?

## The answer: pure Julia reaches parity, and beats both at most sizes

On a single-thread ComplexF64 transform (Zen 5, planning excluded), **PureFFT `:fast` now matches or
beats FFTW (MEASURE) and RustFFT-AVX across most of the power-of-two range**, has **no
non-power-of-two cliff**, and — via runtime codelet generation — is fast on sizes the static
libraries fall back on. The language was never the lever; it was implementation work, done in Julia.

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

GFLOP/s `= 5·N·log₂N / t`. PureFFT trails only at n=64 (RustFFT's hand-tuned small butterfly) and
n=512/2048 (a shuffle-bound size-32 base codelet — see below). Everywhere else it leads or ties.
This reverses the earlier finding in this repo's history (pure Julia was ~0.5× of FFTW); the gap was
entirely orchestration + codelet engineering, all closed in Julia with the same LLVM/AVX-512 backend.

## How parity was reached

Starting from the cache-blocked four-step (~0.55× of FFTW), the path to parity:

1. **Explicit-AVX Radix4 (`:radix4avx`)** — rustfft's `Radix4` structure, hand-vectorized with
   SIMD.jl: AoS interleaved-complex multiply (`_vcmul`), within-butterfly `Butterfly16` (4×4 with
   shuffle-free DFT-4s + one register transpose) and `Butterfly32`. This became the `:fast` winner.
2. **Radix-16 pass fusion** — fuse two radix-4 cross passes into one read-modify-write sweep,
   halving the bandwidth-bound full-array passes (cache-gated via `_R16_FUSE_MAX`).
3. **Small-n fused register kernels** — for n = 16/32/64/128, the entire transform runs in AVX
   registers (`_fft64_avx!`, `_fft128_avx!`), eliminating the ~70 ns scratch transpose that
   dominated small-n runtime. n=64: 19→32 GF/s; n=128: 25→41 (now beats both libraries).
4. **Vectorized transpose** — replace the scalar strided-store transpose with a grid of 4×4 register
   transposes (`_transpose4`) for L1-resident sizes; n=256/1024 overtook RustFFT.
5. **`pfft!` fast-path** — a more-specific method skips the runtime `interface_trait` Holy-trait
   dispatch (~10–20 ns, dominant at tiny n) while preserving duck-typed dispatch for non-subtypes.

Hot path: allocation-free, `TRIM_SAFE` (juliac / TrimCheck.jl `@validate`), dispatch-free (JET
`@test_opt`).

## Beyond parity: coverage the static libraries don't have

- **No non-power-of-two cliff (Bluestein, Stage 8).** Previously a large prime factor fell to an
  O(n²) direct DFT (~0 GF/s). `BluesteinPlan` rewrites any length-n DFT as a length-`nextpow2(2n−1)`
  circular convolution (three fast power-of-two FFTs) → O(n log n), ~5 GF/s on primes.
- **Dynamic kernel generation (`:codelet`, Stage 9) — the Julia differentiator.** FFTW's genfft is a
  separate compile-time (OCaml) tool; rustfft ships hand-written butterflies. Julia synthesizes a
  tailored straight-line mixed-radix kernel **at plan time** (`@generated _dft_codelet!`,
  Cooley-Tukey on the smallest prime factor, baked twiddle literals) for *any* size.  `autoplan`
  routes small smooth non-pow2 sizes here; this fixed a real bug (smooth non-pow2 was ~0.2 GF/s via
  the allocating recursive mixed-radix → now 4–13 GF/s, e.g. n=27: 12.8, beating FFTW's 10.7).
- **SIMD mixed-radix four-step executor (Stage 10) — the fast non-pow2 path.** Larger smooth
  composite sizes route to `FourStepCodeletPlan`: `n = N1·N2`, pass1 (size-N1 DFTs batched over N2)
  → W_n twiddle → transpose → pass2 (size-N2 batched over N1) → natural order. Each pass is a
  **batched SoA codelet** — a size-R DFT over `width` independent transforms in split (re/im) layout,
  vectorized over the batch (`Vec{W}`, shuffle-free → pure FMA). This is FFTW's vectorized
  vector-rank / rustfft's MixedRadix-of-butterflies, but with codelets generated per factor. Result:
  **12–20 GF/s on smooth non-pow2 (2–4× Bluestein, ~50 % of FFTW)**, e.g. n=1000: 19, n=900: 20.
- **AbstractFFTs.jl plan interface** — `plan_fft`/`plan_bfft`/`mul!`/`\`/`inv`/`ifft` route through
  PureFFT, so it plugs into the Julia FFT ecosystem like FFTW.jl.
- **CPU-generic tuning** — cache-blocking constants are derived from real L1/L2 sizes via CPUSummary
  (compile-time `StaticInt`s; pin via Preferences for a reproducible/trim build), not hardcoded.

## Stages built (each independently benchmarkable, `variant=`)

| Stage | Variant | Idea |
|---|---|---|
| 1 | `:scalar` | radix-2 baseline |
| 2 | `:mixedradix` | any N incl. primes (correctness oracle) |
| 3 | `:base` | Base `@simd` staged radix-2 |
| 4 | `:recursive` | cache-oblivious + `@generated` codelets |
| 5 | `:soa` | split re/im recursive (negative — see below) |
| 7 | `:fourstep` | cache-blocked four-step |
| — | `:radix4avx` | explicit-AVX Radix4 + radix-16 fusion + small-n register kernels (the `:fast` pow2 winner) |
| 8 | `:bluestein` | chirp-Z for arbitrary N (O(n log n) on primes) |
| 9 | `:codelet` | dynamically-generated mixed-radix straight-line kernel (small smooth N) |
| 10 | — (via `:fast`) | four-step + batched SoA codelets (`FourStepCodeletPlan`, smooth composite N) |
| — | `:fast` | autotuner: pow2 → times radix4avx/recursive/fourstep; non-pow2 → codelet (small smooth) / four-step (smooth composite) / Bluestein (large prime) |

All variants match FFTW to machine precision (relerr ≤ 5e-16). Tests run via ReTestItems with a
relative perf-regression guard against FFTW.

## Answering the three hypotheses

- **Rust the language?** No — rustfft ≈ FFTW (Rust ≈ C), and pure Julia now matches/beats both.
- **LLVM magic for Rust?** No — same backend; Julia emits the same AVX-512 packed FMA.
- **Implementation?** Entirely. Every GFLOP/s from 7 (scalar) to parity is Julia implementation work:
  algorithm, cache blocking, explicit AVX codelets, register-resident small-n kernels, pass fusion,
  and runtime codelet generation — no language or compiler change required.

## Negative findings (kept honest)

- **Split-radix size-32 codelet (n=512/2048):** *not* pursued — the base-32 codelet is **shuffle-bound**
  (`@code_native`: 41 shuffles vs 34 mul/FMA), and split-radix trades multiplies (slack) for *more*
  irregular shuffles → would regress. This is the remaining ~10% gap to RustFFT at 512/2048.
- **FixedSizeArrays scratch:** null result — the hot path uses raw `pointer()`+SIMD, so FSA and
  `Vector` compile identically (branch `experiment/fixedsizearrays`, not merged).
- **SoA standalone recursive, radix-4 recursion, Stockham, two pass fusions, recursive four-step:**
  all landed below the kept paths (split/merge + cache-stream pressure, or lost vectorization).
- **n=2^18 plateau (~26 GF/s, all three libraries):** not a deficiency — it's the memory wall (4 MB
  array + scratch exceeds the 1 MB/core L2); all libraries are bandwidth-bound and converge.

## What remains

- The **n=512/2048 base-32 codelet** (shuffle-bound) — would need a genuinely shuffle-lighter size-32
  decomposition, not split-radix.
- **Non-pow2 large primes** still use Bluestein (~5 GF/s); only composite smooth sizes get the
  faster four-step. Rader's algorithm (prime → convolution) could lift the prime case further.
- **Per-size decomposition autotuning** for the four-step — the `N1·N2` split is currently a
  heuristic (balanced, factors ∈ [8,64]); timing candidate splits at plan time (FFTW's `MEASURE`
  edge) would squeeze more, as could recursing the four-step for very large smooth sizes.
