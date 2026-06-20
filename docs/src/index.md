# PureFFT.jl

**A from-scratch pure-Julia FFT that matches or beats FFTW and rustfft's hand-tuned AVX path.**

## The headline result

PureFFT's `:fast` variant (autotuned) reaches **40–48 GFLOP/s** on power-of-two sizes (AMD Zen 5,
AVX-512, single-thread) and **matches or beats both FFTW (MEASURE) and rustfft-AVX across most of
the range** — leading at n=128/256/16384/65536, at parity at 1024/4096, trailing only at n=64 and
n=512. It also has **no non-power-of-two cliff** (Bluestein chirp-Z) and generates tailored kernels
at plan time (dynamic mixed-radix codelets) for sizes the static libraries fall back on.

The key finding: **same algorithm in both languages performs the same**. The gap between a naive
Julia FFT and rustfft-AVX was purely implementation work — algorithm choice, cache blocking, SIMD
codelets, pass fusion, register-resident small-n kernels. The language (Julia vs Rust) is not a
factor; both share the same LLVM backend. See the [Benchmarks](benchmarks.md) page for numbers and
the [Performance](performance.md) page for the engineering techniques that closed the gap.

## Install

```julia
using Pkg
Pkg.add(url = "https://github.com/el-oso/PureFFT.jl")
```

## Quick start

```julia
using PureFFT

x = randn(ComplexF64, 1024)

# One-shot transform (allocates output)
y = pfft(x)

# Plan once, apply many times (zero-allocation hot path)
plan = plan_pfft(x; variant = :fast)   # :fast = autotuned default
pfft!(x, plan)                          # in-place

# Inverse
ipfft!(x, plan)
```

All variants match FFTW to machine precision (relative error ≤ 5e-16); the ReTestItems suite
(correctness, JET dispatch-free, TrimCheck trim-safety, and a perf-regression guard) passes.

## Variants

| Variant | GFLOP/s (F64) | Notes |
|---|---:|---|
| `:scalar` | 7–9 | Radix-2 baseline |
| `:mixedradix` | — | Any N including primes |
| `:base` | 11–17 | `@simd`-staged radix-2 |
| `:recursive` | 13–24 | Cache-oblivious + `@generated` codelets |
| `:soa` | 13–21 | Split re/im recursive |
| `:fourstep` | 16–22 | Cache-blocked four-step (best at medium/large N) |
| `:radix4` | 27–28 | Port of rustfft's Radix4 |
| `:radix4avx` | **40–48** | Radix4 + AVX Butterfly16/32, radix-16 fusion, small-n register kernels |
| `:bluestein` | non-pow2 | chirp-Z, O(n log n) on primes |
| `:codelet` | non-pow2 | dynamically-generated mixed-radix kernel |
| `:fast` | **best-of** | Autotuner picks fastest per size (pow2 + non-pow2) |

Reference: FFTW/rustfft ≈ 35–46 GFLOP/s on the same hardware.

A real-input variant (`prfft`/`pirfft`) is under development.
