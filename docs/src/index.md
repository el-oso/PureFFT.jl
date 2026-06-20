# PureFFT.jl

**A from-scratch pure-Julia FFT that reaches parity with rustfft's hand-tuned AVX path.**

## The headline result

PureFFT's `:fast` variant (autotuned) achieves **38–40 GFLOP/s** on power-of-two sizes — within
~1.2× of rustfft's hand-tuned AVX planner across all tested sizes, and **faster at n=65536
(1.04×)**. It is **~2× faster than rustfft-scalar**. The CPU is AMD Zen 5 (AVX-512), single-thread.

The key finding: **same algorithm in both languages performs the same**. The gap between a naive
Julia FFT and rustfft-AVX is purely implementation work — algorithm choice, cache blocking, SIMD
vectorization. The language (Julia vs Rust) is not a factor; both share the same LLVM backend and
emit equivalent native code for equivalent source. See the [Benchmarks](benchmarks.md) page for
numbers and the [Performance](performance.md) page for the engineering techniques that closed the gap.

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

All variants match FFTW to machine precision (relative error ≤ 5e-16). 392 tests pass.

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
| `:radix4avx` | 35–40 | Radix4 + explicit AVX Butterfly16/32 |
| `:fast` | **best-of** | Autotuner picks fastest per size |

Reference: FFTW/rustfft ≈ 35–46 GFLOP/s on the same hardware.

A real-input variant (`prfft`/`pirfft`) is under development.
