# API Guide

## Public API

### `plan_pfft`

```julia
plan_pfft(x::AbstractVector{<:Complex}; variant = :fast) -> Plan
```

Build a plan for an in-place forward FFT of a vector with the same length as `x`. The `variant`
keyword selects the algorithm (see [Variants](#variants) below). Planning is cheap — no
measurement is performed at this stage; `:fast` auto-selects at first use.

### `pfft`

```julia
pfft(x::AbstractVector{<:Complex}; variant = :fast) -> Vector
```

Allocating forward FFT. Builds a plan internally and applies it to a copy of `x`. Convenient for
one-off transforms; use `pfft!` in performance-sensitive loops.

### `pfft!`

```julia
pfft!(x::AbstractVector{<:Complex}, plan) -> x
```

In-place forward FFT. Applies `plan` to `x`, overwriting it with the transform result.
Zero-allocation on the hot path (verified with AllocCheck).

### `ipfft`

```julia
ipfft(x::AbstractVector{<:Complex}; variant = :fast) -> Vector
```

Allocating inverse FFT. Returns the unnormalized inverse (divide by `n` if you need the
normalized form, matching FFTW's convention).

### `ipfft!`

```julia
ipfft!(x::AbstractVector{<:Complex}, plan) -> x
```

In-place inverse FFT. The `plan` returned by `plan_pfft` is reusable for both forward and inverse
transforms.

## Variants

| Variant | Description |
|---|---|
| `:scalar` | Plain radix-2 DIT, no SIMD annotations. Baseline. |
| `:mixedradix` | Mixed-radix Cooley-Tukey for any N (including primes via DFT fallback). |
| `:base` | Radix-2 with `@simd ivdep` across the cross-pass loop. |
| `:recursive` | Cache-oblivious recursive decomposition with `@generated` base codelets. |
| `:soa` | Split-of-array: separate `re`/`im` arrays, shuffle-free combine. |
| `:fourstep` | Cache-blocked four-step algorithm: best for medium/large N. |
| `:radix4` | Port of rustfft's `Radix4`: bit-reversed transpose + log₄ Butterfly4 cross-passes. |
| `:radix4avx` | `:radix4` + explicit SIMD.jl AVX Butterfly4 cross-pass + AVX base codelets. |
| `:fast` | Autotuned: picks the fastest variant for each size at runtime. Default. |

## Examples

```julia
using PureFFT

# Basic usage
x = randn(ComplexF64, 4096)
y = pfft(x)                          # allocating

# Plan-based (recommended for repeated transforms)
plan = plan_pfft(x; variant = :fast)
pfft!(x, plan)                        # forward, in-place
ipfft!(x, plan)                       # inverse, in-place (x restored)

# Force a specific variant
plan_r4avx = plan_pfft(x; variant = :radix4avx)
pfft!(x, plan_r4avx)

# Non-power-of-two sizes
xp = randn(ComplexF64, 1500)
plan_p = plan_pfft(xp; variant = :mixedradix)
pfft!(xp, plan_p)

# Correctness check vs FFTW
import FFTW
x = randn(ComplexF64, 1024)
y_pure = pfft(copy(x))
y_fftw = FFTW.fft(x)
@assert maximum(abs, y_pure .- y_fftw) / maximum(abs, y_fftw) < 1e-13
```

## AbstractFFTs integration

PureFFT registers itself with AbstractFFTs.jl, so you can use the standard `fft`/`ifft` interface
from packages that accept any FFT provider:

```julia
using AbstractFFTs, PureFFT
using LinearAlgebra  # for mul!

x = randn(ComplexF64, 512)
plan = plan_fft(x)   # AbstractFFTs entry point, dispatches to PureFFT
y = plan * x
```
