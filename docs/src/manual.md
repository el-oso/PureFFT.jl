# PureFFT Manual (FFTW users' map)

This manual mirrors the structure of the [FFTW manual](https://www.fftw.org/fftw3_doc/) and maps each
FFTW concept to its PureFFT equivalent, so an FFTW user can find what they need. It documents **only what
PureFFT currently implements**; the [Gaps vs FFTW](#gaps-vs-fftw) section at the end lists what is not (yet)
available.

For task-oriented walk-throughs with figures, see the [API Guide](guide.md). This page is the reference map.

## Introduction

PureFFT is a from-scratch, dependency-free Julia FFT library. Where FFTW is a C library you call through
`ccall`/FFTW.jl, PureFFT *is* Julia — plans are Julia objects, transforms are Julia functions, and it plugs
into `AbstractFFTs.jl` so `fft`/`ifft`/`rfft`/`plan_fft`/`mul!`/`\` all work. On this machine it matches or
beats FFTW and RustFFT across power-of-two and non-power-of-two sizes (`Float64` and `Float32`).

Two ways to call it:

- **Native API** — `pfft`/`ipfft`, `prfft`/`pirfft`, `r2r`/`dct`, always PureFFT.
- **`AbstractFFTs` API** — `fft`/`ifft`/`rfft`/`plan_fft`, the idiomatic Julia interface (PureFFT is the
  provider when FFTW.jl isn't also loaded).

## Data Types and Conventions

| FFTW | PureFFT |
|---|---|
| `fftw_complex` (interleaved double) | `Complex{Float64}` / `Complex{Float32}` arrays (interleaved; standard Julia) |
| `double` real arrays | `Float64` / `Float32` arrays |
| `fftw_malloc` for SIMD alignment | not needed — plain `Array`s; alignment is handled internally |
| in-place vs out-of-place plans | `pfft!`/`mul!` (in place) vs `pfft`/`p*x` (allocating); `plan_fft!` for in-place |
| **unnormalized** transforms (you divide by N) | **same** — forward·inverse = N; `ifft`/`idct` apply the 1/N for you |
| row-major arrays | Julia is **column-major** — the first dimension is contiguous (the natural r2c dim) |

## Complex DFTs

FFTW: `fftw_plan_dft_1d` / `fftw_plan_dft` / `fftw_plan_dft_2d`/`3d`.

```julia
using PureFFT                       # native
x = randn(ComplexF64, 4096)
p = plan_pfft(x); pfft!(x, p)       # in-place 1-D; ipfft!(x, p) inverts
A = randn(ComplexF64, 256, 256, 64)
G = pfft(A)                         # all dims;  pfft(A, (1,3)) → only dims 1 and 3

using AbstractFFTs                  # or the generic interface
F = fft(A); fft(A, 2); ifft(F)      # N-D fft / per-dim / inverse — plan_fft, mul!, \, inv all work
```

Any rank, any subset of dimensions (a `region`: `Int`, tuple, range, or `:`) — full FFTW generality.
N-D is separable but fast (strided dims vectorize across the contiguous batch; no transpose).

## Real-data DFTs (r2c / c2r)

FFTW: `fftw_plan_dft_r2c` / `fftw_plan_dft_c2r`. A length-`n` real input has a Hermitian spectrum, so only
`n÷2+1` complex outputs are stored.

```julia
using PureFFT
s = randn(512); S = prfft(s)        # 257-element half-spectrum;  pirfft(S, 512) inverts (needs original n)

using AbstractFFTs                  # N-D real
img = randn(256, 256)
R = rfft(img)                       # 129×256;  irfft(R, 256) inverts (give the first-dim length)
```

`prfft` requires an **even** transformed length (matches FFTW's preferred r2c case). The r2c dimension is
the first element of `region` (halved), FFTW's convention.

## Real-to-real Transforms (DCT / DST)

FFTW: `fftw_plan_r2r` with a `kind`. PureFFT implements all eight **DCT/DST** kinds, named exactly as FFTW:

| PureFFT | FFTW | | PureFFT | FFTW |
|---|---|---|---|---|
| `REDFT00` | DCT-I | | `RODFT00` | DST-I |
| `REDFT10` | DCT-II | | `RODFT10` | DST-II |
| `REDFT01` | DCT-III | | `RODFT01` | DST-III |
| `REDFT11` | DCT-IV | | `RODFT11` | DST-IV |

```julia
using PureFFT
y = r2r(v, REDFT10)                 # unnormalized DCT-II, like FFTW.r2r(v, FFTW.REDFT10)
c = dct(v); idct(c)                 # orthonormal DCT-II + inverse (FFTW.jl / scipy norm="ortho")
plan_r2r(v, RODFT11) \ w            # inverse via the plan
```

Bit-exact vs `FFTW.r2r` for `Float64`/`Float32`, any `N`; small `N` uses fully-unrolled `@generated`
codelets, large `N` the real-FFT reduction. **Not implemented:** the `R2HC`/`HC2R` half-complex and `DHT`
(Hartley) r2r kinds — see gaps.

## Plans: Creation, Execution, Reuse

| FFTW | PureFFT |
|---|---|
| `fftw_plan_*` (planning) | `plan_pfft` / `plan_r2r` / `AbstractFFTs.plan_fft` / `plan_rfft` |
| `fftw_execute(plan)` | `pfft!(x, p)` / `mul!(y, p, x)` |
| new-array execute (`fftw_execute_dft(p, in, out)`) | `mul!(y, p, x)` — one plan, reused on any same-size/type arrays |
| `fftw_plan_with_flags(ESTIMATE/MEASURE/…)` | `variant = :fast` autotunes at first use (times candidate kernels, caches the winner in the plan) |
| `fftw_destroy_plan` | GC — plans are ordinary Julia objects |

Planning is cheap and pure-Julia; there is no separate measurement phase to manage, no wisdom file to load.

## The AbstractFFTs Interface

The idiomatic path. PureFFT registers `plan_fft`/`plan_fft!`/`plan_bfft`, `plan_rfft`/`plan_brfft`, and the
plans are `AbstractFFTs.Plan`s, so the whole generic surface works — 1-D and N-D, real and complex:

```julia
using AbstractFFTs, PureFFT, LinearAlgebra
A = randn(ComplexF64, 128, 96)
P = plan_fft(A, (1, 2))
B = P * A; mul!(similar(A), P, A); A ≈ inv(P) * B; A ≈ ifft(fft(A))   # all true
```

> If FFTW.jl is loaded in the same session it wins dispatch for concrete `Array`s. Call `PureFFT.pfft(...)`
> (or don't load FFTW) to force PureFFT.

## Gaps vs FFTW

What FFTW offers that PureFFT does **not** (yet). None of these block the common DFT/real/DCT workflows above.

| FFTW feature | Status in PureFFT | Notes |
|---|---|---|
| **Planner rigor flags** (`ESTIMATE`/`MEASURE`/`PATIENT`/`EXHAUSTIVE`) | ✗ | One `:fast` autotuner; no user-tunable planning effort knob. |
| **Wisdom** (export/import plans across sessions) | ✗ | Autotune results live only in the plan object; no persistence/serialization. |
| **Advanced interface** (`fftw_plan_many_dft`: `howmany`/`istride`/`idist`) | ◑ | The N-D plan covers batched transforms over array dimensions; arbitrary stride/distance batching is not exposed. |
| **Guru / guru64 interface** (arbitrary I/O tensors) | ✗ | No guru API. |
| **Split-complex arrays** (`fftw_plan_guru_split_dft`) | ✗ | Interleaved `Complex` only (some kernels are SoA internally, not exposed). |
| **r2r `R2HC` / `HC2R`** (half-complex) | ✗ | Use `rfft`/`irfft` (r2c/c2r) for real FFTs; the in-place half-complex format isn't provided. |
| **r2r `DHT`** (discrete Hartley) | ✗ | Not implemented. |
| **Multi-threading** | ✗ | Single-threaded (a planned ROADMAP item, deferred below the flagship codelet-generator research). |
| **Distributed memory (MPI)** | ✗ | Out of scope. |
| **`long double` / `__float128` precision** | ✗ | `Float64` and `Float32` only. |
| **C / Fortran callable** | n/a | Julia-native; call from Julia or via `AbstractFFTs`. |
| **SIMD-aligned allocation** (`fftw_malloc`) | n/a | Plain `Array`s; alignment handled internally. |

Legend: ✗ not available · ◑ partial · n/a not applicable to a Julia-native library.
