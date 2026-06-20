# PureFFT.jl

[![CI](https://github.com/el-oso/PureFFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/PureFFT.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureFFT.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch, pure-Julia FFT built as an **investigation**: *where does the speed of
rustfft / FFTW actually come from — the algorithm, the LLVM compiler, or Rust the language? And
can pure Julia match it?*

See **[REPORT.md](REPORT.md)** for the full findings, including an honest accounting of what is and
isn't proven. Short version: **on the real workload (one transform of size N), Rust's `rustfft` is
~2× faster than this pure-Julia FFT** — parity was not reached. What *is* established: rustfft (Rust)
≈ FFTW (C) (a Rust-vs-C result), and Julia's compiled *batched* kernel hits ~38 GFLOP/s on a
friendly microbenchmark (necessary, not sufficient). The belief that the remaining 2× is
implementation rather than language is reasonable but **unproven** — it would take a Julia FFT at
parity, or a same-algorithm Julia-vs-Rust head-to-head, to settle.

## Layout

```
src/         core deps: AbstractFFTs, TypeContracts (compile-time plan contract)
  contracts.jl   AbstractFFTPlan interface (TypeContracts @contract/@verify, zero runtime cost)
  radix2.jl      Stage 1: scalar radix-2 baseline
  mixedradix.jl  Stage 2: mixed-radix, any N incl. primes
  staged.jl      Stage 3: staged radix-2 + Base @simd kernel
  codelets.jl    @generated straight-line DFT codelets (Julia's genfft-equivalent; AoS + SoA)
  recursive.jl   Stage 4: cache-oblivious recursive DIT + generated codelets
  soa.jl         Stage 5: split re/im recursive (negative finding — see REPORT)
  blocked.jl     Stage 7: cache-blocked four-step + batched SIMD kernel (best at medium/large N)
  autotune.jl    Stage 9: :fast — times candidates, keeps the fastest
test/        correctness vs FFTW (392 tests)
bench/       compare.jl, batched_proof.jl (≥FFTW kernel), llvm_inspect.jl, alloccheck.jl
```

## Use

```julia
using PureFFT
x = randn(ComplexF64, 4096)
y = pfft(x)                        # forward
x2 = ipfft(y)                      # inverse (normalized)

p = plan_pfft(x; variant = :fast)  # autotuned. ∈ :scalar :mixedradix :base :recursive :soa :fourstep :fast
pfft!(x, p)                        # in-place, zero allocations
```

### Real-input FFT (rfft / irfft)

```julia
using PureFFT
x = randn(Float64, 4096)           # real input, even length

X = prfft(x)                       # forward: length n÷2+1 complex output
x2 = pirfft(X, length(x))         # inverse: back to real, normalized by 1/n

# Or via AbstractFFTs interface:
using AbstractFFTs
X = AbstractFFTs.rfft(x)
x2 = AbstractFFTs.irfft(X, length(x))

# Zero-alloc hot path via plan:
p  = plan_prfft(Float64, 4096)
ip = plan_pirfft(Float64, 4096)
out_c = p.outbuf
PureFFT.apply_rfft!(p, x, out_c)   # 0 bytes after warmup
out_r = similar(x)
PureFFT.apply_irfft!(ip, Complex{Float64}.(out_c), out_r)
```

## Reproduce

```bash
julia --project=.     -e 'using Pkg; Pkg.test()'      # correctness (392 tests)
julia --project=bench bench/compare.jl                # FFTW / RustFFT / PureFFT table
julia --project=bench bench/batched_proof.jl          # batched kernel ≥ FFTW throughput
julia --project=bench bench/llvm_inspect.jl           # confirm AVX-512 autovectorization
julia --project=bench bench/alloccheck.jl             # confirm zero allocations
```
