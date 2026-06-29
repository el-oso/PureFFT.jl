# PureFFT.jl

[![CI](https://github.com/el-oso/PureFFT.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/el-oso/PureFFT.jl/actions/workflows/CI.yml)
[![Coverage](https://coveralls.io/repos/github/el-oso/PureFFT.jl/badge.svg?branch=master)](https://coveralls.io/github/el-oso/PureFFT.jl?branch=master)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://el-oso.github.io/PureFFT.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A from-scratch, pure-Julia FFT built as an **investigation**: *where does the speed of
rustfft / FFTW actually come from — the algorithm, the LLVM compiler, or Rust the language? And
can pure Julia match it?*

**Answer: yes — pure Julia matches it.** PureFFT's `:fast` (autotuned) reaches **40–48 GFLOP/s** on
power-of-two sizes (AMD Zen 5, AVX-512, single-thread) and **matches or beats both FFTW (MEASURE) and
rustfft-AVX across most of the range** — power-of-two and smooth non-power-of-two alike, with no size
cliff. The gap an earlier round measured (~2×) was **implementation, not language**: algorithm choice,
cache blocking, SIMD codelets, radix-16 pass fusion, register-resident small-n kernels, and a *faithful
op-for-op port* of rustfft's AVX mixed-radix (`src/avxradix/`) closed it. Julia and Rust share the LLVM
backend — same algorithm, same speed. PureFFT even runs an **AVX-512 (`Vec{8}`) non-pow2 path that beats
rustfft** (which is AVX2-only) on the sizes it covers — see the
[docs](https://el-oso.github.io/PureFFT.jl/dev/) (Benchmarks + Performance) for numbers and the
engineering. (The remaining gap is a ~0.85–0.92× floor on 3-heavy / radix-9-12 non-pow2 sizes — see
`docs/src/performance.md` §15.)

> Historical note: an early `REPORT.md` concluded the 2× was unproven; it predates the radix-4 AVX
> engine and the faithful port and is superseded by the current results.

## Where this is headed

PureFFT began as an investigation — *can pure Julia match FFTW/rustfft?* The answer turned out to be
**yes**, and that changes what it can be. The goal now is to **mature it into a general-purpose FFT
library that is also a showcase of what Julia makes possible**:

> A **general-purpose, pure-Julia, MIT-licensed, dependency-free** FFT that **matches FFTW and RustFFT in
> speed** — and does what an ahead-of-time C/Rust library structurally *cannot*: **on-demand codelet
> specialization** (`@generated` / runtime codegen), **one type-generic kernel set** across
> `Float32`/`Float64`, and **`AbstractFFTs`-native composability**.

Two of those are a concrete, practical reason to reach for PureFFT *today*, independent of the showcase:
it is **pure Julia with no binary dependency** and **MIT-licensed**, where FFTW is **GPL** (via `FFTW.jl`)
— which rules FFTW out for some permissively-licensed and commercial users.

The showcase features and the library features are mostly the **same investment** — robust any-size
coverage, type-generic kernels, and ecosystem composability serve both at once. The one piece that is
"library, not showcase" is **multi-threading**: a real, scheduled milestone (see `ROADMAP.md`), but not
first. The flagship next step is the **Julia-native codelet generator** — the genfft analogue only Julia
can build.

## Layout

```
src/         core deps: AbstractFFTs, TypeContracts (compile-time plan contract)
  contracts.jl    AbstractFFTPlan interface (TypeContracts @contract/@verify, zero runtime cost)
  radix2/mixedradix/staged/recursive/soa/blocked.jl   variant ladder: scalar → cache-blocked four-step
  codelets.jl     @generated straight-line DFT codelets (Julia's genfft-equivalent; AoS + SoA)
  radix4_avx.jl   the power-of-two AVX-512 radix-4 engine (:radix4avx, 40–48 GFLOP/s)
  avxradix/       faithful op-for-op port of rustfft's AVX mixed-radix (non-pow2);
                  width8.jl = the AVX-512 (Vec{8}, W=8) variant that beats rustfft
  rader.jl, bluestein.jl   prime-size paths (Rader cyclic convolution / Bluestein chirp-Z)
  autotune.jl     :fast — times candidates per size, keeps the fastest
test/        correctness + JET dispatch-free + TrimCheck + StrictMode + perf guard (ReTestItems, ~970 checks)
bench/       plot_compare.jl (comparison plots), cpufreq_lock.sh (stable benchmarking),
             strictmode_audit.jl, alloccheck.jl, llvm_inspect.jl
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
julia --project=.     -e 'using Pkg; Pkg.test()'      # correctness + JET + TrimCheck (~970 checks)
julia --project=bench bench/compare.jl                # FFTW / RustFFT / PureFFT table
julia --project=bench bench/batched_proof.jl          # batched kernel ≥ FFTW throughput
julia --project=bench bench/llvm_inspect.jl           # confirm AVX-512 autovectorization
julia --project=bench bench/alloccheck.jl             # confirm zero allocations
```
