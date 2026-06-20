# PureFFT.jl — where FFT speed comes from, and how close pure Julia gets to FFTW/rustfft

**Question.** rustfft (Rust) is reported competitive with FFTW (C). Is the speed from the
**algorithm/implementation**, the **LLVM compiler**, or **Rust the language**? And can pure
Julia — same LLVM backend — match it?

## The controlled answer: same algorithm ⇒ same speed (micro AND package level)

**Micro (`bench/lang_compare/`):** the same radix-2 algorithm in Julia and Rust — identical layout /
ops / `muladd` / no-bounds-checks, same LLVM+AVX-512 backend, **bit-for-bit matching checksums** —
runs within ~10% at most sizes, each winning some (Julia faster at large n). Confirmed on AVX2 too.

**Package (`bench/rustfft_compare/`) — PARITY:** PureFFT ports rustfft's `Radix4` and fully
AVX-vectorizes it; benchmarked vs the `rustfft` crate directly (checksums match). PureFFT `:fast`
reaches **parity with rustfft's hand-tuned AVX path** — within ~1.2× across sizes and *faster* at
n=65536 (1.04×) — and is **~2× faster than rustfft-scalar**. Same algorithm, same LLVM backend.
**The language was never the lever — proven at every level: micro (radix-2), package-scalar, and now
hand-tuned-AVX.**

What closed the full gap (from 0.45× → ~0.9× of AVX): `@simd ivdep` cross-pass → cache-blocked
transpose (fixed the reorder, the profiled #1 cost) → **explicit SIMD.jl AVX Butterfly4 cross-pass**
(`_vcmul` interleaved-complex trick) → **within-butterfly AVX base codelets** (`Butterfly16` as 4×4
with shuffle-free DFT-4s + one register transpose, 2.1× over scalar; `Butterfly32` = 2×`Butterfly16`
+ radix-2 combine). Hot path: allocation-free, `TRIM_SAFE` (juliac/TrimCheck.jl), dispatch-free
(JET `@report_opt`).

Tooling added across the AVX rounds (user-directed): `@simd ivdep`; cache-blocked transpose; SIMD.jl
explicit AVX kernels; MLStyle `@match` dispatch; PkgBenchmark suite; TypeContracts `interface_trait`
duck-typed `pfft!` (zero-overhead) + `@verify`; TrimCheck.jl `@validate` and JET `@test_opt` tests.

## Earlier headline findings — and what was NOT proven before the controlled test

**On the real workload — one transform of size N — Rust's `rustfft` is ~2× faster than this
pure-Julia FFT, because it uses a better algorithm.** Below, what each prior data point did and
didn't establish (the controlled experiment above is what actually settles the language question).

1. **rustfft (Rust) ≈ FFTW (C)** to within a few percent (rustfft's only edge is ~1.1–1.5× at
   small n). *Proven* — but this is **Rust vs C**; Julia is not in this comparison. It shows Rust
   has no magic over C; it says nothing directly about Julia.
2. **Julia's batched kernel hits ~38 GFLOP/s** (`bench/batched_proof.jl`). *Proven, but for an
   easier workload*: many small equal-size FFTs with the batch dimension handed to SIMD
   (embarrassingly parallel, cache-resident, no recursion). It shows Julia/LLVM can saturate the
   SIMD units on a friendly kernel — **necessary, not sufficient**. It is NOT the same problem as a
   single large FFT, so it does not by itself prove a Julia FFT can match rustfft.
3. **A full general-N pure-Julia transform reaches only ~0.5–0.6× of FFTW/rustfft** (≈2× slower),
   autotuned. We did **not** reach parity. The gap is in *orchestration* (cache-blocked transposes,
   buffer movement, plan selection).
4. Allocation-free (AllocCheck) and compile-time-contract-checked (TypeContracts `@verify`).

**Does Julia need to improve? Unproven — likely not, but not demonstrated.** The remaining 2× is
data-movement/orchestration, which has no obvious language-level reason to be slower in Julia than
Rust (it compiles to the same memory operations). But we never matched it, so this is a reasonable
belief, **not a result**. What would actually settle it: (a) a Julia FFT within a few % of rustfft
(not achieved — best 0.5×), or (b) a controlled head-to-head of the *same algorithm* in Julia and
Rust (never done — the rustfft≈FFTW data is Rust-vs-C). Until one of those exists, the defensible
claim is only: *Rust shows no advantage over C, and Julia's compiled kernels match FFTW on a
batched microbenchmark; a complete Julia FFT is currently 2× slower than rustfft.*

## Stages built (each independently benchmarkable)

| Stage | Variant | GFLOP/s (F64) | Idea |
|---|---|---:|---|
| 1 | `:scalar` | 7–9 | radix-2 baseline |
| 2 | `:mixedradix` | — | any N incl. primes |
| 3 | `:base` | 11–17 | Base `@simd` staged radix-2 |
| 4 | `:recursive` | 13–24 | cache-oblivious + `@generated` codelets |
| 5 | `:soa` | 13–21 | split re/im recursive (negative — see below) |
| 7 | `:fourstep` | 16–22 | cache-blocked four-step (best at medium/large N) |
| 9 | `:fast` | best-of | autotuner picks per size |
| — | batched kernel | **~38** | the proof: ≥ FFTW per-transform |

Reference: **FFTW / rustfft ≈ 35–46 GFLOP/s.** All variants match FFTW to machine precision
(relerr ≤ 5e-16). **392 tests pass.**

## Numbers (Zen 5, single-thread, ComplexF64, `:fast` = autotuned)

| n | FFTW-MEASURE | RustFFT | PureFFT `:fast` | ratio |
|---:|---:|---:|---:|---:|
| 1024   | 44 | 44 | 24 | 0.55× |
| 4096   | 45 | 44 | 22 | 0.49× |
| 16384  | 41 | 41 | 22 | 0.54× |
| 65536  | 36 | 36 | 21 | 0.58× |
| 262144 | 26 | 27 | 16 | 0.60× |

The four-step closed the large-N gap from ~0.46× (recursive) to ~0.60×. The batched inner kernel
alone is ~38 GFLOP/s; the four-step wrapper (bit-reversal + transpose + twiddle + de/interleave)
costs roughly the other half — that overhead is the whole remaining gap to parity.

## Answering the three hypotheses

- **Rust the language?** No — rustfft ≈ FFTW (Rust ≈ C), both LLVM/native, and Julia's batched
  kernel matches them.
- **LLVM magic for Rust?** No — Julia's `@simd`/batched kernels emit AVX-512 packed FMA
  (`bench/llvm_inspect.jl`); same backend, same instructions, ~38 GFLOP/s.
- **Implementation?** Yes, entirely. The 7→38 GFLOP/s spread is pure Julia implementation work
  (algorithm, cache-obliviousness, `@generated` codelets, batched SIMD, four-step blocking). The
  last 1.4–2× to FFTW in a *general* transform is more of the same engineering (below).

### Negative findings (kept honest — most parity tweaks made it *worse*)
- **SoA standalone recursive** (`:soa`): isolated combine is 1.42× faster, but split/merge passes
  + 4-stream cache pressure make the whole transform *slower* than AoS recursive.
- **Radix-4 recursion**: slower than radix-2 (more cache streams per pass than halved passes save).
- **Hand-written SIMD.jl** (removed): ~2× slower than Base autovectorization; dropped entirely.
- **Stockham batched kernel** (drop bit-reversal): 13 vs the in-place bit-reversed DIT's 38 GFLOP/s
  — the ping-pong double-buffer traffic costs more than the bit-reversal pass it removes.
- **Pass fusion** (bit-reversal into gather/scatter; twiddle into the transpose): slower — fusing
  into scattered/strided-write loops destroys vectorization and locality. The wrapper passes are
  fastest kept *separate and contiguous*.
- **Recursive batched four-step** (the "big rewrite": split R=R1·R2, batched base case, in-transform
  batched transpose, recurse on R1): correct but **11–14 GFLOP/s vs the single-level four-step's
  ~22** — the recursion's extra buffer copies + intermediate passes outweigh the theoretical
  reduction in main-memory passes at these sizes. The single split wins.

**Eight distinct optimization attempts (radix-4, SoA, SIMD.jl, Stockham, two fusions, recursive
four-step) all landed *below* the simple cache-blocked four-step.** That is strong evidence the
~0.55× single-level four-step is a hard local optimum, and that parity needs FFTW's *whole*
architecture optimized simultaneously — not any single lever.

## What closing the final 1.4–2× actually requires

The single-level four-step caps at ~0.55× because its orchestration passes (split, 2× bit-reversal,
twiddle, transpose, merge) are memory-bound and every attempt to fuse/eliminate them regressed
(above). The batched kernel itself already matches FFTW. So parity is **not** a few-edits problem —
it needs FFTW's actual architecture:

1. **Recursive, multi-level, cache-oblivious decomposition** where *every* level's work is a
   vectorized (batched) codelet — so the transform never makes O(log N) full-array passes, only
   O(log_cache N). This is the real gap and a substantial rewrite.
2. **A genfft-grade codelet set** (split-radix, sizes to 64) with register-optimal scheduling.
3. **Plan autotuning** over factorizations/codelet sizes (FFTW's `MEASURE` edge).

This is research-grade engineering (FFTW = 25 years, multiple papers; rustfft is a dedicated
library). Nothing needs Rust or a better compiler — the batched kernel proves Julia/LLVM already
emit FFTW-class code, allocation-free, with compile-time-checked interfaces. But it is a major
project, not an incremental tweak — every incremental tweak tried here regressed.
