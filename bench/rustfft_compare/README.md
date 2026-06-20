# Full package vs full package: PureFFT.jl (Julia) vs the rustfft crate (Rust)

Same algorithm, two languages, at the **package** level. PureFFT's `:radix4` is a faithful port of
rustfft's `Radix4` (bit-reversed transpose → base butterflies → log₄ Butterfly4 cross-passes). We
benchmark it against the `rustfft` crate's **scalar** planner (same algorithm — the apples-to-apples
checkpoint) and its **AVX** planner (hand-written SIMD codelets — the north star).

Both at max optimization on the same LLVM + Zen 5 (AVX-512) backend; identical copy-subtract harness
(25-trial min, deterministic input). **Checksums match bit-for-bit across all four implementations**
→ they compute the same transform.

## Results — ns per transform (lower is better)

After the AVX-round optimizations to PureFFT `:radix4` (`@simd ivdep` cross-pass + **cache-blocked
transpose** with the digit-reversal folded into the base-butterfly source):

| n | PureFFT `:radix4` | rustfft scalar | rustfft AVX | radix4 ÷ scalar | radix4 ÷ AVX |
|---:|---:|---:|---:|---:|---:|
| 1024   | 1854  | 2526   | 1183   | **0.73×** | 1.57× |
| 4096   | 8757  | 12876  | 5217   | **0.68×** | 1.68× |
| 16384  | 41378 | 59406  | 25966  | **0.70×** | 1.59× |
| 65536  | 196684| 296492 | 160933 | **0.66×** | 1.22× |
| 262144 | 1.26M | 1.46M  | 0.91M  | **0.86×** | 1.38× |

GFLOP/s: PureFFT `:radix4` ~27–28, rustfft-scalar ~17–20, rustfft-AVX ~26–51.

## Final: PARITY with rustfft-AVX (variant `:radix4avx` / autotuned `:fast`), GFLOP/s

Full AVX implementation: cache-blocked transpose + explicit SIMD.jl Butterfly4 cross-pass +
**within-butterfly AVX base codelets** (`Butterfly16` as 4×4 with a register transpose; `Butterfly32`
as two `Butterfly16` + radix-2 combine).

| n | PureFFT `:fast` | rustfft-scalar | rustfft-AVX | vs scalar | vs AVX |
|---:|---:|---:|---:|---:|---:|
| 1024   | 38 | 21 | 44 | **1.8×** | 0.86× |
| 4096   | 40 | 19 | 48 | **2.1×** | 0.83× |
| 16384  | 37 | 20 | 44 | **1.9×** | 0.83× |
| 65536  | 37 | 19 | 35 | **1.9×** | **1.04×** |
| 262144 | 28 | 17 | 30 | **1.6×** | 0.92× |

Odd-power sizes (base-32 AVX) likewise reach ~34–37 GFLOP/s, matching/beating FFTW-MEASURE.

## Conclusion

**A from-scratch pure-Julia FFT reaches parity with `rustfft`'s hand-tuned AVX path** (within
~1.2× across sizes, *faster* at n=65536) and is **~2× faster than rustfft-scalar** — same
algorithm, same LLVM backend. The full arc that got here:
- `@simd ivdep` cross-pass (22→28 GFLOP/s)
- cache-blocked transpose (→30; fixed the reorder, the profiled #1 cost)
- explicit SIMD.jl AVX Butterfly4 cross-pass (`_vcmul` interleaved-complex trick)
- **within-butterfly AVX base codelets** (`Butterfly16` 4×4 with shuffle-free DFT-4s + one register
  transpose → 2.1× over scalar; `Butterfly32` = 2×`Butterfly16` + radix-2 combine)

Hot path: allocation-free, `TRIM_SAFE` (juliac, via TrimCheck.jl), dispatch-free (JET-verified).
**The language was never the lever** — proven at the micro level (`../lang_compare/`), the package
level (scalar), and now at the hand-tuned-AVX level too.

## Run it

```bash
cd rust && cargo run --release                                   # rustfft scalar + AVX
julia -O3 --project=.. ../rustfft_compare/julia_radix4.jl        # PureFFT :radix4 + :fast
# (or:  julia -O3 --project=bench bench/rustfft_compare/julia_radix4.jl  from the package root)
```
