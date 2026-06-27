# Float32 pow2 — faithful RustFFT port to close the 0.96× gate

> **Status:** scoped, foundation set. Resumable. The bit-exact SIMD port below should be done with fresh
> focus, one layer at a time, verifying each against Rust/FFTW golden values before the next (the
> faithful-port methodology — reinterpreting/rushing it "repeatedly plateaued").

## Goal (hard gate — see memory `enforce-parity-gate`)
Every benchmarked pow2 `ComplexF32` size ≥ **0.96×** vs **FFTW AND RustFFT**. Currently 5 sizes fail:
256 (vs Rust 0.88), 512 (0.86), 2048 (0.90), 8192 (0.83), 32768 (0.84). **Float64 already clears all** —
the gap is F32-specific.

## Root cause (measured, in this session)
F64 clears the odd-power/small sizes via the **`AvxMixedRadixPlan` monolith path** (faithful ports of
RustFFT's **f64** `Butterfly256`/`Butterfly512`, which are `V4f = Vec{4,Float64}` = 2-complex/256-bit).
That path is `T===Float64`-gated. F32 falls to the `Radix4Avx` 256-bit base-32, which is less mature than
RustFFT's hand-tuned f32 kernels. RustFFT has **separate f32 butterflies** that pack **4 complex** per
`__m256` — which maps to PureFFT's **`Vec{8}` W8 layout** (`V8f`=512-bit F64, `V8f32`=256-bit F32), the
same 4-complex layout `B64W8`/`MR*W8` already use.

## What was fixed this session (committed `b3aeafb`, branch `perf/f32-pow2-rust`, validated, suite 1225/0)
- **Transpose-gate fix** (`radix4_avx.jl`): vectorized scratch transpose now used for ALL F32 sizes (the
  n≤2048 cap was a Float64-tuned false premise). Closed the even-power sizes (4096/16384/65536).
- **`MR4W8` radix-4 W8 kernel** + `plan_tree_w8` spans all pow2 (`rem2 = 3a + 2·c4`). Bit-exact. Marginal
  (W8 is 256-bit, narrower than `Radix4Avx`'s 512-bit cross for large sizes).
- **`autotune.jl`**: `AvxMixedRadixPlanW8` timed in the pow2 branch (F32's only monolith candidate).

## The port (the remaining work)
Faithfully port RustFFT 6.4.1's **f32** `Butterfly256Avx`/`Butterfly512Avx` into **generic `Vec{8,T}`** W8
monolith bases `B256W8`/`B512W8`, so F32 gets the same monolith F64 has — at the matching 256-bit width.
Source of truth: `~/.cargo/registry/src/index.crates.io-*/rustfft-6.4.1/src/avx/avx32_butterflies.rs`
(lines 1500–1720) + `avx32_utils.rs`. Verify each layer bit-exact (Rust golden harness in
`bench/rustfft_compare/`, or vs FFTW for whole butterflies).

### Rust structure (read from source)
- **`Butterfly256<f32>`** (256 = 32×8): phase-1 = `column_butterfly8` down 8 columnsets (load rows at
  `columnset*4 + 32*r`) → `mul_complex` twiddles `[r-1+7*columnset]` → `transpose8_packed` → store at
  `columnset*32 + i*8 (+4)`; phase-2 = `column_butterfly32` ×2 columnsets (load/store `columnset*4 +
  index*8`). twiddles: `gen_butterfly_twiddles_separated_columns!(8, 32, 0)` (56 vecs) + butterfly32's 6.
- **`Butterfly512<f32>`** (512 = 32×16): phase-1 = `column_butterfly16` ×16 (load `columnset*4 + 32*index`)
  → twiddles (15/column) → `transpose4_packed` per chunk → store `columnset*64 + row*16 + 4*chunk`;
  phase-2 = `column_butterfly32` ×4 (load/store `columnset*4 + index*16`). twiddles:
  `gen_butterfly_twiddles_separated_columns!(16, 32, 0)` (120) + bf32's 6 + bf16's 2.

### Mapping to PureFFT (generic over `Vec{8,T}`)
Already generic (`::Vec{8}`): `avx_column_butterfly8`, `avx_transpose8_packed`, `avx_transpose4_packed`,
`avx_column_butterfly4`, `avx_mul_complex`, `avx_rotate90`, `avx_bf8_tw1/3`, `avx_neg`, `_rot90_*`.
**Missing W8 primitives (the existing ones are `V4f`):**
1. `avx_column_butterfly32` — make width-generic. It is *already a faithful port* of Rust's
   `column_butterfly32_loadfn!` (`avxport.jl:209`); only its internal `avx_load_complex`/
   `avx_store_complex!` (V4f) and `tw::NTuple{6,V4f}` are width-specific. **Infer `VT` from the twiddle
   tuple, dispatch load/store on it** — do NOT rewrite the op structure.
2. `avx_column_butterfly16` — same treatment (`avxport.jl:234`).
3. twiddle generation (`gen_butterfly_twiddles_separated_columns`) at W8 — port the F64 `bf256_*_tw`/
   `bf512_*_tw` builders (`kernels.jl`) generic over `T`, producing `Vec{8,T}`.

### TypeContracts design (per the user — formalize the width-generic vector interface)
Define a `@contract` for the AVX complex-vector interface so the butterflies depend on the interface, not
concrete `V4f`/`V8f`/`V8f32`, and `@verify` (precompile, zero runtime, trim-safe) that all three satisfy it:
```julia
@contract AvxCVec begin
    _loadc(::Type{Self}, ::Any, ::Int)::Self        # width-dispatched load (V4f vs Vec{8})
    _storec!(::Type{Self}, ::Any, ::Int, ::Self)
    avx_mul_complex(::Self, ::Self)::Self
    avx_column_butterfly4(::Self, ::Self, ::Self, ::Self, ::Any)::NTuple{4,Self}
    # … the ops the W8 butterflies consume …
end
@verify AvxCVec  # at precompile, for V4f, V8f, V8f32
```
This is the "keep it generic while mechanically converting" guard: the contract makes the width-generic
requirement explicit and machine-checked.

### Kernels to add (`width8.jl`, mirror `B64W8`/`butterfly64_w8!`)
- `_colbf32_w8!` (phase-2 column_butterfly32) and the `B256W8`/`B512W8` phase-1 (cb8/cb16 + twiddle +
  transpose + store) — exact index/twiddle math copied from the Rust source above, NOT reinterpreted.
- `B256W8{T}`, `B512W8{T}` structs + constructors (generic over `T`, like `B64W8`).
- `plan_tree_w8`: use `B256W8`/`B512W8` as the base for the relevant pow2 sizes (Rust's `8xn` scheme: base
  256/512 + radix-8/4 chain), like the F64 `avxradix/planner.jl:66` does with `B256`/`B512`.

### Bit-exact gates (per layer, before proceeding)
1. W8 `avx_column_butterfly32` (V8f32): a size-32 transform == FFTW size-32 (≤1e-6 F32, ≤1e-13 F64).
2. W8 `avx_column_butterfly16` likewise (size-16).
3. `B256W8` standalone: size-256 == FFTW (both F32/F64). `B512W8`: size-512.
4. End-to-end via `plan_tree_w8`: 256/512/2048/8192/32768, F32+F64, bit-exact.
5. `@test_opt`/AllocCheck clean; F64 W8 (existing) not regressed.

### Success / parity gate
Re-run the authoritative measurement (`scratchpad/full.jl` pattern → reproducible
`bench/run_compare_f32.jl`): EVERY pow2 F32 size ≥ 0.96× vs FFTW AND Rust. If radix-4-based pow2 still
plateaus < 0.96× at the very small sizes (256/512) like the non-pow2 radix-9/12 floor did, that is a
finding to surface — not a silent "done".

## Risk
- The non-pow2 faithful port "repeatedly plateaued ~0.5–0.85× when reinterpreted"; only mechanical op-for-op
  reached parity. Same discipline required here.
- 256/512 vs RustFFT's hand-tuned tiny kernels may be the hardest; the monolith (Rust's own kernel) is the
  best shot, but verify it actually clears 0.96×, don't assume.
