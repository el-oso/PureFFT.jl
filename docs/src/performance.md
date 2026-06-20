# Julia FFT Performance Tricks

This page documents the concrete techniques that moved PureFFT from 7 GFLOP/s (scalar baseline) to
~38–40 GFLOP/s (parity with rustfft-AVX), with the measured effect of each. These are general
Julia/SIMD patterns; the FFT context makes the tradeoffs concrete.

## 1. `@simd ivdep` for cross-pass loops

**Effect: 22 → 28 GFLOP/s** on the Butterfly4 cross-pass.

The standard `@simd` annotation tells Julia the loop body can be reordered and vectorized. The
`ivdep` qualifier additionally asserts **no loop-carried data dependencies** — necessary when the
compiler cannot prove independence (e.g. butterfly reads from the same array it writes, but at
non-overlapping indices that the compiler cannot statically verify).

```julia
@simd ivdep for k in 0:(m - 1)
    @inbounds begin
        a = x[k + 1]
        b = x[k + m + 1] * w[k + 1]
        x[k + 1] = a + b
        x[k + m + 1] = a - b
    end
end
```

Without `ivdep`, LLVM generates scalar code for this pattern. With it, the loop vectorizes to
AVX-512 packed loads/FMAs. Always pair with `@inbounds`; bounds checks defeat vectorization.

## 2. Cache-blocked transpose (digit-reversal + twiddle)

**Effect: 28 → 30 GFLOP/s.** Fixed the profiled #1 bottleneck (the bit-reversal pass).

A naive bit-reversal pass over a large array is memory-bound with stride-N scatter writes. The fix:

- Process the input in **cache-line-sized tiles** (e.g. 8×8 blocks) so both source and destination
  fit in L1 during the copy.
- **Fold the digit-reversal into the base-butterfly's source offset**: instead of a separate
  reorder pass, the base butterfly reads from the bit-reversed address directly. This eliminates one
  full-array pass.

The lesson: the "obvious" structure (separate bit-reversal pass, then butterfly passes) is not
cache-optimal. Profiling found the reorder pass was the single largest cost. The fix is to keep
contiguous reads by blocking the transpose, not by eliminating it.

**Negative finding — pass fusion:** Fusing the bit-reversal into a gather/scatter inside the first
butterfly pass (to save a full-array scan) is slower. The fused loop has strided writes that destroy
SIMD vectorization and L1 locality. The wrapper passes are fastest **kept separate and contiguous**.

## 3. FMA for complex multiplication (`muladd`)

Julia's `Complex{T}` `*` operator does **not** automatically contract to FMA on its own — the
compiler follows IEEE rules by default and won't fuse `a*b + c` across the two real multiplies that
implement complex `*`. Write it explicitly:

```julia
# Slow: two independent multiplies, no FMA
c = a * b

# Fast: manual FMA-fused complex multiply
function cmul_fma(a::Complex{T}, b::Complex{T}) where {T}
    re = muladd(real(a), real(b), -imag(a) * imag(b))
    im = muladd(real(a), imag(b), imag(a) * real(b))
    Complex{T}(re, im)
end
```

`muladd` maps to a single FMA instruction on AVX2/AVX-512. This is the standard trick in
high-performance Julia numerical code. The Rust lang-compare benchmark used `mul_add` identically,
confirming the two produce bit-for-bit matching results.

## 4. `@generated` functions **must** be `@inline`

**Effect: 2.8× slower without `@inline` on `@generated` base codelets.**

`@generated` functions let you specialize on type-level constants (e.g. the butterfly size 4, 8,
16, 32 as a type parameter). But if the generated function is not inlined, Julia inserts a
dynamic dispatch to the generated method — even though the specialization is fully static. The
codelet call then has a function-call overhead equal to the work itself.

```julia
# Wrong: generated but not inlined — slow
@generated function butterfly!(x, ::Val{N}) where {N}
    return quote ... end
end

# Correct: always inline generated codelets
@inline @generated function butterfly!(x, ::Val{N}) where {N}
    return quote ... end
end
```

This applies to any `@generated` function whose call site is in a hot inner loop. Always `@inline`
them unless you want the specialization only for compile-time code generation, not for inlining.

## 5. Explicit SIMD beats autovectorization on uniform unit-stride loops; loses on irregular ones

**Win: ~1.2× on the Butterfly4 cross-pass (uniform stride, vectorization-friendly).**
**Loss: SIMD.jl was ~2× slower than autovectorization on the memory-bound radix-2 pass (and was
removed entirely).**

The rule is: explicit SIMD (SIMD.jl `Vec`, `vload`, `vstore`) wins only when:

1. The loop is **unit-stride** (or a known, small stride) and **uniform** (all iterations do the
   same thing, same register types, no branches).
2. The compiler's autovectorizer fails for a correctable reason (e.g. aliasing conservatism or
   missing `ivdep`).

On the Butterfly4 cross-pass — a tight loop over interleaved complex pairs with `_vcmul` packing
real/imag as a single `Vec{4, Float64}` — explicit SIMD won because the `_vcmul` trick (treating
a complex pair as a 4-wide float vector, shuffling for the FFTW-style twiddle multiply) cannot be
expressed as autovectorizable scalar code.

On the radix-2 stride-N pass — memory-bound with a non-unit twiddle stride — the manual SIMD
introduced gather loads that were slower than the scalar loop LLVM auto-vectorized. **Autovec is
often better on memory-bound paths; manual SIMD is for compute-bound paths with known good
register layout.**

## 6. Within-butterfly AVX: 4×4 decomposition with a register transpose

**Effect: 2.1× over the scalar Butterfly16 codelet.**

The key insight for a DFT-16 in registers: decompose it as a 4×4 matrix of DFT-4 sub-problems,
separated by twiddle multiplications. A naïve 4×4 DFT-4 has strided reads on either the row or
column DFT-4s, forcing gather/scatter and defeating SIMD.

The fix: after the row-wise DFT-4s, perform a **register-level 4×4 transpose** (using AVX shuffle
intrinsics) so that the column-wise DFT-4s also see contiguous data. Both DFT-4 stages are then
**shuffle-free** — they operate on contiguous packed floats with no gather, which is what AVX
wants.

```
Input 16 complexes (as 4 AVX registers of 4 complex pairs each)
  ↓ four DFT-4 sub-problems (row-wise, contiguous → AVX FMA)
  ↓ register transpose (4×4 shuffle, ~4 vpermpd/vperm2f128 instructions)
  ↓ twiddle multiplications
  ↓ four DFT-4 sub-problems (column-wise, now also contiguous → AVX FMA)
```

The register transpose costs ~4 shuffle instructions for the whole Butterfly16, which is a rounding
error against the ~64 FMA flops. The Butterfly32 is implemented as two Butterfly16 + a radix-2
combine stage.

## 7. SoA vs AoS: split re/im is shuffle-free but incurs split/merge overhead

**The `:soa` variant is ultimately slower than `:recursive` (AoS) end-to-end.**

A split-of-array (SoA) layout (`re[]`, `im[]` as separate arrays) makes the butterfly combine
shuffle-free: real/imag are adjacent in their own arrays, so SIMD loads are unit-stride. The
isolated SoA combine is **1.42×** faster than AoS.

But a full FFT needs an AoS↔SoA split at input, SoA↔AoS merge at output, and the extra cache
pressure from two separate streams (4 arrays instead of 2) at large N partially defeats L1/L2.
The split/merge passes cost more than the shuffle savings. SoA wins only if the input data is
already in SoA layout and you can avoid the conversion passes.

## 8. Zero-allocation via preallocated plans

Julia's GC is the enemy of performance in tight loops. PureFFT achieves **zero allocation on the
hot path** (verified with AllocCheck.jl `check_allocs`):

```julia
plan = plan_pfft(x; variant = :fast)  # allocates once here
for _ in 1:N
    pfft!(x, plan)  # zero allocation
end
```

The plan preallocates all twiddle tables and scratch buffers at construction. The `pfft!` apply
path carries no heap allocation, confirmed by:

```julia
using AllocCheck
@check_allocs pfft!(x, plan)  # errors if any allocation occurs
```

## 9. Static dispatch via `interface_trait` (TypeContracts)

PureFFT uses TypeContracts.jl's `interface_trait` to implement duck-typed `pfft!` with
**zero runtime overhead**:

```julia
interface_trait(::Type{<:SomePlan}) = HasPfft()

function pfft!(x, plan)
    pfft!(interface_trait(typeof(plan)), x, plan)
end
```

`interface_trait` folds to a concrete singleton type at compile time, so the dispatch is **fully
static** — no dynamic lookup, no allocation, no overhead. JET's `@report_opt` confirms the hot
path is dispatch-free.

## 10. Verification tooling

The correctness and performance invariants are checked by a suite of tools:

- **AllocCheck** (`check_allocs`): asserts zero heap allocation on the apply path.
- **TrimCheck.jl** (`@validate` + juliac `TRIM_SAFE`): ensures the hot path survives
  `--trim=unsafe` (no dynamic dispatch, no runtime inference, compatible with static compilation).
- **JET** (`@report_opt` / `@test_opt`): static analysis for type instabilities and unresolved
  dispatch in the transform pipeline.
- **Runic**: formatter, run with `runic --inplace <paths>` before committing.

All three tool checks pass on the `:fast` hot path.

## Summary table

| Trick | Effect | Notes |
|---|---|---|
| `@simd ivdep` cross-pass | 22 → 28 GFLOP/s | Asserts no loop-carried dep |
| Cache-blocked transpose | 28 → 30 GFLOP/s | Fixes #1 profiled cost |
| `muladd` FMA complex | ~10–15% | Julia's `Complex *` won't contract |
| `@inline @generated` codelets | 2.8× over non-inlined | Always inline generated fns |
| Explicit SIMD (AVX Butterfly4 cross-pass) | +1.2× | Wins on unit-stride loops |
| Explicit SIMD (radix-2 stride-N pass) | −2× | Autovec is better on memory-bound |
| Within-butterfly 4×4 + register transpose | 2.1× over scalar codelet | Shuffle-free both DFT-4 stages |
| SoA layout (full transform) | −15% vs AoS | Split/merge overhead cancels shuffles |
| Zero-allocation plans | no GC pauses | AllocCheck-verified |
| Static dispatch via `interface_trait` | zero overhead | JET-verified, `TRIM_SAFE` |
