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

## 11. Runtime tuple indexing boxes — unroll with `@generated`

**Effect: 135× (a size-36 kernel: 2293 ns → 18.43 ns).**

Indexing a tuple with a **runtime** variable (`t[r]`, `t[2r+1]`, `ntuple(r -> ... t[r] ..., N)`) is
type-unstable — Julia can't prove the element type for a non-literal index, so it **boxes and
allocates**. This is the single most expensive mistake we hit. A faithful port of RustFFT's
`Butterfly36Avx64` written with `ntuple`/runtime-indexed loops ran at 2293 ns; rewriting it as
straight-line code with **literal** indices brought it to 18.43 ns (0.92× of rustfft's 16.94 ns).

Rust's `[T; N]` arrays with `for i in 0..N` const-range loops unroll cleanly; the direct Julia port
of such a loop must be **unrolled**, not transcribed as `for r in 0:N; t[r]; end`. The scalable way to
unroll (when the count is a *type parameter*) is a **`@generated` function** that emits the
straight-line body — this is exactly the genfft-style code generation in `src/codelets.jl`. (Macro
unrollers like Unroll.jl's `@unroll` only work when the bound is a macro-time literal/constant; prefer
`@generated` since our counts come from type parameters. Don't depend on Unroll.jl — it's inactive.)

## 12. x86 SIMD intrinsics with no SIMD.jl wrapper via `llvmcall`

SIMD.jl covers arithmetic, `shufflevector`, `reinterpret`, and `muladd` (FMA). For instructions it
doesn't expose — notably **`fmaddsub`/`fmsubadd`** (alternating subtract/add, the core of a SIMD
complex multiply) — call the exact LLVM intrinsic via `Base.llvmcall` in **module+entry form**,
converting `Vec ↔ v.data::NTuple{N,VecElement{T}}`:

```julia
const _IR = """
declare <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double>, <4 x double>, <4 x double>)
define <4 x double> @entry(<4 x double> %a, <4 x double> %b, <4 x double> %c) #0 {
  %r = call <4 x double> @llvm.x86.fma.vfmaddsub.pd.256(<4 x double> %a, <4 x double> %b, <4 x double> %c)
  ret <4 x double> %r
}
attributes #0 = { alwaysinline }"""
@inline fmaddsub(a, b, c) = Vec(Base.llvmcall((_IR, "entry"), NT4, Tuple{NT4,NT4,NT4}, a.data, b.data, c.data))
```

This produces a single `vfmaddsub231pd` and is **bit-exact** with Rust's `_mm256_fmaddsub_pd`. See
`src/avxradix/avxport.jl`. When porting an intrinsic-based kernel, match the **exact lane patterns** of each
`_mm256_permute/unpack/movedup_pd` with `shufflevector`, and verify bit-exact against a Rust golden.

## 13. Faithful porting beats reinterpretation (the local-maximum lesson)

The biggest strategic finding: **reinterpreting** an algorithm in our own style (SoA codelets,
custom four-step/recursive mixed-radix) repeatedly plateaued at *local maxima* — non-pow2 stalled at
~0.5–0.85× FFTW across many rounds (4096 cliff → 16384 cliff → orchestration-overhead frontier). A
**faithful, mechanical, op-for-op port** of RustFFT's actual AVX kernels (same algorithm, same SIMD,
no deviation) reaches parity: Butterfly36 = 0.92×. When matching a reference implementation's speed is
the goal, **duplicate it exactly and verify bit-exact at each layer** (Rust golden harness in
`bench/rustfft_compare/`) rather than re-deriving — re-derivation drifts into a slower local optimum.

## 14. Benchmarking tiny SIMD kernels (the parity-gate methodology)

Confirming a kernel is ≥0.96× of a Rust reference is dominated by *measurement* artifacts at sub-20ns:

- **Call via a `@noinline` concrete wrapper, not a closure.** `@b x (w->kernel!(w,consts))` (or passing
  a closure to a higher-order timer) puts the closure's call indirection inside the timed region — it
  reported a "0.82×" that was really ~0.93×. Use `@noinline run!(w)=kernel!(w,CONST1,CONST2)` (consts as
  `const` globals) and call `run!(w)` directly.
- **Use repeated in-place reps, not copy-subtract.** rustfft's harness times `copy+transform` and
  subtracts `copy`; subtracting two similar noisy numbers gives ±15% swings at n≈36. Looping the
  in-place transform with no copy (data → NaN, but FP *throughput* is identical) + a DCE sink is far
  more stable.
- **Pin a core (`taskset -c N`)**, warm up, and **compare MEDIAN times (not min)** with their sigmas.
  Min rewards a lucky outlier — comparing julia-median to rust-*min* gave a false "0.93×"; rust's σ
  showed its min was unrepresentative. On **median-to-median** the faithful port is at/above parity,
  and Julia's σ (0.2–0.4%) is *tighter* than rust's (0.3–3.7%). Gate: `rust_median/julia_median ≥ 0.96`
  with both distributions tight + comparable.

Measured (median, core-pinned): **Butterfly7 1.04×, Butterfly36 1.05×, MixedRadix4xn-144 0.98×** —
all ≥0.96×. (The docs comparison plots in `bench/plot_compare.jl` likewise use median, not min.)

## 15. Radix choice dominates; per-radix micro-opt hits a measurement floor (the non-pow2 push)

Hard-won lessons from pushing the faithful port's non-power-of-two coverage to ≥0.96×:

- **Match RustFFT's *radix choice*, not just its algorithm.** The avx_planner prefers **radix-8/9/12/6**
  ("blazing fast 8xn"), decomposing 2·3-smooth sizes as 8ⁿ·9ᵐ·12ᵏ·6ʲ — *not* radix-4/5. With radix-8,
  composites reach parity **even at depth 2** (2304 = MR8(MR8(B36)) = 0.97×). Building trees from radix-4/5
  instead plateaus at ~0.91× — that "depth-2 plateau" was a wrong-radix artifact, not a language limit.
- **radix-8 is intrinsically cheap; radix-9/12 are not.** A size-8 column butterfly is adds + rotations
  with **no twiddle multiplies** (21 shuffles / 7 FMAs per pass); the size-9 (3×3) needs complex twiddle
  mults (**36 shuffles / 24 FMAs** — ~3×). So radix-9/12-heavy (3-heavy) sizes are intrinsically ~3× more
  shuffle/FMA-heavy than radix-8 and sit at a **~0.85–0.92×** floor *vs rustfft*; radix-8-dominated sizes
  are at parity. **Update:** that vs-rustfft gap is NOT a Julia codegen/scheduling limit — the
  `julia-sched-mwe/` reproducer shows a matched radix-9 butterfly *and* full step compile identically and
  run ≥ Rust in Julia. The gap is rustfft's *implementation* being more optimized than PureFFT's, i.e.
  recoverable by matching its algorithm (decomposition / in-place / transpose), not a fundamental Julia floor.
- **One scratch buffer, not per-level.** RustFFT's in-place/out-of-place alternation reuses a *single*
  size-n scratch at every recursion level (pass `scr`, not `buf`, as the inner's scratch). Allocating a
  distinct buffer per level (depth×n) blows the working set out of cache and makes the per-level gap
  *compound* with depth (576: 0.82× → 0.97× after the fix on shallow sizes).
- **Micro-opts that failed (documented so nobody re-tries them):** (a) **chunk-loop unrolling** backfires
  on already-large kernel bodies — register pressure, 0.96×. (b) **Pre-duplicated twiddles**
  (store `dup_re`/`dup_im`, drop 2 shuffles/mult) is +5% in an *isolated* colbf micro-bench but
  **net-neutral/negative end-to-end** — it trades a shuffle for a load and doubles the twiddle array
  (cache). Isolated micro-benchmarks mislead; always re-measure in the full kernel.
- **The measurement floor is ~7% on the *ratio*, run-to-run.** Even the in-process *interleaved* harness
  (rust via ccall to a cdylib, alternating blocks same-process, median+σ, `taskset`) has σ≈2% *within* a
  run but the absolute ratio drifts ±7% *between* process launches (144× measured at 0.90 / 0.945 / 0.97
  on identical code). **You cannot validate a ≤5% optimization against a 7% floor** — sub-noise per-radix
  tuning is not productive without first pinning CPU frequency (`cpupower`, needs root) or ~10× longer
  averaging. Know the floor before chasing small wins.
- **Base coverage gaps oscillate the plot — add the base RustFFT uses, don't reinterpret.** The avx W=4
  planner originally had only base `B36 = 2²·3²`, so `2^odd·3²·5^c` sizes (90 = 2·3²·5, 360 = 2³·3²·5)
  had no valid tree (their `B36` leftover needs an unsupported radix-2) and fell to slow fallbacks —
  90 ≈ 0.57×, 360 ≈ 0.51×, while the neighbours 180/720 (which *do* fit `B36`) sat at parity. That
  size-to-size base availability *is* the non-pow2 oscillation, not autotuner noise (min vs median rank
  the candidates identically — measured). Fix: port RustFFT's **`Butterfly18Avx64`** op-for-op as a new
  base `B18 = 2·3²` (3×6 dual-width: `column_butterfly6` → twiddle → `transpose_3x6_to_6x3` →
  `column_butterfly3` → packed 6×3 store). Then 90 = `B18·5`, 360 = `B18·4·5`; both reach the fast path
  (90 → ~1.8×, 360 → ~1.0× of FFTW), bit-exact (n=18 rel 2e-16, 409-size sweep clean). Note `B9` (our
  packed 3×3) is **not** a drop-in base — only standard-layout butterflies (`B36`/`B18`) compose under
  MR-wrapping; reusing `B9` corrupted 8 sizes. The lesson: extend coverage by faithfully porting rust's
  base for the gap, never by reinterpreting an existing kernel into a role it wasn't built for.

## 16. AVX-512 (Vec{8}) for non-pow2: a small, mostly-generic gain — not the ~2× one might hope (Phase 8)

RustFFT is AVX2-only (256-bit, 2 complex/vector), so a natural idea is to run the non-pow2 mixed-radix at
**Vec{8,Float64}** (512-bit, 4 complex/vector) to *exceed* it on Zen5. Measured, not assumed
(`port/avx512_poc.jl`):

- A width-doubled column butterfly (`cb8` at Vec{8}) is **bit-correct** and emits **genuine AVX-512**
  (`@code_native`: 159 zmm refs, 0 ymm — not LLVM-split into 2× 256-bit).
- The column-butterfly pass gives a **real but small ~1.03–1.04× per-complex** vs Vec{4} when compute-bound
  (L1-resident; consistent across runs), and **~1.0×** when memory-bound (large arrays — bandwidth-limited).
- **Why so small:** the kernels are **shuffle/permute-bound** (rotate90 swaps + the twiddle
  `mul_complex`'s dup_re/dup_im/swap — ~3 shuffles per twiddle multiply), and **Zen5's 512-bit shuffle
  throughput doesn't double** over 256-bit. Wider vectors only fully speed up FMA/add-bound code — which is
  why the *pow2* `radix4_avx.jl` path (FMA-heavy, register-transpose minimizes shuffles) *does* use Vec{8}.
- **It is, however, mostly width-generic.** `cb4`/`cb8` and the arithmetic primitives work at both widths
  with only width-dispatched shuffle patterns + constants (no rewrite) — so AVX-512 support for the
  butterfly/compute layer is cheap. The **transposes are width-specific** (512-bit lane patterns differ;
  `unpacklo/hi` don't compose across 128-bit lanes), so a *full* Vec{8} FFT still needs W=8 transpose
  variants re-derived + partial-column handling.

**Conclusion (PoC, cb8 alone):** width-doubling a single column-butterfly pass is mostly-generic but the
per-pass payoff is low-single-digit % (diluted by the shuffle bottleneck). The bigger lever would be an
AVX-512-*native* redesign cutting shuffle count (e.g. `vpermt2pd` doing in one op what several 256-bit
shuffles do) — a research effort, not a port.

**Update — the full W=8 path does better than the cb8-alone PoC suggested.** Built as the faithful W=8
tree (`src/avxradix/width8.jl`: `B64W8`/`MR8W8`/`MR9W8`/`MR12W8` + W=8 transposes), same-tree W=8 beats
W=4 **1.03–1.11×** and is at/above RustFFT parity (**0.96–1.07×**) across L1→L3 (768/9216/110592) — i.e.
**also on the memory-bound size** (110592 = 1.11×), contradicting the PoC's "memory-bound ⇒ no gain". Two
reasons the full path beats the cb8 PoC: (1) the *radix-12/9* column butterflies gain more at W=8 (≈1.20–
1.25× — 3-heavy, more FMA per shuffle) than cb8 (≈1.03×); (2) an early version regressed badly at large n
(9216 ≈ 0.72× W4) — that was **not** memory bandwidth but **runtime tuple indexing** in the store loops
(`for k in 1:N; t[k]`, the CLAUDE.md rule-#1 trap); unrolling with `@nexprs` (literal indices) fixed it.
So the net payoff is a real, consistent few-percent over W=4 across the size range — and W=8 is the routed
default for the sizes it covers (`autoplan` times it). Coverage now spans **2·3·5-smooth** (W=8
`transpose5`/`transpose9` derived as one+ `transpose4` block + leftover rows + bridging shuffles): autoplan
routes 5-smooth sizes (2880/23040/46080) to W=8 too, each beating FFTW and approaching rust (0.88–1.00×).
radix-5/9 stay just under rust (the intrinsic shuffle-bound floor for radices that aren't a multiple of
CPV=4); only the `vpermt2pd`-native redesign would close that.

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
| Unroll via `@generated` (no runtime tuple index) | **135×** | runtime `t[r]` boxes; literal/`@generated` unroll |
| `fmaddsub`/`fmsubadd` via `llvmcall` | bit-exact w/ rust | x86 intrinsics SIMD.jl lacks |
| Faithful RustFFT port vs reinterpretation | 0.5–0.85× → **0.92×** | mechanical op-for-op port reaches parity |
| Match rust's radix (8/9/12/6 not 4/5) | plateau 0.91× → **0.97×** | radix-8 depth-2 parity; single scratch buffer |
| radix-9/12 vs radix-8 cost | ~0.90× floor | 3× shuffles/FMAs (twiddle mults); fundamental |
| Per-radix micro-opt below measurement floor | ≤5% vs **7%** noise | ratio drifts ±7% run-to-run; pin freq to chase |
