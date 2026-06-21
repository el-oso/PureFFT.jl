# PureFFT.jl — agent guidelines

These are project-specific REQUIREMENTS for anyone (human or agent) working on PureFFT. They are
hard-won — violating them has caused 100×+ regressions and multi-session dead ends. Full detail and
measurements are in `docs/src/performance.md`; this file is the must-follow summary.

**Roadmap / planned work: see [`ROADMAP.md`](ROADMAP.md)** (checked-in, the canonical status + next steps).

## Performance requirements (MUST follow)

1. **Unroll fixed-count loops with `@generated`. NEVER index a tuple with a runtime variable.**
   `t[r]` / `t[2r+1]` where `r` is a loop variable is type-unstable and **boxes/allocates**
   (measured **135× slowdown**: a size-36 kernel went 2293ns → 18ns just from removing it). Emit
   straight-line code from a `@generated` function (the count comes from a type parameter), or unroll
   with **literal** indices. PureFFT already does this in `src/codelets.jl` (8 `@generated` codelets).
   - `@generated` is preferred over macro-based unrollers (e.g. Unroll.jl's `@unroll`) because our
     counts are *type parameters*, not macro-time literals. **Do not add Unroll.jl as a dependency**
     (it's inactive) — borrow the idea (emit straight-line code, unroll counter-dependent branches)
     via `@generated`.

2. **`@simd ivdep` + `@inbounds`** on dependency-free, contiguous numeric loops (twiddle multiplies,
   AoS↔SoA split/merge, copies). NOT on strided/scatter loops or loops with real cross-iteration deps.

3. **x86 SIMD intrinsics with no SIMD.jl wrapper → `Base.llvmcall`** to the exact LLVM intrinsic,
   module+entry form, converting `Vec ↔ v.data::NTuple{N,VecElement{T}}`. Example (verified bit-exact):
   `fmaddsub`/`fmsubadd` via `@llvm.x86.fma.vf{madd,msub}sub.pd.256`. See `src/avxradix/avxport.jl`.

4. **SIMD via SIMD.jl**: `Vec`, `shufflevector` (lane permutes — match the intrinsic's exact lane
   pattern when porting), `reinterpret` (bitwise ops like xor on floats), `muladd` (FMA). `vload`/
   `vstore` over `reinterpret(Ptr{T}, pointer(x))` for complex buffers.

5. **Keep the hot path dispatch-free**: concrete types, transform size / factors as *type parameters*
   so `@generated` specializes. Verify with JET `@test_opt`. Keep trim-safe (`Vector{Any}` only in
   `@generated` generators, never at runtime — verify with TrimCheck `@validate`).

6. **Benchmark tiny kernels correctly** (else the parity gate is unmeasurable): call via a `@noinline`
   concrete wrapper (`@noinline run!(w)=kernel!(w,CONST...)`), NOT a closure/lambda passed to a timer
   (closure indirection lands in the timed region). Use repeated **in-place** reps (no copy-subtract —
   data → NaN but FP throughput is identical, far more stable) + `taskset -c N` core pinning + a DCE
   sink. **Compare MEDIAN times, not min** (min rewards lucky outliers — it gave a false "0.93×" vs
   rust's unrepresentative min; on median the port is at/above parity). Report **σ** and require both
   distributions tight + comparable (rel-σ within a few %). Parity gate = rust_median/julia_median ≥ 0.96.

## Faithful-port methodology (for the non-pow2 / RustFFT parity work)

- **Duplicate RustFFT's algorithm EXACTLY** (op-for-op, same SIMD), do NOT reinterpret. Reinterpreting
  in our own style repeatedly plateaued at local maxima (~0.5–0.85× FFTW across many attempts — see
  `docs/src/performance.md` and the memory notes). A faithful mechanical port reaches parity
  (Butterfly36 = 0.92× rust). Source of truth: rustfft 6.4.1 (`/tmp/RustFFT`, = the crate).
- **Verify each layer BIT-EXACT against Rust golden values** (the harness in
  `bench/rustfft_compare/`, `cargo run --bin verify` → `golden.txt`; Julia diff in `port/`) before
  building the next layer up. Numerically expect ≤1e-15 rel-error (twiddle cos/sin may differ 1 ULP).
- **Match rust's RADIX CHOICE, not just the algorithm** (see `docs/src/performance.md` §15). The planner
  prefers **radix-8/9/12/6** (8ⁿ·9ᵐ·12ᵏ·6ʲ), NOT radix-4/5 — radix-8 reaches depth-2 parity (0.97×);
  radix-4/5 plateau ~0.91×. radix-9/12 are intrinsically ~3× more shuffle/FMA-heavy (twiddle mults) than
  radix-8, so 3-heavy sizes have a real **~0.85–0.92× floor** — not a bug. Use **ONE** size-n scratch
  buffer reused at every level (rust's in/out-of-place alternation; pass `scr` not `buf` as inner scratch),
  not depth×n (cache).
- **Know the measurement floor before micro-optimizing.** Even the in-process interleaved harness
  (`port/measure.jl`, rust via the cdylib) drifts **±7% on the ratio run-to-run** (σ≈2% within a run). Do
  NOT chase ≤5% per-radix opts against it — they're sub-noise (chunk-unroll and pre-dup twiddles both
  tested, both fail end-to-end). Pin CPU frequency first if you must. Isolated micro-benchmarks mislead —
  re-measure in the full kernel.
- **AVX-512 (Vec{8}) for non-pow2 = small, mostly-generic gain — not ~2×** (Phase-8, `docs/src/performance.md`
  §16, evidence `port/avx512_poc.jl`). Measured ~1.03–1.04× compute-bound / ~1.0× memory-bound vs Vec{4}
  (genuine zmm, not split): the kernels are shuffle/permute-bound and Zen5's 512-bit shuffle throughput
  doesn't double. `cb4`/`cb8` + arithmetic ARE width-generic (cheap), but a full Vec{8} FFT also needs
  width-specific W=8 transposes re-derived. Low payoff; pursue only for genericity/future-proofing.

## Standing rules

- `isnothing(x)` / `!isnothing(x)` — never `=== nothing` / `!== nothing`.
- **Regenerate `bench/plot_compare.jl` plots before every push** (catches regressions; benches need
  the CPU `performance` governor for stable numbers).
- Commit author email: `15278831+el-oso@users.noreply.github.com` (never a real address).
- End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- No Python anywhere (per the global rule).
