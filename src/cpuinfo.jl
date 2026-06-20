# CPU-generic cache parameters.
#
# Cache sizes come from CPUSummary as compile-time `StaticInt`s — detected at precompile via
# CpuId, or pinned through CPUSummary's Preferences keys ("cs"/"ci"/"nc"/"syst") for a
# reproducible, trim-compatible build (so no CpuId runtime ccall is needed). Everything below
# folds to a constant, keeping the tuned kernels branch-free and juliac-trimmable.
#
# This only generalizes the cache-blocking constants that were hand-tuned for Zen5; the SIMD
# kernels already target 512-bit AVX-512 registers (`Vec{8,Float64}` / `Vec{16,Float32}`).

using CPUSummary: cache_size

# Per-core data-cache sizes in bytes. Fallbacks keep this sane if a level reports 0.
const _L1_BYTES = let s = Int(cache_size(Val(1)))
    s > 0 ? s : 32 * 1024
end
const _L2_BYTES = let s = Int(cache_size(Val(2)))
    s > 0 ? s : 16 * _L1_BYTES
end

# A working tile of ~1/3 of L1, in complex elements (leaves room for a second array + twiddles).
# Drives the transpose blocking, the vectorized-transpose cutoff, and the four-step tile.
# = 1024 on Zen5 (L1 = 48 KiB).
const _L1_TILE = _L1_BYTES ÷ 16 ÷ 3

# Side length of the square cache-blocked transpose tile (four-step). = 32 on Zen5.
const _BTRANSPOSE_BLK = isqrt(_L1_TILE)

# Radix-16 fusion stays cache-local while the fused 16-stream block fits ~1/8 of L2 (in complex
# elements). = 8192 on Zen5 (L2 = 1 MiB).
const _L2_FUSE = _L2_BYTES ÷ 16 ÷ 8
