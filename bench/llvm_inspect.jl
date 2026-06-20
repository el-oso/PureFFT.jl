# Confirms (or refutes) that the Base `@simd` kernel actually vectorizes, and inspects the
# native code of the hot butterfly loop. The question "is it LLVM or Rust?" is partly
# answered here: Julia and Rust share the LLVM backend, so if Julia's hot loop emits packed
# FMA/vector ops, any remaining gap to rustfft is algorithm/memory, not codegen.
#
# Run:  julia --project=bench bench/llvm_inspect.jl

import PureFFT
using InteractiveUtils

const T = Float64
x = randn(Complex{T}, 4096)
stages = PureFFT.staged_twiddles(Complex{T}, length(x))

# Capture native code of the Base @simd staged kernel.
buf = IOBuffer()
code_native(buf, PureFFT.radix2_base_simd!, (Vector{Complex{T}}, typeof(stages)); syntax = :intel, debuginfo = :none)
asm = String(take!(buf))

# Count vector vs scalar FP instructions as a crude vectorization signal.
vec_fma = count(m -> true, eachmatch(r"vf(m|n)(add|sub|msub|madd)\w*\s+[zy]mm", asm))
vec_mul = count(m -> true, eachmatch(r"vmulpd\s+[zy]mm", asm))
vec_add = count(m -> true, eachmatch(r"vaddpd\s+[zy]mm", asm))
scal_mul = count(m -> true, eachmatch(r"vmulsd", asm))
ymm = count(m -> true, eachmatch(r"\bymm\d", asm))
zmm = count(m -> true, eachmatch(r"\bzmm\d", asm))

println("Base @simd radix-2 kernel — native code signals (Float64, $(Sys.CPU_NAME)):")
println("  packed FMA (vfmadd/sub … y/zmm) : ", vec_fma)
println("  packed mul  (vmulpd y/zmm)      : ", vec_mul)
println("  packed add  (vaddpd y/zmm)      : ", vec_add)
println("  scalar mul  (vmulsd)            : ", scal_mul)
println("  ymm register mentions          : ", ymm)
println("  zmm register mentions          : ", zmm)
println()
println(
    vec_fma + vec_mul + vec_add > 0 ?
        "→ Base @simd DID autovectorize the complex butterflies (packed SIMD present)." :
        "→ No packed SIMD detected; the loop stayed scalar."
)
