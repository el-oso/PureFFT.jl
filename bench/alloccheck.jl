# Proves the transform hot path allocates nothing. Two independent checks:
#   1. AllocCheck.check_allocs — static guarantee that the kernel has no allocation sites.
#   2. runtime @allocated after warm-up — empirical confirmation through the public API.
# Plan construction (twiddle tables etc.) is allowed to allocate — it runs once, not per
# transform. Only `pfft!` on a prebuilt plan is the hot path.
#
# Run:  julia --project=bench bench/alloccheck.jl

import PureFFT
using AllocCheck

const T = ComplexF64
const N = 4096

println("Static check (AllocCheck.check_allocs) on transform kernels:")
for (name, f, argtypes) in (
        (
            "recursive_fft!", PureFFT.recursive_fft!,
            (Vector{T}, Vector{T}, Vector{Vector{T}}, Bool),
        ),
        ("radix2_staged!", PureFFT.radix2_staged!, (Vector{T}, Vector{Vector{T}})),
        ("radix2_base_simd!", PureFFT.radix2_base_simd!, (Vector{T}, Vector{Vector{T}})),
        ("radix2_dit!", PureFFT.radix2_dit!, (Vector{T}, Vector{T})),
    )
    allocs = check_allocs(f, argtypes)
    status = isempty(allocs) ? "OK  no allocations" : "XX  $(length(allocs)) allocation site(s)"
    println("  ", rpad(name, 20), status)
end

println("\nRuntime @allocated through pfft!(x, plan) after warm-up (bytes):")
for v in (:scalar, :staged, :base, :recursive, :soa, :fourstep, :radix4, :radix4avx, :fast)
    x = randn(T, N)
    p = PureFFT.plan_pfft(x; variant = v)
    PureFFT.pfft!(copy(x), p)                     # warm-up / compile
    y = copy(x)
    b = @allocated PureFFT.pfft!(y, p)
    println("  ", rpad(string(v), 12), b, b == 0 ? "  ✔" : "  ✗")
end
