# Dogfooding StrictMode.jl (https://github.com/el-oso/StrictMode.jl) — it turns AllocCheck + JET +
# @inferred into *declarable* performance guarantees, unifying the ad-hoc `@test_opt` (dispatch-free)
# and AllocCheck `check_allocs` (alloc-free) checks PureFFT already relies on. Here we assert PureFFT's
# hot path holds those guarantees through StrictMode's API, and audit the whole compiled surface.
#
# Checks are gated by a compile-time Preference (test/LocalPreferences.toml ships them enabled). When
# disabled the macros are zero-cost no-ops, so we skip the assertions rather than pass them vacuously.

@testitem "StrictMode dogfood: PureFFT declares its perf guarantees" begin
    # StrictMode's analysis backend (AllocCheck + JET) is a weak dependency — both must be loaded for the
    # :full checks to run (else StrictMode errors with a clear message). They're test-env deps.
    using StrictMode, AllocCheck, JET

    if !StrictMode.checks_enabled()
        @info "StrictMode checks disabled in this env — skipping dogfood (enable_checks! + restart to run)"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureFFT
        # One representative plan per routing path — mirrors the existing `@test_opt` hot-path testset.
        plans = Any[
            (P.Radix4AvxPlan(ComplexF64, 1024), 1024),   # power-of-two AVX-512 radix-4
            (P.CodeletPlan(ComplexF64, 12), 12),         # small non-pow2 generated codelet
            (P.autoplan(ComplexF64, 768), 768),          # non-pow2, AVX-512 (W=8) radix-12 tree
            (P.autoplan(ComplexF64, 576), 576),          # non-pow2, AVX-512 (W=8) radix-9 tree
            (P.autoplan(ComplexF64, 2880), 2880),        # non-pow2, AVX-512 (W=8) radix-5 tree (5-smooth)
            (P.autoplan(ComplexF64, 1080), 1080),        # non-pow2, AVX2 (W=4) faithful tree
            (P.BluesteinPlan(ComplexF64, 97), 97),       # chirp-Z
            (P.RaderPlan(ComplexF64, 769), 769),         # prime via cyclic convolution
        ]
        # @assert_typestable / @assert_noalloc / @assert_trim_safe throw StrictViolation on failure, so
        # reaching the @test means the declared guarantee held for that kernel (qualified name — see the
        # StrictMode F2 note). :trimsafe (TypeContracts, no backend) is a cheap juliac --trim=safe-style
        # scan; the authoritative deep TrimCheck @validate lives in the "TrimCheck trim-safety" testitem.
        # (The whole-module `audit`/`check_compiled` surface scan is slower; it lives in
        # bench/strictmode_audit.jl, mirroring how AllocCheck's static scan lives in bench/alloccheck.jl.)
        for (p, n) in plans
            x = randn(ComplexF64, n)
            @assert_typestable P.apply_unnormalized!(p, x)
            @assert_noalloc P.apply_unnormalized!(p, x)
            @assert_trim_safe P.apply_unnormalized!(p, x)
            @test true
        end
    end
end
