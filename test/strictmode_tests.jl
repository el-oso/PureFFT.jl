# Dogfooding StrictMode.jl (https://github.com/el-oso/StrictMode.jl) — it turns AllocCheck + JET +
# @inferred into *declarable* performance guarantees, unifying the ad-hoc `@test_opt` (dispatch-free)
# and AllocCheck `check_allocs` (alloc-free) checks PureFFT already relies on. Here we assert PureFFT's
# hot path holds those guarantees through StrictMode's API, and audit the whole compiled surface.
#
# Checks are gated by a compile-time Preference (test/LocalPreferences.toml ships them enabled). When
# disabled the macros are zero-cost no-ops: locally we skip rather than pass vacuously, and under CI
# assert_enabled() errors outright — a green CI run with checks off proves nothing.

@testitem "StrictMode dogfood: PureFFT declares its perf guarantees" begin
    # StrictMode's analysis backend (AllocCheck + JET) is a weak dependency — both must be loaded for the
    # :full checks to run (else StrictMode errors with a clear message). They're test-env deps.
    using StrictMode, AllocCheck, JET

    if !StrictMode.assert_enabled()   # errors under CI when checks are disabled
        @info "StrictMode checks disabled in this env — skipping dogfood (enable_checks! + restart to run)"
        @test_skip StrictMode.checks_enabled()
    else
        P = PureFFT
        # One representative plan per routing path — mirrors the existing `@test_opt` hot-path testset.
        plans = Any[
            (P.Radix4AvxPlan(ComplexF64, 1024), 1024),   # power-of-two AVX-512 radix-4
            (P.CodeletPlan(ComplexF64, 12), 12),         # small non-pow2 generated codelet
            (P.GenPPCodeletPlan(ComplexF64, 289), 289),  # generated column-packed P² codelet (17²)
            (P.GenPPCompositePlan(ComplexF64, 578, 17, 2), 578),  # radix-M DIT over gen_pp (2·17²)
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
        # @assert_inlined — guard that the hot-LOOP generated kernels stay @inline. A missing @inline on
        # avx_colbf_prime regressed the MR5/7/13 passes to ~0.7× (2026-06-30; op-counts were IDENTICAL,
        # so only the per-iteration non-inlined-call overhead differed) — this catches that class
        # statically, before any benchmark. StrictMode treats :inlined as informational (inlining is a
        # heuristic), but for these large @generated codelets Julia's and LLVM's decisions align, and the
        # N=13 case is where a non-@inline @generated kernel actually fails to inline into the pass loop.
        let v4 = P.AvxRadix.V4f(ntuple(_ -> 1.0, 4))
            @assert_inlined P.AvxRadix.avx_colbf_prime(ntuple(_ -> v4, 13), ntuple(_ -> v4, 6))  # radix-13 prime butterfly
            @assert_inlined P.AvxRadix.gen_transpose_packed(ntuple(_ -> v4, 13))                  # packed transpose, large N
            @test true
        end
    end
end

# Coverage gate: every exported/public function must either register its guarantees below or be
# exempted VISIBLY. A new public function makes this testitem fail until it declares itself —
# that's the point. The registration list is the manifest; check_all() enforces what it promises.
@testitem "StrictMode coverage: public surface declares its guarantees" begin
    using StrictMode, AllocCheck, JET

    if !StrictMode.assert_enabled()   # errors under CI when checks are disabled
        @test_skip StrictMode.checks_enabled()
    else
        P = PureFFT
        empty!(StrictMode.registered_strict())
        empty!(StrictMode.exempt_strict())

        p = P.autoplan(ComplexF64, 64)

        # Hot static path: the full kernel guarantees, enforced (probed 2026-07-02: these hold).
        StrictMode.register_strict!(
            P.pfft!, (Vector{ComplexF64}, typeof(p));
            guarantees = (:typestable, :noalloc, :trimsafe)
        )
        # alloc_scratch ships with the scratch-decouple branch; register it when present so this
        # item is green on master and on that branch alike.
        if isdefined(P, :alloc_scratch)
            prec = plan_pfft(ComplexF64, 64; variant = :recursive)
            StrictMode.register_strict!(P.alloc_scratch, (typeof(prec),); guarantees = (:typestable,))
        end

        # The one-shot / planning convenience API selects a plan at RUNTIME by design — JET-full
        # rightly flags the internal dynamic dispatch (r2r/plan_r2r even infer abstract returns),
        # and it is cold: called once, never in a loop. Exempted VISIBLY here; all hot work lands
        # in pfft!/apply_unnormalized!, which carry the full guarantees (see the dogfood item).
        # Removing a name from this tuple = promising StrictMode it holds (:typestable,) — the
        # gate then enforces that promise.
        COLD_API = (
            :pfft, :ipfft, :ipfft!, :plan_pfft, :prfft, :pirfft, :plan_prfft, :plan_pirfft,
            :r2r, :r2r!, :plan_r2r, :dct, :dct!, :idct, :idct!, :plan_dct, :plan_idct,
            :tryr2r, :tryplan_r2r,
        )

        fs = check_all()                                   # the declared guarantees hold…
        @test nfailures(fs) == 0
        # …and nothing public is undeclared (new exports fail here until they register or exempt).
        cov = audit(P; require = :public, exempt = COLD_API, io = devnull)
        for f in cov
            f.guarantee === :coverage && @info "uncovered public function" f.func f.suggestion
        end
        @test nfailures(cov) == 0

        empty!(StrictMode.registered_strict())
        empty!(StrictMode.exempt_strict())
    end
end
