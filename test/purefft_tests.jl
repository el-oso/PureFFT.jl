# PureFFT test suite, organized as ReTestItems `@testitem`s (independent, parallel-runnable).
# Each `@testitem` auto-imports `PureFFT` and `Test`; other packages are `using`-ed per item.
# Shared helpers live in the `FFTUtil` `@testsetup` module (relerr/tol), pulled in via `setup`.

@testsetup module FFTUtil
    export relerr, tol
    # Relative L2 error between a candidate transform and the reference.
    relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))
    # Per-precision tolerance: near machine epsilon scaled by the transform depth.
    tol(::Type{Float64}) = 1.0e-11
    tol(::Type{Float32}) = 1.0e-4
end

@testitem "Stage 1: radix-2 baseline" setup = [FFTUtil] begin
    using FFTW
    @testset "forward vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 64, 256, 1024, 4096)

        x = randn(Complex{T}, n)
        ref = fft(x)
        @test relerr(PureFFT.radix2_rec(x, false), ref) < tol(T)
        y = copy(x)
        pfft!(y, plan_pfft(y; variant = :scalar))
        @test relerr(y, ref) < tol(T)
        x0 = copy(x)
        @test relerr(pfft(x), ref) < tol(T)
        @test x == x0
    end

    @testset "inverse round-trips ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 8, 64, 1024)

        x = randn(Complex{T}, n)
        @test relerr(ipfft(pfft(x)), x) < tol(T)
        @test relerr(ipfft(x), ifft(x)) < tol(T)
    end

    @testset "errors on non-power-of-two" begin
        @test_throws ArgumentError plan_pfft(Complex{Float64}, 6; variant = :scalar)  # :scalar is pow2-only
        @test plan_pfft(Complex{Float64}, 6) isa PureFFT.AbstractFFTPlan              # but the default :fast handles any n
    end
end

@testitem "Stage 2: mixed-radix" setup = [FFTUtil] begin
    using FFTW
    @testset "factorize" begin
        @test PureFFT.factorize(1) == Int[]
        @test PureFFT.factorize(8) == [4, 2]
        @test PureFFT.factorize(16) == [4, 4]
        @test prod(PureFFT.factorize(1000)) == 1000
        @test PureFFT.factorize(7) == [7]
        @test prod(PureFFT.factorize(210)) == 210
    end

    @testset "forward vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 3, 5, 6, 7, 12, 15, 36, 100, 210, 1000, 1024, 2048)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :mixedradix), fft(x)) < tol(T)
    end

    @testset "inverse round-trip ($T, n=$n)" for T in (Float64, Float32),
            n in (6, 7, 100, 210, 1000)

        x = randn(Complex{T}, n)
        @test relerr(ipfft(pfft(x; variant = :mixedradix); variant = :mixedradix), x) < tol(T)
    end
end

@testitem "Stage 3: staged radix-2" setup = [FFTUtil] begin
    using FFTW
    @testset "$variant vs FFTW ($T, n=$n)" for variant in (:staged, :base),
            T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant); variant), x) < tol(T)
    end
end

@testitem "Stage 4: cache-oblivious recursive" setup = [FFTUtil] begin
    using FFTW
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :recursive), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :recursive); variant = :recursive), x) < tol(T)
    end
end

@testitem "Stage 5: SoA recursive" setup = [FFTUtil] begin
    using FFTW
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :soa), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :soa); variant = :soa), x) < tol(T)
    end
end

@testitem "Stage 7: cache-blocked four-step" setup = [FFTUtil] begin
    using FFTW
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 32768, 65536, 262144)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :fourstep), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :fourstep); variant = :fourstep), x) < tol(T)
    end
end

@testitem "Radix4 variants" setup = [FFTUtil] begin
    using FFTW
    @testset "$variant vs FFTW ($T, n=$n)" for variant in (:radix4, :radix4simd, :radix4avx),
            T in (Float64, Float32),
            n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant); variant), x) < tol(T)
    end
end

@testitem "Stage 8: Bluestein (chirp-Z, arbitrary n)" setup = [FFTUtil] begin
    using FFTW
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 3, 5, 7, 11, 13, 17, 64, 91, 100, 181, 256, 362, 1000, 5793)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :bluestein), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :bluestein); variant = :bluestein), x) < tol(T)
    end
end

@testitem "Stage 9: dynamic mixed-radix codelet" setup = [FFTUtil] begin
    using FFTW, JET
    # Generated straight-line codelet for ANY size: primes, prime powers, composites, pow2.
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 16, 25, 27, 45, 49, 64, 81, 100, 121)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :codelet), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :codelet); variant = :codelet), x) < tol(T)
    end

    @testset ":fast routes small smooth + small primes to a fast (non-Bluestein) plan; only large primes → Bluestein" begin
        # `:fast` (autoplan) now TIMES candidates and keeps the fastest, so the exact kernel for a small
        # smooth size is size-dependent (e.g. n=96 → FourStepCodeletPlan beats the size-96 CodeletPlan).
        # The invariant is the routing *intent*: smooth sizes reach a fast codelet/smooth path, never the
        # high-overhead Bluestein. (Correctness for these sizes is covered by the "vs FFTW" testset above.)
        for n in (6, 9, 12, 27, 48, 96)
            p = plan_pfft(ComplexF64, n; variant = :fast)
            @test p isa PureFFT.CodeletPlan || p isa PureFFT.FourStepCodeletPlan ||
                  p isa PureFFT.RecursiveMixedRadixPlan || p isa PureFFT.AvxMixedRadixPlan
            @test !(p isa PureFFT.BluesteinPlan)
        end
        # Small odd primes/prime-powers/composites (≤43) now route through the AVX odd-prime radix tree
        # (BP leaf), NOT Bluestein — running a chirp-Z convolution for a size-7 FFT was the mis-route.
        for n in (7, 11, 13, 25, 49, 33, 65)
            @test !(plan_pfft(ComplexF64, n; variant = :fast) isa PureFFT.BluesteinPlan)
        end
        @test plan_pfft(ComplexF64, 1009; variant = :fast) isa PureFFT.BluesteinPlan  # large prime, no split
    end

    @testset "codelet hot path is dispatch-free" begin
        x = randn(ComplexF64, 12)
        p = plan_pfft(x; variant = :codelet)
        @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(p, x)
    end

    @testset "autoplan boundary: runtime kernel selection ⇒ a (non-concrete) Union return" begin
        # `autoplan` picks the kernel by runtime timing, so its inferred return cannot be a single
        # concrete type. The tuple-based timing (vs the old `AbstractFFTPlan[]` vector, see StrictMode F34)
        # narrows it from the bare supertype to a *Union* of the candidate plan types — strictly more
        # precise, but still past Julia's union-split limit, so a caller that uses the returned plan still
        # pays one dispatch per `apply` (the documented plan-constructor exception to "concrete returns").
        # The dispatch-free/alloc-free guarantees are asserted on the *concrete* plan types elsewhere; this
        # pins the boundary so a change in breadth (toward concrete, or back to bare-abstract) is visible.
        rt = Base.return_types(PureFFT.autoplan, (Type{ComplexF64}, Int))[1]
        @test !isconcretetype(rt)              # runtime selection ⇒ not one concrete type (yet)
        # Bounded under the (unparameterized) plan supertype. NB: not `AbstractFFTPlan{Float64}` — the
        # pow2 branch's `AutoPlan{T, typeof(best)}` widens its T param (best is a Union), so the union
        # contains bare `AutoPlan`, which is `<: AbstractFFTPlan` but not `<: AbstractFFTPlan{Float64}`.
        @test rt <: PureFFT.AbstractFFTPlan
        @test PureFFT.plan_pfft(ComplexF64, 1080; variant = :fast) isa PureFFT.AbstractFFTPlan{Float64}
    end
end

@testitem "Stage 10: four-step batched-codelet executor" setup = [FFTUtil] begin
    using FFTW, JET
    # Smooth composite non-pow2 sizes (>128) route here; batched SoA codelets for both passes.
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (144, 180, 720, 900, 1000, 1080, 2520)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :fast), fft(x)) < tol(T)
        pf = plan_pfft(x; variant = :fast); y = copy(x); pfft!(y, pf)
        pii = plan_pfft(Complex{T}, n; variant = :fast, inverse = true); pfft!(y, pii)  # normalized
        @test relerr(y, x) < tol(T)
    end

    @testset "routes smooth composite >128 → four-step / recursive / faithful rust-port plan" begin
        # autotuner picks the fastest of FourStepCodeletPlan (2-factor) / RecursiveMixedRadixPlan /
        # AvxMixedRadixPlan (the faithful rust-port tree, used when its 2·3·5-smooth tree wins the timing).
        for n in (144, 900, 1000, 2520)
            p = plan_pfft(ComplexF64, n; variant = :fast)
            @test p isa PureFFT.FourStepCodeletPlan || p isa PureFFT.RecursiveMixedRadixPlan || p isa PureFFT.AvxMixedRadixPlan
        end
        # large smooth composite (>16384, no valid four-step split) → recursive or rust-port (no Bluestein cliff)
        for n in (23040, 46080)
            p = plan_pfft(ComplexF64, n; variant = :fast)
            @test p isa PureFFT.RecursiveMixedRadixPlan || p isa PureFFT.AvxMixedRadixPlan
        end
    end

    @testset "faithful rust-port plan (AvxMixedRadixPlan)" begin
        @test isnothing(PureFFT.AvxMixedRadixPlan(ComplexF64, 97))     # prime → outside coverage
        @test isnothing(PureFFT.AvxMixedRadixPlan(ComplexF32, 1080))   # Float64-only port
        for n in (720, 1080, 1440, 11520)
            p = PureFFT.AvxMixedRadixPlan(ComplexF64, n)
            @test p isa PureFFT.AvxMixedRadixPlan
            x = randn(ComplexF64, n)
            y = copy(x); pfft!(y, p)                                # forward (unnormalized) = fft
            @test relerr(y, fft(x)) < 1e-10
            pii = PureFFT.AvxMixedRadixPlan(ComplexF64, n; inverse = true)
            pfft!(y, pii)                                           # normalized inverse → back to x
            @test relerr(y, x) < 1e-10
        end
        x = randn(ComplexF64, 1080)                                 # hot path dispatch-free
        @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(PureFFT.AvxMixedRadixPlan(ComplexF64, 1080), x)
    end

    @testset "AVX-512 (W=8) path (AvxMixedRadixPlanW8)" begin
        @test isnothing(PureFFT.AvxMixedRadixPlanW8(ComplexF64, 1080))  # 2^3·3^3·5: v2=3, below the small-base v2≥4 floor
        @test isnothing(PureFFT.AvxMixedRadixPlanW8(ComplexF64, 56))    # 2^3·7: factor 7 has no W=8 radix pass
        @test isnothing(PureFFT.AvxMixedRadixPlanW8(ComplexF64, 48))    # small-base W=8 path is Float32-ONLY (F64 uses the tuned W=4 tree)
        # Float32 W=8 = Vec{8,Float32} = 256-bit AVX2 (NOT AVX-512) — the PRIMARY Float32 AVX path, so it
        # is always buildable (no _HAS_AVX512 gate) and tested unconditionally, unlike the Float64 W=8 path.
        # 768 = the main (B64W8) solver; the rest = small-base path: B4(12/36), B8(24/360), B16/B32/B64 +
        # radix-3/5/7/9 (48/96/112/192/240/448/480/720). The v2=1 (2·odd) sizes 6/18/54/90/162/270/486/810
        # exercise the W=8 partial-column subsystem (B2W8 base + rem=2 tails on MR3/MR5/MR9).
        for n in (768, 12, 24, 36, 40, 48, 96, 112, 192, 240, 360, 448, 480, 720,
                  6, 18, 54, 90, 162, 270, 486, 810,
                  14, 98,                                    # v2=1 · 7^k (2·7, 2·7²) — radix-7 rem=2 tail
                  7, 9, 21, 25, 27, 49, 63, 75, 81, 105, 225, 343, 441, 1225,  # v2=0 odd {3,5,7}-smooth:
                                                             # B1 base + odd-M rem∈{1,3} tails (49=7² keystone)
                  11, 13, 17, 19, 95, 22, 26, 190)           # D2: bare-prime BP-W8 leaf (11/13/19) +
                                                             # radix-2 (MR2W8) for the 2·P composites
            pf = PureFFT.AvxMixedRadixPlanW8(ComplexF32, n)
            @test pf isa PureFFT.AvxMixedRadixPlan
            x = randn(ComplexF32, n); y = copy(x); pfft!(y, pf)        # forward (unnormalized) = fft
            @test relerr(y, fft(x)) < tol(Float32)
            pii = PureFFT.AvxMixedRadixPlanW8(ComplexF32, n; inverse = true)
            pfft!(y, pii)                                              # normalized inverse → back to x
            @test relerr(y, x) < tol(Float32)
            @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(pf, randn(ComplexF32, n))
        end
        # The Float64 W=8 plans are gated on real AVX-512 (`plan_tree_w8` returns nothing without it), so the
        # hardware-specific correctness only runs where AVX-512 is present (skipped on a non-AVX-512 runner).
        if isnothing(PureFFT.AvxMixedRadixPlanW8(ComplexF64, 768))
            @info "AVX-512 not detected — W=8 plans gated off; skipping W=8 correctness"
        else
            for n in (768, 9216, 2880)                                  # radix-12 / radix-9 / radix-5 W=8-clean trees
                p = PureFFT.AvxMixedRadixPlanW8(ComplexF64, n)
                @test p isa PureFFT.AvxMixedRadixPlan
                x = randn(ComplexF64, n)
                y = copy(x); pfft!(y, p)
                @test relerr(y, fft(x)) < 1e-10
                pii = PureFFT.AvxMixedRadixPlanW8(ComplexF64, n; inverse = true)
                pfft!(y, pii)
                @test relerr(y, x) < 1e-10
            end
            @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(PureFFT.AvxMixedRadixPlanW8(ComplexF64, 768), randn(ComplexF64, 768))
        end
    end

    @testset "four-step hot path is dispatch-free" begin
        x = randn(ComplexF64, 144)
        p = plan_pfft(x; variant = :fast)
        @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(p, x)
    end
end

@testitem "Stage 11: Rader's algorithm (prime sizes)" setup = [FFTUtil] begin
    using FFTW
    # primes with smooth p-1 (route here via :fast) and direct :rader checks incl. a hard p-1
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (193, 257, 769, 1153, 389, 1009)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :rader), fft(x)) < tol(T)
        pf = plan_pfft(x; variant = :rader); y = copy(x); pfft!(y, pf)
        pii = plan_pfft(Complex{T}, n; variant = :rader, inverse = true); pfft!(y, pii)
        @test relerr(y, x) < tol(T)
    end

    @testset ":fast routes smooth-p-1 primes → RaderPlan" begin
        for n in (193, 257, 769, 1153)            # p-1 = 2^a·3^b, p ≥ 128
            @test plan_pfft(ComplexF64, n; variant = :fast) isa PureFFT.RaderPlan
        end
        @test plan_pfft(ComplexF64, 1009; variant = :fast) isa PureFFT.BluesteinPlan  # p-1 has factor 7
        @test plan_pfft(ComplexF64, 181; variant = :fast) isa PureFFT.BluesteinPlan   # p-1=180 has factor 5 (Rader loses)
        @test plan_pfft(ComplexF64, 97; variant = :fast) isa PureFFT.BluesteinPlan     # p < 128
    end
end

@testitem "JET optimization check (hot path is dispatch-free)" begin
    using JET
    for v in (:radix4avx, :radix4, :recursive, :soa, :fourstep)
        x = randn(ComplexF64, 4096)
        p = plan_pfft(x; variant = v)
        @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(p, x)
    end
end

@testitem "interface_trait duck-typed pfft!" setup = [FFTUtil] begin
    using FFTW
    # a plan satisfying the AbstractFFTPlan contract WITHOUT subtyping it (duck-typed)
    struct DuckPlan
        inner::Any
    end
    PureFFT.plan_length(p::DuckPlan)::Int = PureFFT.plan_length(p.inner)
    PureFFT.plan_inverse(p::DuckPlan)::Bool = PureFFT.plan_inverse(p.inner)
    PureFFT.apply_unnormalized!(p::DuckPlan, y) = PureFFT.apply_unnormalized!(p.inner, y)

    x = randn(ComplexF64, 1024)
    dp = DuckPlan(plan_pfft(x; variant = :radix4))
    y = copy(x)
    pfft!(y, dp)
    @test relerr(y, fft(x)) < tol(Float64)
    @test_throws ArgumentError pfft!(copy(x), "not a plan")
end

@testitem "autotuned :fast" setup = [FFTUtil] begin
    using FFTW
    @testset "vs FFTW ($T, n=$n)" for T in (Float64,),
            n in (8, 64, 256, 1024, 4096, 65536,   # power of two
                96,                                 # smooth non-pow2 → mixed-radix
                181, 5793, 11585)                   # large-prime-factor → Bluestein

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :fast), fft(x)) < tol(T)
    end
end

@testitem "real-input rfft/irfft" setup = [FFTUtil] begin
    using FFTW
    @testset "prfft vs FFTW.rfft ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)

        x = randn(T, n)
        got = prfft(x)
        @test length(got) == n ÷ 2 + 1
        @test relerr(got, FFTW.rfft(x)) < tol(T)
    end

    @testset "pirfft round-trip ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)

        x = randn(T, n)
        @test relerr(pirfft(prfft(x), n), x) < tol(T)
    end

    @testset "plan_prfft zero-alloc hot path" begin
        x = randn(Float64, 4096)
        p = plan_prfft(Float64, 4096)
        out = p.outbuf
        PureFFT.apply_rfft!(p, x, out)          # warmup
        @test (@allocated PureFFT.apply_rfft!(p, x, out)) == 0
    end
end

@testitem "AbstractFFTs plan interface" setup = [FFTUtil] begin
    using FFTW, AbstractFFTs
    using LinearAlgebra: mul!
    @testset "fft/ifft/bfft/mul!/inv ($T, n=$n)" for T in (Float64, Float32),
            n in (8, 256, 1024, 96, 1000)

        x = randn(Complex{T}, n)
        ref = fft(x)
        p = PureFFT._pure_plan_fft(x)
        @test size(p) == (n,)
        @test AbstractFFTs.fftdims(p) == 1:1
        @test relerr(p * x, ref) < tol(T)
        @test x == x
        y = Vector{Complex{T}}(undef, n)
        mul!(y, p, x)
        @test relerr(y, ref) < tol(T)
        @test relerr(p \ (p * x), x) < tol(T)
        @test relerr(inv(p) * (p * x), x) < tol(T)
        @test relerr(AbstractFFTs.plan_inv(p) * (p * x), x) < tol(T)
        pb = PureFFT._pure_plan_fft(x; inverse = true)
        @test relerr(pb * x, n .* ifft(x)) < tol(T)
    end

    @testset "in-place plan mutates and matches" begin
        x = randn(ComplexF64, 512)
        p! = PureFFT._pure_plan_fft(x; inplace = true)
        z = copy(x)
        w = p! * z
        @test w === z
        @test relerr(w, fft(x)) < tol(Float64)
    end
end

@testitem "TrimCheck trim-safety (hot path)" begin
    using TrimCheck
    # Deep juliac TRIM_SAFE verification. Confirms the @generated dynamic codelets and the four-step
    # executor are trim-safe — the Vector{Any} they use is compile-time-only (AST construction in the
    # generators); the trimmed binary contains the generated straight-line kernels, not the generator.
    @validate(
        init = begin
            using PureFFT
        end,
        PureFFT.apply_unnormalized!(PureFFT.Radix4AvxPlan{Float64}, Vector{ComplexF64}),
        PureFFT.apply_unnormalized!(PureFFT.Radix4Plan{Float64}, Vector{ComplexF64}),
        PureFFT.apply_unnormalized!(PureFFT.CodeletPlan{Float64, 12}, Vector{ComplexF64}),
        PureFFT.apply_unnormalized!(PureFFT.FourStepCodeletPlan{Float64, 12, 12}, Vector{ComplexF64}),
        PureFFT.apply_unnormalized!(PureFFT.RecursiveMixedRadixPlan{Float64, (12, 12, 12)}, Vector{ComplexF64}),
        # N-D c2c apply (Task 5): the @generated-over-D apply + per-dim transpose are trim-safe too. The
        # concrete NDPlan type is autotune-dependent (the inner :fast plans), so derive it via `typeof` of
        # a constructed plan (@validate Main.evals each arg) rather than hardcoding the fragile inner type.
        PureFFT.apply_unnormalized!(typeof(PureFFT._pure_plan_fft_nd(Array{ComplexF64}(undef, 8, 5), (1, 2); inverse = false)), Array{ComplexF64, 2}),
        PureFFT.apply_unnormalized!(typeof(PureFFT._pure_plan_fft_nd(Array{ComplexF32}(undef, 6, 4, 5), (1, 3); inverse = false)), Array{ComplexF32, 3}),
        # strided prime dim → BatchedRaderDim (batched Rader, Task 7f): the gather/scatter + batched inner
        # FFT are trim-safe (RaderPlan/BatchPlan precompute is plan-time; the hot apply is concrete).
        PureFFT.apply_unnormalized!(typeof(PureFFT._pure_plan_fft_nd(Array{ComplexF64}(undef, 16, 127), (1, 2); inverse = false)), Array{ComplexF64, 2}),
    )
end

@testitem "Performance regression (relative to FFTW)" retries = 2 skip = (get(ENV, "CI", "false") == "true") begin
    using FFTW, BenchmarkTools
    FFTW.set_num_threads(1)
    # A catastrophic-regression guard for DEV hardware (skipped on CI). The SIMD kernels target
    # 512-bit AVX-512 (`Vec{8,Float64}`); on a runner without AVX-512 they run emulated/split and
    # PureFFT is several× slower than FFTW (which adapts via MEASURE) — a hardware mismatch, not a
    # regression, so the ratio assertion is meaningless there. Precise tracking belongs in a
    # dedicated benchmark on fixed hardware. Thresholds are generous (2×): a real regression
    # (kernel falling back to mixed-radix/scalar) is 3×+; `retries` absorbs transient contention.
    @testset "PureFFT within ratio of FFTW (n=$n, ≤$(ratio)×)" for (n, ratio) in
            ((1024, 2.0), (16384, 2.0), (65536, 2.0))

        x = randn(ComplexF64, n)
        pp = plan_pfft(x; variant = :fast)
        pm = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE)
        tp = @belapsed pfft!(w, $pp) setup = (w = copy($x)) evals = 1 samples = 500
        tm = @belapsed $pm * w setup = (w = copy($x)) evals = 1 samples = 500
        @test tp <= ratio * tm
    end
end

@testitem "public API edges: tiny n, strided views, inference, ldiv!" setup = [FFTUtil] begin
    using FFTW, AbstractFFTs
    using LinearAlgebra: ldiv!

    @testset "tiny n via the default :fast ($T, n=$n)" for T in (Float64, Float32), n in (1, 2, 3)
        x = randn(Complex{T}, n)
        @test relerr(pfft(x), fft(x)) < tol(T)              # default :fast (not pow2-only :scalar) handles any n
        @test relerr(ipfft(pfft(x)), x) < tol(T)
    end

    @testset "views: contiguous + strided are correct, never silently corrupted (n=$n)" for n in (8, 90, 512)
        b = randn(ComplexF64, 2n)
        p = plan_pfft(ComplexF64, n; variant = :fast)
        cb = copy(b); cv = view(cb, 1:n); rc = fft(collect(cv)); pfft!(cv, p)          # contiguous view, in place
        @test relerr(cv, rc) < tol(Float64)
        sb = copy(b); sv = view(sb, 1:2:2n); rs = fft(collect(sv)); pfft!(sv, p)       # strided view → copy-back
        @test relerr(sv, rs) < tol(Float64)
        @test relerr(pfft(view(b, 1:2:2n)), fft(collect(view(b, 1:2:2n)))) < tol(Float64)  # out-of-place strided
    end

    @testset "inference: the public surface returns concrete vectors" begin
        x = randn(ComplexF64, 256)
        p = plan_pfft(ComplexF64, 256; variant = :fast)
        @test @inferred(pfft(x)) isa Vector{ComplexF64}
        @test @inferred(ipfft(x)) isa Vector{ComplexF64}
        @test @inferred(pfft!(copy(x), p)) isa AbstractVector{ComplexF64}
    end

    @testset "ldiv! round-trips to the input" begin
        x = randn(ComplexF64, 128)
        p = PureFFT._pure_plan_fft(x)
        z = similar(x); ldiv!(z, p, p * x)
        @test relerr(z, x) < tol(Float64)
    end
end

@testitem "Aqua project quality" begin
    using Aqua
    # piracies = false: PureFFT intentionally extends `AbstractFFTs.plan_fft`/`plan_bfft` etc. on plain
    # `Vector`s to plug into the Julia FFT ecosystem (the same deliberate extension FFTW.jl makes). Every
    # other Aqua check (ambiguities, unbound args, undefined exports, stale deps, compat, extras) is on.
    Aqua.test_all(PureFFT; piracies = false)
end
