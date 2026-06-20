using PureFFT
using FFTW
using Test
using TrimCheck
using JET
using AbstractFFTs
using LinearAlgebra: mul!

# Relative L2 error between a candidate transform and the FFTW reference.
relerr(a, b) = sqrt(sum(abs2, a .- b) / sum(abs2, b))

# Per-precision tolerance: FFTW and a well-conditioned radix-2 should agree to near
# machine epsilon scaled by the transform depth.
tol(::Type{Float64}) = 1.0e-11
tol(::Type{Float32}) = 1.0e-4

@testset "PureFFT Stage 1: radix-2 baseline" begin
    @testset "forward vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 64, 256, 1024, 4096)

        x = randn(Complex{T}, n)
        ref = fft(x)

        # recursive reference
        @test relerr(PureFFT.radix2_rec(x, false), ref) < tol(T)

        # in-place iterative baseline through the public plan API
        y = copy(x)
        p = plan_pfft(y; variant = :scalar)
        pfft!(y, p)
        @test relerr(y, ref) < tol(T)

        # out-of-place convenience does not mutate input
        x0 = copy(x)
        @test relerr(pfft(x), ref) < tol(T)
        @test x == x0
    end

    @testset "inverse round-trips ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 8, 64, 1024)

        x = randn(Complex{T}, n)
        @test relerr(ipfft(pfft(x)), x) < tol(T)

        # inverse matches FFTW ifft
        @test relerr(ipfft(x), ifft(x)) < tol(T)
    end

    @testset "errors on non-power-of-two" begin
        @test_throws ArgumentError plan_pfft(Complex{Float64}, 6)
    end
end

@testset "PureFFT Stage 2: mixed-radix" begin
    @testset "factorize" begin
        @test PureFFT.factorize(1) == Int[]
        @test PureFFT.factorize(8) == [4, 2]
        @test PureFFT.factorize(16) == [4, 4]
        @test prod(PureFFT.factorize(1000)) == 1000
        @test PureFFT.factorize(7) == [7]
        @test prod(PureFFT.factorize(210)) == 210   # 2·3·5·7
    end

    @testset "forward vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 3, 5, 6, 7, 12, 15, 36, 100, 210, 1000, 1024, 2048)

        x = randn(Complex{T}, n)
        ref = fft(x)
        @test relerr(pfft(x; variant = :mixedradix), ref) < tol(T)
    end

    @testset "inverse round-trip ($T, n=$n)" for T in (Float64, Float32),
            n in (6, 7, 100, 210, 1000)

        x = randn(Complex{T}, n)
        y = pfft(x; variant = :mixedradix)
        @test relerr(ipfft(y; variant = :mixedradix), x) < tol(T)
    end
end

@testset "PureFFT Stage 3: staged radix-2 (scalar + Base @simd)" begin
    @testset "$variant vs FFTW ($T, n=$n)" for variant in (:staged, :base),
            T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096)

        x = randn(Complex{T}, n)
        ref = fft(x)
        @test relerr(pfft(x; variant), ref) < tol(T)

        # inverse round-trip through the same kernel
        @test relerr(ipfft(pfft(x; variant); variant), x) < tol(T)
    end
end

@testset "PureFFT Stage 4: cache-oblivious recursive" begin
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :recursive), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :recursive); variant = :recursive), x) < tol(T)
    end
end

@testset "PureFFT Stage 5: SoA recursive" begin
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :soa), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :soa); variant = :soa), x) < tol(T)
    end
end

@testset "PureFFT Stage 7: cache-blocked four-step" begin
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 32768, 65536, 262144)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :fourstep), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :fourstep); variant = :fourstep), x) < tol(T)
    end
end

@testset "PureFFT Radix4 ($variant)" for variant in (:radix4, :radix4simd, :radix4avx)
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536, 262144)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant); variant), x) < tol(T)
    end
end

@testset "PureFFT Stage 8: Bluestein (chirp-Z, arbitrary n)" begin
    # Primes, prime powers, and large-prime-factor composites — the sizes mixed-radix would
    # handle with an O(n²) direct DFT. Includes a power of two (M-path edge case).
    @testset "vs FFTW ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 3, 5, 7, 11, 13, 17, 64, 91, 100, 181, 256, 362, 1000, 5793)

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :bluestein), fft(x)) < tol(T)
        @test relerr(ipfft(pfft(x; variant = :bluestein); variant = :bluestein), x) < tol(T)
    end
end

# a plan that satisfies the AbstractFFTPlan contract WITHOUT subtyping it (duck-typed)
struct DuckPlan
    inner::Any
end
PureFFT.plan_length(p::DuckPlan)::Int = PureFFT.plan_length(p.inner)
PureFFT.plan_inverse(p::DuckPlan)::Bool = PureFFT.plan_inverse(p.inner)
PureFFT.apply_unnormalized!(p::DuckPlan, y) = PureFFT.apply_unnormalized!(p.inner, y)

@testset "JET optimization check (hot path is dispatch-free)" begin
    for v in (:radix4avx, :radix4, :recursive, :soa, :fourstep)
        x = randn(ComplexF64, 4096)
        p = plan_pfft(x; variant = v)
        @test_opt target_modules = (PureFFT,) PureFFT.apply_unnormalized!(p, x)
    end
end

@testset "interface_trait duck-typed pfft!" begin
    x = randn(ComplexF64, 1024)
    dp = DuckPlan(plan_pfft(x; variant = :radix4))
    y = copy(x)
    pfft!(y, dp)                                   # dispatched via interface_trait
    @test relerr(y, fft(x)) < tol(Float64)
    @test_throws ArgumentError pfft!(copy(x), "not a plan")
end

@testset "PureFFT autotuned :fast" begin
    @testset "vs FFTW ($T, n=$n)" for T in (Float64,),
            n in (8, 64, 256, 1024, 4096, 65536,   # power of two
                96,                                 # smooth non-pow2 → mixed-radix
                181, 5793, 11585)                   # large-prime-factor → Bluestein

        x = randn(Complex{T}, n)
        @test relerr(pfft(x; variant = :fast), fft(x)) < tol(T)
    end
end

@testset "PureFFT real-input rfft/irfft" begin
    @testset "prfft vs FFTW.rfft ($T, n=$n)" for T in (Float64, Float32),
            n in (2, 4, 8, 16, 64, 256, 1024, 4096, 16384, 65536)

        x = randn(T, n)
        ref = FFTW.rfft(x)
        got = prfft(x)
        @test length(got) == n ÷ 2 + 1
        @test relerr(got, ref) < tol(T)
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
        allocs = @allocated PureFFT.apply_rfft!(p, x, out)
        @test allocs == 0
    end
end

@testset "AbstractFFTs plan interface" begin
    # _pure_plan_fft forces the PureFFT path even with FFTW loaded (FFTW's StridedVector methods
    # are otherwise more specific). Covers pow2 (radix4avx) and non-pow2 (Bluestein) sizes.
    @testset "fft/ifft/bfft/mul!/inv ($T, n=$n)" for T in (Float64, Float32),
            n in (8, 256, 1024, 96, 1000)

        x = randn(Complex{T}, n)
        ref = fft(x)
        p = PureFFT._pure_plan_fft(x)
        @test size(p) == (n,)
        @test AbstractFFTs.fftdims(p) == 1:1
        @test relerr(p * x, ref) < tol(T)                         # * (out-of-place fft)
        @test x == x                                              # * did not mutate input
        y = Vector{Complex{T}}(undef, n)
        mul!(y, p, x)
        @test relerr(y, ref) < tol(T)                             # mul! into preallocated
        @test relerr(p \ (p * x), x) < tol(T)                     # \ round-trip
        @test relerr(inv(p) * (p * x), x) < tol(T)                # inv() cached round-trip
        ifp = AbstractFFTs.plan_inv(p)                            # ifft = ScaledPlan(bfft, 1/N)
        @test relerr(ifp * (p * x), x) < tol(T)
        pb = PureFFT._pure_plan_fft(x; inverse = true)            # bfft = unnormalized inverse
        @test relerr(pb * x, n .* ifft(x)) < tol(T)
    end

    @testset "in-place plan mutates and matches" begin
        x = randn(ComplexF64, 512)
        ref = fft(x)
        p! = PureFFT._pure_plan_fft(x; inplace = true)
        z = copy(x)
        w = p! * z
        @test w === z                                            # operated in place
        @test relerr(w, ref) < tol(Float64)
    end
end

# Deep trim-safety check (juliac TRIM_SAFE verifier, via TrimCheck.jl) of the hot path —
# stronger than TypeContracts' shallow trim_compat. Confirms no dynamic dispatch / trim-unsafe
# calls reach the actual trimmed-compilation verifier, incl. the SIMD.jl AVX kernel.
@validate(
    init = begin
        using PureFFT
    end,
    PureFFT.apply_unnormalized!(PureFFT.Radix4AvxPlan{Float64}, Vector{ComplexF64}),
    PureFFT.apply_unnormalized!(PureFFT.Radix4Plan{Float64}, Vector{ComplexF64}),
)
