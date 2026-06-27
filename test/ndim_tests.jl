@testitem "NDPlan builds + region canonicalization" begin
    using PureFFT
    x = randn(ComplexF64, 4, 6, 8)
    p = PureFFT._pure_plan_fft_nd(x, (3, 1); inverse=false)   # unsorted, partial region
    @test p isa PureFFT.NDPlan
    @test p.dims == (1, 3)                # sorted, deduped
    @test length(p.plans) == 2
    @test p.sz == (4, 6, 8)
    @test_throws ArgumentError PureFFT._pure_plan_fft_nd(x, (1, 4); inverse=false)  # dim 4 ∉ 1:3
    @test PureFFT._pure_plan_fft_nd(x, (1, 1); inverse=false).dims == (1,)          # dup → deduped
end

@testitem "N-D c2c along dim 1 bit-exact vs FFTW" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32)
        for sz in ((8,), (8,5), (6,4,3))
            x = randn(Complex{T}, sz...)
            p = PureFFT._pure_plan_fft_nd(x, (1,); inverse=false)
            y = copy(x); PureFFT.apply_unnormalized!(p, y)
            ref = fft(x, 1)                      # FFTW along dim 1
            @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol(T)
        end
    end
end

@testitem "N-D c2c full generality bit-exact vs FFTW" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    cases = (((8,5), 2), ((8,5), (1,2)), ((6,4,5), 3), ((6,4,5), (1,3)), ((6,4,5), (1,2,3)), ((4,4,4,4), (2,4)))
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(Complex{T}, sz...)
        p = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        y = copy(x); PureFFT.apply_unnormalized!(p, y)
        ref = fft(x, region)
        @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol(T)
    end
end

@testitem "N-D public API: mul!/dispatch/column-tiling (PureFFT path)" begin
    using PureFFT, FFTW, LinearAlgebra
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32), (sz, region) in (((8,5), :), ((6,4,5), (1,3)))
        x = randn(Complex{T}, sz...)
        # Force PureFFT's NDPlan directly — FFTW's StridedArray method is more specific than
        # AbstractArray{<:Complex}, so plan_fft(::Matrix) routes to FFTW when FFTW is loaded.
        dims = region === (:) ? (1:ndims(x)) : region
        pin = PureFFT._pure_plan_fft_nd(x, dims; inverse=false)
        y = similar(x); mul!(y, pin, x); @test maximum(abs.(y .- fft(x, dims))) < tol(T)*max(1,maximum(abs.(x)))
    end
    # rank-1 vector: AbstractVector is more specific than AbstractArray → PureFFT wrapper wins over NDPlan
    v = randn(ComplexF64, 8)
    @test PureFFT._pure_plan_fft(v) isa PureFFT.PureFFTPlanWrapper
    # column-tiling branch: dim 2, n_d=40 > 32, inner=3 (exercises _transpose_block! with n_d > blk)
    x = randn(ComplexF64, 3, 40); p = PureFFT._pure_plan_fft_nd(x, 2; inverse=false)
    @test maximum(abs.((p*x) .- fft(x, 2)))/maximum(abs.(fft(x,2))) < 1e-12
end

@testitem "N-D inv/plan_inv round-trip (PureFFT path)" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32), (sz, region) in (((8,5), (1,2)), ((6,4,5), (1,3)))
        x = randn(Complex{T}, sz...)
        pf = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        # inv(pf) returns _NDScaledPlan wrapping inverse NDPlan scaled by 1/∏sz[region]
        @test maximum(abs.(inv(pf) * (pf * x) .- x)) < tol(T)
    end
end

@testitem "N-D mul! DimensionMismatch guard" begin
    using PureFFT, LinearAlgebra
    x = randn(ComplexF64, 4, 6)
    p = PureFFT._pure_plan_fft_nd(x, (1,2); inverse=false)
    y_bad = similar(x, 3, 6)   # wrong first dim
    @test_throws DimensionMismatch mul!(y_bad, p, x)
    x_bad = similar(x, 4, 5)   # wrong second dim
    @test_throws DimensionMismatch mul!(similar(x), p, x_bad)
end

@testitem "pfft(::AbstractArray, dims) correctness" begin
    using PureFFT, FFTW
    # pfft(::Matrix) routes through FFTW's plan_fft when FFTW is loaded (FFTW's StridedArray
    # method is more specific); correctness is still testable — both sides use the same AbstractFFTs path.
    x = randn(ComplexF64, 8, 5)
    @test maximum(abs.(PureFFT.pfft(x, (1,2)) .- fft(x, (1,2))))/maximum(abs.(fft(x,(1,2)))) < 1e-12
    @test maximum(abs.(PureFFT.pfft(x, 1) .- fft(x, 1)))/maximum(abs.(fft(x,1))) < 1e-12
    # 1-D vector routes to PureFFT (AbstractVector more specific than AbstractArray)
    v = randn(ComplexF64, 8)
    @test maximum(abs.(PureFFT.pfft(v) .- fft(v)))/maximum(abs.(fft(v))) < 1e-12
end

@testitem "N-D c2c hot path: dispatch-free + zero-alloc" begin
    using PureFFT, JET, LinearAlgebra
    # NOTE: `plan_fft(x, region)` is intentionally NOT used here — with FFTW loaded its StridedArray
    # method shadows PureFFT's AbstractArray method, so `plan_fft` returns an FFTW plan, not an NDPlan
    # (verified; cf. the routing comment in src/ndim.jl). To gate PureFFT's own N-D hot path we build
    # the NDPlan directly via `_pure_plan_fft_nd`, exactly as the other ndim testitems do.
    # Cover BOTH routings: transpose-routed (non-pow2 d>1) AND batched-routed (pow2 d>1) plans.
    cases = (((8,5),(1,2)),       # dim2 n_d=5 non-pow2 → TransposeDim
             ((6,4,5),(1,3)),     # dim3 n_d=5 non-pow2 → TransposeDim
             ((8,16),(1,2)),      # dim2 n_d=16 pow2  → BatchedDim
             ((4,8,16),(2,3)))    # dim2 n_d=8, dim3 n_d=16 pow2 → BatchedDim (both d>1)
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(Complex{T}, sz...); y = similar(x)
        p = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        mul!(y, p, x)                                   # warmup
        @test (@allocated mul!(y, p, x)) == 0
        @test_opt target_modules=(PureFFT,) mul!(y, p, x)
    end
end
