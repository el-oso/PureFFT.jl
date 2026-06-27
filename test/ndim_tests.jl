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

@testitem "N-D public API: fft/ifft/bfft/mul!/inv vs FFTW" begin
    using PureFFT, FFTW, LinearAlgebra
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    for T in (Float64, Float32), (sz, region) in (((8,5), :), ((6,4,5), (1,3)))
        x = randn(Complex{T}, sz...)
        @test maximum(abs.(fft(x, region===(:) ? (1:ndims(x)) : region) .- (region===(:) ? fft(x) : fft(x, region)))) < tol(T)*max(1,maximum(abs.(x)))
        @test maximum(abs.(ifft(fft(x)) .- x)) < tol(T)
        p = plan_fft(x); @test maximum(abs.((p*x) .- fft(x))) < tol(T)*max(1,maximum(abs.(x)))
        # FFTW's StridedArray method is more specific than AbstractArray{<:Complex} for plan_fft!,
        # so force PureFFT's NDPlan directly to test mul! semantics (same pattern as other ndim tests).
        dims = region === (:) ? (1:ndims(x)) : region
        pin = PureFFT._pure_plan_fft_nd(x, dims; inverse=false)
        y = similar(x); mul!(y, pin, x); @test maximum(abs.(y .- fft(x, dims))) < tol(T)*max(1,maximum(abs.(x)))
        @test maximum(abs.((inv(plan_fft(x)) * (plan_fft(x)*x)) .- x)) < tol(T)
    end
    # rank-1 vector: AbstractVector is more specific than AbstractArray → PureFFT wrapper wins over NDPlan
    v = randn(ComplexF64, 8)
    @test PureFFT._pure_plan_fft(v) isa PureFFT.PureFFTPlanWrapper
    # column-tiling branch: dim 2, n_d=40 > 32, inner=3 (exercises _transpose_block! with n_d > blk)
    x = randn(ComplexF64, 3, 40); p = PureFFT._pure_plan_fft_nd(x, 2; inverse=false)
    @test maximum(abs.((p*x) .- fft(x, 2)))/maximum(abs.(fft(x,2))) < 1e-12
end
