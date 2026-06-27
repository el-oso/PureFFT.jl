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
