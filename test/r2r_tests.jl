@testitem "r2r kinds + error type defined" begin
    using PureFFT
    # the 8 FFTW kind singletons exist and are R2RKind
    for k in (REDFT00, REDFT01, REDFT10, REDFT11, RODFT00, RODFT01, RODFT10, RODFT11)
        @test k isa PureFFT.R2RKind
    end
    # distinct singletons
    @test REDFT10 !== REDFT01
    # error type constructs
    @test PureFFT.R2RError(PureFFT.ERR_UNSUPPORTED_KIND, "x") isa PureFFT.R2RError
end

@testitem "tryplan_r2r returns Err for unsupported kind" begin
    using PureFFT, ErrorTypes
    @test ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(8), REDFT11))
end

@testitem "DCT-II (REDFT10) bit-exact vs FFTW + naive (even N)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    naive(x) = [2*sum(x[j+1]*cos(pi*(2j+1)*k/(2length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for T in (Float64, Float32), n in (2, 4, 8, 16, 100, 256, 1000)
        x = randn(T, n)
        y = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10))
        ref = FFTW.r2r(x, FFTW.REDFT10)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
        @test maximum(abs.(y .- T.(naive(Float64.(x))))) / max(maximum(abs.(ref)), eps(T)) < tol(T)
    end
end

@testitem "DCT-II (REDFT10) odd-N bit-exact vs FFTW" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (1, 3, 5, 7, 9, 99, 257)
        x = randn(T, n)
        y = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10))
        ref = FFTW.r2r(x, FFTW.REDFT10)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
    end
end
