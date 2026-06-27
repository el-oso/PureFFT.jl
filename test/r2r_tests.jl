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
