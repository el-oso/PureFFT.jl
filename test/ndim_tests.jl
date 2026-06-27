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
