@testitem "ESTIMATE classifier (_estimate_plan) routes each size class" begin
    P = PureFFT
    C = ComplexF64
    # pow2 → AutoPlan-wrapped Radix4Avx
    @test P._estimate_plan(C, 1024) isa P.AutoPlan{Float64, <:P.Radix4AvxPlan}
    # 2·3·5-smooth → AvxMixedRadixPlan (raw)
    @test P._estimate_plan(C, 720) isa P.AvxMixedRadixPlan
    # prime-square (17²=289) → GenPP codelet (5²/7² use hand B25/B49 via the smooth tree, NOT _gen_pp_prime)
    @test P._estimate_plan(C, 289) isa P.GenPPCodeletPlan
    # large prime with 2^a·3^b p-1 → Rader (257-1 = 256 = 2^8; RADER_MAX_PM1_PRIME=3 requires p-1 = 2^a·3^b)
    @test P._estimate_plan(C, 257) isa P.RaderPlan
    # unclassified (19946 = 2·9973, 9973 prime → not pow2/Rader/GenPP/smooth) → nothing (fallback)
    @test isnothing(P._estimate_plan(C, 19946))
    # returns a valid PlanRigor enum
    @test P.ESTIMATE isa P.PlanRigor && P.MEASURE isa P.PlanRigor
end
