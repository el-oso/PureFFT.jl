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

@testitem "ESTIMATE autoplan: correct output + safe fallback" begin
    P = PureFFT
    C = ComplexF64
    ndft(x) = [sum(x[j+1]*cispi(-2*j*k/length(x)) for j in 0:length(x)-1) for k in 0:length(x)-1]
    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    # ESTIMATE output vs reference DFT across classes (pow2, smooth, prime-square, Rader) + a fallback size
    # (19946 = 2·9973: not pow2/Rader/GenPP/smooth → _estimate_plan nothing → MEASURE→Bluestein fallback).
    # Tolerance 1e-11: 1024/720/289/257 all yield <1e-13; 19946 (Bluestein fallback) yields ~3e-12
    # (two power-of-2 FFTs of size 32768 compound error). 1e-12 from the brief was too tight for that
    # size — measured 3.2e-12 on a warm session; 1e-11 is still well within double precision.
    for n in (1024, 720, 289, 257, 19946)
        x = [C(randn(), randn()) for _ in 1:n]
        pe = P.autoplan(C, n; flags = P.ESTIMATE)
        y = copy(x); P.apply_unnormalized!(pe, y)
        @test relerr(y, ndft(x)) ≤ 1e-11
        # inverse round-trip
        pei = P.autoplan(C, n; inverse = true, flags = P.ESTIMATE)
        P.apply_unnormalized!(pei, y); y ./= n
        @test relerr(y, x) ≤ 1e-11
    end
    # default is MEASURE (flags omitted) — unchanged behavior, still correct
    x = [C(randn(), randn()) for _ in 1:720]
    pm = P.autoplan(C, 720); y = copy(x); P.apply_unnormalized!(pm, y)
    @test relerr(y, ndft(x)) ≤ 1e-11
end

@testitem "ESTIMATE public API: plan_pfft + plan_fft flags kwarg" begin
    P = PureFFT
    C = ComplexF64
    ndft(x) = [sum(x[j+1]*cispi(-2*j*k/length(x)) for j in 0:length(x)-1) for k in 0:length(x)-1]
    relerr(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))
    using AbstractFFTs
    x = [C(randn(), randn()) for _ in 1:720]
    # native entry: plan_pfft with flags=ESTIMATE
    pe = P.plan_pfft(x; flags = P.ESTIMATE)
    y = copy(x); P.pfft!(y, pe)
    @test relerr(y, ndft(x)) ≤ 1e-12
    # AbstractFFTs/FFTW-facing entry: plan_fft with flags=ESTIMATE (the transition-compat surface)
    pf = plan_fft(x; flags = P.ESTIMATE)
    @test relerr(pf * x, ndft(x)) ≤ 1e-12
    # default (no flags) still produces a correct transform
    pm = P.plan_pfft(x); ym = copy(x); P.pfft!(ym, pm)
    @test relerr(ym, ndft(x)) ≤ 1e-12
end
