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
    cases = (((8,5), 2), ((8,5), (1,2)), ((6,4,5), 3), ((6,4,5), (1,3)), ((6,4,5), (1,2,3)), ((4,4,4,4), (2,4)),
             # 2^a·3^b strided dims route to BatchedSmoothDim (mixed-radix batched, no transpose):
             ((4,48), 2), ((8,96), 2), ((8,8,24), 3), ((4,12,8), 2), ((9,16), 2),   # n_d=48,96,24,12,9
             ((3,48), 2), ((6,96), 2), ((4,4,48), (2,3)))                            # scalar/partial tails + inv-free
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(Complex{T}, sz...)
        p = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        y = copy(x); PureFFT.apply_unnormalized!(p, y)
        ref = fft(x, region)
        @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol(T)
    end
end

@testitem "N-D BatchedDim1 (batched dim-1, F32) bit-exact incl outer%W tail + inv" begin
    using PureFFT, FFTW
    tol = 1f-4
    # n1 routed to BatchedDim1 for F32: 64 (pow2→BatchPlan8), 48/96 (smooth→BatchPlanMR). outer chosen
    # to exercise the outer%W=8 tail (last chunk m not a multiple of W) and multi-chunk (outer>M).
    cases = ((64, 19), (64, 200), (48, 11), (96, 13), (48, 3, 5))   # outer = ∏ trailing dims
    for sz in cases
        # confirm routing actually hit BatchedDim1 (else the test is vacuous)
        pd = PureFFT._mk_dim(ComplexF32, 1, sz; inverse=false)
        @test pd isa PureFFT.BatchedDim1
        x = randn(ComplexF32, sz...)
        p = PureFFT._pure_plan_fft_nd(x, (1,); inverse=false)
        y = copy(x); PureFFT.apply_unnormalized!(p, y)
        ref = fft(x, 1)
        @test maximum(abs.(y .- ref))/maximum(abs.(ref)) < tol
        # inverse round-trip through the batched dim-1 path
        pinv = PureFFT._pure_plan_fft_nd(x, (1,); inverse=true)
        z = copy(y); PureFFT.apply_unnormalized!(pinv, z)
        @test maximum(abs.(z ./ sz[1] .- x)) < tol
    end
    # F64 same shapes must NOT route to BatchedDim1 (stay per-column Dim1Plan)
    @test PureFFT._mk_dim(ComplexF64, 1, (64, 19); inverse=false) isa PureFFT.Dim1Plan
    # large pow2 n1 stays per-column even for F32 (batched loses there)
    @test PureFFT._mk_dim(ComplexF32, 1, (256, 256); inverse=false) isa PureFFT.Dim1Plan
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
    for T in (Float64, Float32), (sz, region) in (((8,5), (1,2)), ((6,4,5), (1,3)), ((8,96), (1,2)), ((4,4,48), (2,3)))
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
             ((4,8,16),(2,3)),    # dim2 n_d=8, dim3 n_d=16 pow2 → BatchedDim (both d>1)
             ((8,48),(1,2)),      # dim2 n_d=48=2^4·3 → BatchedSmoothDim (mixed-radix batched)
             ((4,8,24),(2,3)),    # dim2 n_d=8 pow2, dim3 n_d=24=2^3·3 smooth → mixed routing
             ((64,19),(1,)),      # dim1 n1=64 F32 → BatchedDim1 (pow2), outer%W tail (F64 → Dim1Plan)
             ((48,13),(1,)))      # dim1 n1=48 F32 → BatchedDim1 (smooth), outer%W tail
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(Complex{T}, sz...); y = similar(x)
        p = PureFFT._pure_plan_fft_nd(x, region; inverse=false)
        mul!(y, p, x)                                   # warmup
        @test (@allocated mul!(y, p, x)) == 0
        @test_opt target_modules=(PureFFT,) mul!(y, p, x)
    end
end

# ── Real N-D (rfft / irfft / brfft) ──────────────────────────────────────────
# FFTW.jl shadows AbstractFFTs.plan_rfft/plan_brfft for StridedArrays, so we build PureFFT's
# RealNDPlan directly via _pure_plan_rfft_nd / _pure_plan_brfft_nd (exactly as the c2c testitems do)
# and use FFTW's rfft/irfft/brfft as the golden reference.
@testitem "Real N-D rfft forward bit-exact vs FFTW" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    # r2c dim = first(region) must be EVEN; order matters (NOT sorted). c2c rest may be any length.
    cases = (((8,6), 1), ((8,6), 2), ((8,6), (1,2)), ((8,6), (2,1)),
             ((6,4,8), (1,2,3)), ((6,4,8), (1,3)), ((4,6,8), 2),
             ((6,5), 1), ((6,5), (1,2)), ((8,9), (1,2)), ((6,4), (2,1)))
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(T, sz...)
        Y = PureFFT._pure_plan_rfft_nd(x, region) * x
        ref = rfft(x, region)
        @test size(Y) == size(ref)
        @test maximum(abs.(Y .- ref)) / maximum(abs.(ref)) < tol(T)
    end
    # Colon region == 1:ndims (FFTW.rfft can't take Colon, so reference with the range)
    x = randn(Float64, 8,6,4)
    @test maximum(abs.((PureFFT._pure_plan_rfft_nd(x, :) * x) .- rfft(x, 1:3))) / maximum(abs.(rfft(x,1:3))) < 1e-12
end

@testitem "Real N-D irfft/brfft bit-exact vs FFTW + round-trip" begin
    using PureFFT, FFTW
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    cases = (((8,6), 1), ((8,6), 2), ((8,6), (1,2)), ((8,6), (2,1)),
             ((6,4,8), (1,2,3)), ((6,4,8), (1,3)), ((4,6,8), 2),
             ((6,5), (1,2)), ((8,9), (1,2)))
    for T in (Float64, Float32), (sz, region) in cases
        x = randn(T, sz...)
        Y = rfft(x, region)                       # FFTW half-spectrum (same as ours, verified above)
        rl = region isa Int ? region : first(region)
        n = sz[rl]
        # brfft (unnormalized) bit-exact vs FFTW
        bb = PureFFT._pure_plan_brfft_nd(Y, n, region) * Y
        @test maximum(abs.(bb .- brfft(Y, n, region))) / maximum(abs.(brfft(Y, n, region))) < tol(T)
        # irfft (normalized via prefixed helper) round-trips AND matches FFTW.irfft
        yb = PureFFT.pirfft(Y, n, region)
        @test maximum(abs.(yb .- x)) / maximum(abs.(x)) < tol(T)
        @test maximum(abs.(yb .- irfft(Y, n, region))) / maximum(abs.(x)) < tol(T)
    end
end

@testitem "Real N-D AbstractFFTs derivation (ScaledPlan irfft) + mul!" begin
    using PureFFT, FFTW, AbstractFFTs, LinearAlgebra
    x = randn(Float64, 6,4,8)
    region = (1,3); n = 6
    pf = PureFFT._pure_plan_rfft_nd(x, region)
    Y = Array{ComplexF64}(undef, AbstractFFTs.rfft_output_size(x, region))
    mul!(Y, pf, x)                                   # forward mul!
    @test maximum(abs.(Y .- rfft(x, region))) / maximum(abs.(rfft(x,region))) < 1e-12
    # irfft derived from our plan_brfft through AbstractFFTs.ScaledPlan (the drop-in path)
    pb = PureFFT._pure_plan_brfft_nd(Y, n, region)
    sp = AbstractFFTs.ScaledPlan(pb, AbstractFFTs.normalization(Float64, AbstractFFTs.brfft_output_size(Y, n, region), region))
    @test maximum(abs.((sp * Y) .- x)) / maximum(abs.(x)) < 1e-12
    @test AbstractFFTs.fftdims(pf) == (1, 3)
end

@testitem "Real N-D error paths" begin
    using PureFFT
    @test_throws ArgumentError PureFFT._pure_plan_rfft_nd(randn(7,4), 1)        # odd r2c dim
    @test_throws ArgumentError PureFFT._pure_plan_rfft_nd(randn(8,4), (1,3))    # dim 3 ∉ 1:2
    @test_throws ArgumentError PureFFT._pure_plan_rfft_nd(randn(8,4), 3)        # r2c dim out of bounds
    # brfft size guard: size(X, first(region)) must == n÷2+1
    X = randn(ComplexF64, 5, 4)
    @test_throws ArgumentError PureFFT._pure_plan_brfft_nd(X, 10, 1)            # 10÷2+1=6 ≠ 5
end
