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

@testitem "tryplan_r2r returns Ok for all 8 implemented kinds" begin
    using PureFFT, ErrorTypes
    for k in (REDFT00, REDFT10, REDFT01, REDFT11, RODFT00, RODFT10, RODFT01, RODFT11)
        @test !ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(8), k))
    end
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

@testitem "DCT-III (REDFT01) bit-exact vs FFTW + II↔III round-trip" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (2, 4, 8, 16, 100, 256, 3, 5, 99)
        x = randn(T, n)
        y   = ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT01))
        ref = FFTW.r2r(x, FFTW.REDFT01)
        @test maximum(abs.(y .- ref)) / max(maximum(abs.(ref)), eps(T)) < tol(T)
        # unnormalized round-trip: REDFT01 ∘ REDFT10 = 2N·identity
        rt = ErrorTypes.unwrap(PureFFT.tryr2r(ErrorTypes.unwrap(PureFFT.tryr2r(x, REDFT10)), REDFT01))
        @test maximum(abs.(rt ./ (2n) .- x)) / max(maximum(abs.(x)), eps(T)) < tol(T)
    end
end

@testitem "r2r/dct throwing API + orthonormal dct/idct + inv/\\" begin
    using PureFFT, FFTW, LinearAlgebra
    # FFTW also exports `dct`/`idct`/`dct!`/`plan_dct` (owned by FFTW, not AbstractFFTs), so under
    # `using PureFFT, FFTW` those bare names are an ambiguous-binding conflict — qualify PureFFT's.
    const PD = PureFFT
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1f-4
    for T in (Float64, Float32), n in (4, 8, 16, 100, 7)
        x = randn(T, n)
        @test maximum(abs.(r2r(x, REDFT10) .- FFTW.r2r(x, FFTW.REDFT10))) < tol(T)*max(1, maximum(abs.(x))*n)
        @test maximum(abs.(PD.dct(x) .- FFTW.dct(x))) < tol(T)         # orthonormal, matches FFTW.jl
        @test maximum(abs.(PD.idct(x) .- FFTW.idct(x))) < tol(T)       # ortho DCT-III matches FFTW.jl
        @test maximum(abs.(PD.idct(PD.dct(x)) .- x)) < tol(T)          # ortho round-trip
        p = plan_r2r(x, REDFT10); @test maximum(abs.((p*x) .- r2r(x, REDFT10))) < tol(T)*max(1, maximum(abs.(x))*n)
        # mul! into preallocated output
        y = similar(x); mul!(y, p, x); @test maximum(abs.(y .- r2r(x, REDFT10))) < tol(T)*max(1, maximum(abs.(x))*n)
        # inv / \ : unnormalized inverse of REDFT10 (REDFT01 with 1/2N scale) recovers x
        @test maximum(abs.((p \ (p*x)) .- x)) < tol(T)*max(1, maximum(abs.(x))*n)
        # dct! / idct! mutate in place
        xc = copy(x); PD.dct!(xc); @test maximum(abs.(xc .- PD.dct(x))) < tol(T)
    end
    @test_throws ArgumentError plan_r2r(randn(1), REDFT00)            # n<2 size error → throws
    @test_throws ArgumentError r2r(randn(1), REDFT00)
end

@testitem "r2r hot path: zero-alloc + dispatch-free" begin
    using PureFFT, JET, LinearAlgebra
    # Even N: REDFT10/01/RODFT10/01 take the real-FFT route (preallocated rbuf/cbuf);
    # REDFT11/RODFT11/REDFT00/RODFT00 are route-independent (complex or extension, always zero-alloc).
    for kind in (REDFT00,REDFT10,REDFT01,REDFT11,RODFT00,RODFT10,RODFT01,RODFT11), T in (Float64, Float32), n in (8, 256)
        x = randn(T, n); y = similar(x); p = plan_r2r(x, kind)
        mul!(y, p, x)                                  # warmup
        @test (@allocated mul!(y, p, x)) == 0
        @test_opt target_modules=(PureFFT,) mul!(y, p, x)
    end
end

@testitem "DCT-IV (REDFT11) bit-exact vs FFTW + self-inverse" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dct4(x) = [2*sum(x[j+1]*cos(pi*(2j+1)*(2k+1)/(4length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for T in (Float64, Float32), n in (1,2,3,4,5,8,9,16,17,32)
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, REDFT11))
        sc = max(1, maximum(abs.(Float64.(x)))*n)
        @test maximum(abs.(Float64.(y) .- FFTW.r2r(Float64.(x), FFTW.REDFT11)))/sc < tol(T)
        @test maximum(abs.(Float64.(y) .- naive_dct4(Float64.(x))))/sc < tol(T)             # independent ref, F64+F32
        # REDFT11 self-inverse up to 2N
        @test maximum(abs.(Float64.(unwrap(PureFFT.tryr2r(y, REDFT11))) .- 2n .* Float64.(x)))/sc < tol(T)
    end
end

@testitem "DST-IV (RODFT11) bit-exact vs FFTW + naive + self-inverse (F64+F32)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dst4(x) = [2*sum(x[j+1]*sin(pi*(2j+1)*(2k+1)/(4length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for T in (Float64, Float32), n in (1,2,3,4,5,8,9,16,17,32)
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, RODFT11))
        @test maximum(abs.(y .- FFTW.r2r(x, FFTW.RODFT11)))/max(1,maximum(abs.(x))*n) < tol(T)
        @test maximum(abs.(y .- T.(naive_dst4(Float64.(x)))))/max(1,maximum(abs.(x))*n) < tol(T)  # independent ref
        # RODFT11 self-inverse up to 2N
        @test maximum(abs.(unwrap(PureFFT.tryr2r(y, RODFT11)) .- 2n .* x))/max(1,maximum(abs.(x))*n) < tol(T)
    end
    # inv / \ : self-inverse with 1/2N scale recovers x
    for T in (Float64, Float32), n in (4, 8, 17)
        x = randn(T, n); p = plan_r2r(x, RODFT11)
        @test maximum(abs.((p \ (p*x)) .- x))/max(1,maximum(abs.(x))) < tol(T)
    end
end

@testitem "DST-II (RODFT10) bit-exact vs FFTW + naive (F64+F32, even+odd N)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dst2(x) = [2*sum(x[j+1]*sin(pi*(2j+1)*(k+1)/(2length(x))) for j in 0:length(x)-1) for k in 0:length(x)-1]
    for T in (Float64, Float32), n in (1,2,3,4,5,8,9,16,17,32)   # even N → real-FFT route, odd N → complex fallback
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, RODFT10))
        @test maximum(abs.(y .- FFTW.r2r(x, FFTW.RODFT10)))/max(1,maximum(abs.(x))*n) < tol(T)
        @test maximum(abs.(y .- T.(naive_dst2(Float64.(x)))))/max(1,maximum(abs.(x))*n) < tol(T)  # independent ref
    end
end

@testitem "DST-III (RODFT01) bit-exact vs FFTW + naive + II↔III round-trip + inv/\\ (F64+F32, even+odd N)" begin
    using PureFFT, FFTW, ErrorTypes, LinearAlgebra
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dst3(x) = (N=length(x); [((-1)^k)*x[N] + 2*sum((x[j+1]*sin(pi*(j+1)*(2k+1)/(2N)) for j in 0:N-2); init=0.0) for k in 0:N-1])
    for T in (Float64, Float32), n in (1,2,3,4,5,8,9,16,17,32)   # even N → real-IFFT route, odd N → complex fallback
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, RODFT01))
        @test maximum(abs.(y .- FFTW.r2r(x, FFTW.RODFT01)))/max(1,maximum(abs.(x))*n) < tol(T)
        @test maximum(abs.(y .- T.(naive_dst3(Float64.(x)))))/max(1,maximum(abs.(x))*n) < tol(T)  # independent ref
        # unnormalized round-trip: RODFT01 ∘ RODFT10 = 2N·identity
        rt = unwrap(PureFFT.tryr2r(unwrap(PureFFT.tryr2r(x, RODFT10)), RODFT01))
        @test maximum(abs.(rt ./ (2n) .- x))/max(maximum(abs.(x)), eps(T)) < tol(T)
    end
    # inv / \ : the II↔III pair (RODFT01 = 1/2N·inverse of RODFT10) recovers x in BOTH directions
    for T in (Float64, Float32), n in (4, 8, 17)
        x = randn(T, n)
        p10 = plan_r2r(x, RODFT10); @test maximum(abs.((p10 \ (p10*x)) .- x))/max(1,maximum(abs.(x))) < tol(T)
        p01 = plan_r2r(x, RODFT01); @test maximum(abs.((p01 \ (p01*x)) .- x))/max(1,maximum(abs.(x))) < tol(T)
    end
end

@testitem "DCT-I (REDFT00) bit-exact vs FFTW + naive + self-inverse (F64+F32, even+odd N)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dct1(x)=(N=length(x); [x[1]+((-1)^k)*x[N]+2*sum(x[j+1]*cos(pi*j*k/(N-1)) for j in 1:N-2; init=0.0) for k in 0:N-1])
    for T in (Float64, Float32), n in (2,3,4,5,8,9,16,17,32)
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, REDFT00))
        ref = FFTW.r2r(x, FFTW.REDFT00)
        @test maximum(abs.(y .- ref))/max(1,maximum(abs.(ref))) < tol(T)
        @test maximum(abs.(y .- T.(naive_dct1(Float64.(x)))))/max(1,maximum(abs.(ref))) < tol(T)
        # REDFT00 self-inverse up to 2(N−1)
        @test maximum(abs.(unwrap(PureFFT.tryr2r(y, REDFT00)) .- 2*(n-1) .* x))/max(1,maximum(abs.(x))*n) < tol(T)
    end
    # tryplan_r2r returns Ok for REDFT00 (no longer "unsupported")
    @test !ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(8), REDFT00))
    # inv / \ : self-inverse with 1/(2(N−1)) scale recovers x
    for T in (Float64, Float32), n in (4, 8, 17)
        x = randn(T, n); p = plan_r2r(x, REDFT00)
        @test maximum(abs.((p \ (p*x)) .- x))/max(1,maximum(abs.(x))) < tol(T)
    end
    # size < 2 returns Err
    @test ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(1), REDFT00))
end

@testitem "small-N @generated r2r codelet: route selected + bit-exact vs FFTW & naive" begin
    using PureFFT, FFTW, LinearAlgebra
    tol(::Type{Float64}) = 1e-12; tol(::Type{Float32}) = 1.0f-4
    # naive reference sums (FFTW unnormalized r2r definitions)
    nv(::PureFFT.REDFT10_T, x) = (N=length(x); [2*sum(x[j+1]*cos(pi*(2j+1)*k/(2N)) for j in 0:N-1) for k in 0:N-1])
    nv(::PureFFT.RODFT10_T, x) = (N=length(x); [2*sum(x[j+1]*sin(pi*(2j+1)*(k+1)/(2N)) for j in 0:N-1) for k in 0:N-1])
    nv(::PureFFT.REDFT01_T, x) = (N=length(x); [x[1]+2*sum(x[j+1]*cos(pi*j*(2k+1)/(2N)) for j in 1:N-1) for k in 0:N-1])
    nv(::PureFFT.RODFT01_T, x) = (N=length(x); [((-1)^k)*x[N]+2*sum((x[j+1]*sin(pi*(j+1)*(2k+1)/(2N)) for j in 0:N-2); init=0.0) for k in 0:N-1])
    nv(::PureFFT.REDFT00_T, x) = (N=length(x); [x[1]+((-1)^k)*x[N]+2*sum(x[j+1]*cos(pi*j*k/(N-1)) for j in 1:N-2; init=0.0) for k in 0:N-1])
    nv(::PureFFT.RODFT00_T, x) = (N=length(x); [2*sum(x[j+1]*sin(pi*(j+1)*(k+1)/(N+1)) for j in 0:N-1) for k in 0:N-1])
    kinds = ((REDFT10,FFTW.REDFT10),(REDFT01,FFTW.REDFT01),(RODFT10,FFTW.RODFT10),
             (RODFT01,FFTW.RODFT01),(REDFT00,FFTW.REDFT00),(RODFT00,FFTW.RODFT00))
    for T in (Float64, Float32), (pk, fk) in kinds, n in (4, 8, 16, 32)
        PureFFT._use_r2r_codelet(pk, n) || continue          # only the sizes the codelet serves
        x = randn(T, n)
        p = plan_r2r(x, pk)
        @test typeof(p) <: PureFFT.R2RCodeletPlan            # the codelet route IS selected
        y = similar(x); mul!(y, p, x)
        sc = max(1, maximum(abs.(Float64.(x))) * n)
        @test maximum(abs.(Float64.(y) .- FFTW.r2r(Float64.(x), fk))) / sc < tol(T)
        @test maximum(abs.(Float64.(y) .- nv(pk, Float64.(x)))) / sc < tol(T)   # independent naive sum
        @test (@allocated mul!(y, p, x)) == 0               # zero-alloc hot path
    end
end

@testitem "DCT-II parity vs FFTW ≥ 0.96× (even N)" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics, LinearAlgebra
    # `Pkg.test` runs with `--check-bounds=yes`, which overrides @inbounds in PureFFT's Julia
    # loops but not in FFTW's C library — an artificial ~3× handicap to PureFFT in that env.
    # The bench (bench/run_compare_r2r.jl, no forced bounds-checks) is the authoritative
    # measurement: PF/FFTW is 1.45–2.71× for even N 256–65536 (F64+F32 both).
    # Under forced bounds-checks the measurement is unfair, so we skip rather than assert.
    med(b) = median(b.times)
    for T in (Float64, Float32), n in (256, 1024, 4096)
        x = randn(T, n)
        pf = FFTW.plan_r2r(copy(x), FFTW.REDFT10; flags=FFTW.MEASURE)
        pp = plan_r2r(x, REDFT10)
        tf = med(@benchmark $pf * y setup=(y=copy($x)))
        tp = med(@benchmark mul!(y, $pp, $x) setup=(y=similar($x)))
        if Base.JLOptions().check_bounds == 0
            @test tf/tp ≥ 0.96            # fair env: real parity gate, fails on a genuine regression
        else
            @test_skip tf/tp ≥ 0.96      # under forced bounds-checks (Pkg.test default) the measurement is unfair
        end
    end
end

@testitem "DCT-I/DST-I (REDFT00/RODFT00) parity vs FFTW ≥ 0.96× — every n" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics, LinearAlgebra
    # The fix target: DCT-I/DST-I route their inner complex FFT (size n∓1, an odd prime/prime-power/
    # composite) through the AVX odd-prime radix tree (BP leaf + MR3/5/7/9 odd-column tail) instead of
    # the old Bluestein/recursive mis-route. Each size here was RED before (0.47–0.93×); now ≥0.96×.
    # As elsewhere, `Pkg.test` forces --check-bounds=yes (handicaps PureFFT's @inbounds, not FFTW's C):
    # assert only in the fair (bench) env, skip otherwise — same guard as the DCT-II perf item.
    med(b) = median(b.times)
    for (pk, fk) in ((REDFT00, FFTW.REDFT00), (RODFT00, FFTW.RODFT00)), n in (12, 24, 32, 48, 64, 96, 128, 192, 256, 512)
        x = randn(Float64, n)
        pp = plan_r2r(x, pk); y = similar(x)
        pf = FFTW.plan_r2r(copy(x), fk; flags = FFTW.MEASURE)
        tp = med(@benchmark mul!($y, $pp, $x))
        tf = med(@benchmark $pf * z setup = (z = copy($x)))
        if Base.JLOptions().check_bounds == 0
            @test tf / tp ≥ 0.96
        else
            @test_skip tf / tp ≥ 0.96
        end
    end
end

@testitem "small non-pow2 complex parity vs FFTW ≥ 0.96×" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics
    # Small odd primes/prime-powers/composites — the inner sizes feeding DCT-I/DST-I. All previously
    # mis-routed to Bluestein/recursive (0.23–0.69×); now on the AVX odd-prime radix tree. These clear
    # 0.96× (fair env). The very smallest pure-prime / pure-prime-power sizes (7, 13, 25, 49, 125) sit
    # at a documented FFTW tiny-hand-codelet floor (0.77–0.93×) — the tree is still the best PureFFT
    # kernel for them (2–4× over Bluestein/CodeletPlan), and every DCT-I/DST-I transform that USES them
    # clears the gate (above) — so they are tracked as @test_broken, not skipped or hidden.
    med(b) = median(b.times)
    ratio(n) = (x = randn(ComplexF64, n);
        pp = plan_pfft(ComplexF64, n); pf = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE);
        med(@benchmark $pf * y setup = (y = copy($x))) / med(@benchmark PureFFT.apply_unnormalized!($pp, z) setup = (z = copy($x))))
    pass = (11, 17, 19, 23, 33, 35, 45, 55, 63, 65, 77, 91, 95, 129)   # clear the gate
    floor_sizes = (7, 13, 25, 49, 125)                                  # FFTW tiny-codelet floor (tracked)
    if Base.JLOptions().check_bounds == 0
        for n in pass; @test ratio(n) ≥ 0.96; end
        for n in floor_sizes; @test_broken ratio(n) ≥ 0.96; end
    else
        for n in (pass..., floor_sizes...); @test_skip ratio(n) ≥ 0.96; end
    end
end

@testitem "F32 non-pow2 complex parity vs FFTW ≥ 0.96×" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics
    # Float32 non-pow2 sizes routed through the Vec{8,Float32} W=8 tree: small pow2 bases B4/B8/B16/B32/
    # B64W8 (v2≥2) + radix-3/5/7/9 passes. Each was 0.32–0.56× before (scalar recursive/codelet fallback,
    # FFTW-F32 ≈ 2× F64); now ≥0.96× (fair env; canonical bench). Below-gate sizes are tracked, not hidden:
    #  - tiny L1 (12/48/384): FFTW/rust hand codelets win — still ~2× the old fallback, best PureFFT option;
    #  - 120 (0.90) / 3000 (0.92): the documented radix-5 chain/high-power floor;
    #  - v2=1 (54/90/162/270/486/810): NOW GREEN via the W=8 partial-column subsystem (B2W8 base + rem=2
    #    tails on MR3/MR5/MR9; M ≡ 2 mod 4 uniformly) — 1.07–1.48× (radix-9 route).
    med(b) = median(b.times)
    ratio(n) = (x = randn(ComplexF32, n);
        pp = plan_pfft(ComplexF32, n); pf = FFTW.plan_fft!(copy(x); flags = FFTW.MEASURE);
        med(@benchmark $pf * y setup = (y = copy($x))) / med(@benchmark PureFFT.apply_unnormalized!($pp, z) setup = (z = copy($x))))
    pass = (36, 40, 72, 80, 96, 112, 160, 180, 192,      # v2≥2 · {3,5,7,9} — clear the gate (canonical bench)
            224, 240, 360, 448, 480, 720,
            54, 90, 162, 270, 486, 810,                  # v2=1 (2·odd) — W=8 partial-column subsystem
            98,                                          # v2=1 · 7² — radix-7 rem=2 tail (0.22→1.29)
            21, 25, 27, 49, 63, 81, 105, 225, 343, 1225, # v2=0 odd {3,5,7}-smooth — B1 base + odd-M
                                                         # rem∈{1,3} tails (49=7² is the rfft-98/DST-I-48 inner)
            11, 13, 19, 95, 22, 26, 190)                 # D2 bare-prime BP-W8 leaf (11/13/19) + radix-2
                                                         # (MR2W8) for 2·P — 11=1.23 13=1.14 19=2.51 95=2.53
                                                         # 22=1.10 26=0.99 190=2.45 (FFTW; W8≫Codelet/Bluestein)
    floor_sizes = (12, 24, 48, 120, 384, 1500, 3000,     # tiny-L1 (12/24/48/384, FFTW/rust hand codelets win)
                                                         # + radix-5 chain/high-power (120/1500/3000)
                   7, 9, 15, 75)                         # v2=0: tiny-L1 primes 7/9/15 (autoplan→CodeletPlan,
                                                         # FFTW fused codelet ~0.6); 75=3·5² radix-5-chain ~0.94
    if Base.JLOptions().check_bounds == 0
        for n in pass; @test ratio(n) ≥ 0.96; end
        for n in floor_sizes; @test_broken ratio(n) ≥ 0.96; end
    else
        for n in (pass..., floor_sizes...); @test_skip ratio(n) ≥ 0.96; end
    end
end

@testitem "F32 DCT-IV/DST-IV tiny-N codelet parity vs FFTW ≥ 0.96×" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics, LinearAlgebra
    # DCT-IV/DST-IV (REDFT11/RODFT11) had no @generated codelet — they wrapped a size-N complex FFT and
    # lost at tiny N (n=12: 0.80) to FFTW's fused hand codelets. The new full-complex r2r-IV codelet
    # (src/r2r.jl, gated n≤12) lifts n=12 (DCT-IV 1.38, DST-IV 3.87). n=24 stays a proven floor: the
    # codelet (0.82) ≈ the wrap (0.83) — FFTW's fused n=24 codelet wins either way (tracked, not assumed).
    med(b) = median(b.times)
    ratio(pk, fk, n) = (x = randn(Float32, n); pp = plan_r2r(x, pk); pf = FFTW.plan_r2r(copy(x), fk; flags = FFTW.MEASURE); y = similar(x);
        med(@benchmark $pf * z setup = (z = copy($x))) / med(@benchmark PureFFT._apply!($pp, $y, $x)))
    if Base.JLOptions().check_bounds == 0
        @test ratio(REDFT11, FFTW.REDFT11, 12) ≥ 0.96
        @test ratio(RODFT11, FFTW.RODFT11, 12) ≥ 0.96
        @test_broken ratio(REDFT11, FFTW.REDFT11, 24) ≥ 0.96   # FFTW fused n=24 codelet floor (both routes ≈0.82)
        @test_broken ratio(RODFT11, FFTW.RODFT11, 24) ≥ 0.96
    else
        for (pk, fk, n) in ((REDFT11, FFTW.REDFT11, 12), (RODFT11, FFTW.RODFT11, 12),
                            (REDFT11, FFTW.REDFT11, 24), (RODFT11, FFTW.RODFT11, 24))
            @test_skip ratio(pk, fk, n) ≥ 0.96
        end
    end
end

@testitem "F32 DCT-I/DST-I prime-power inner parity vs FFTW ≥ 0.96×" tags=[:perf] begin
    using PureFFT, FFTW, BenchmarkTools, Statistics, LinearAlgebra
    # DCT-I/DST-I route their inner real FFT (plan_prfft, size 2(n∓1)) whose half-size complex inner is a
    # v2=0 ODD prime/prime-power. DST-I n=48 → rfft-98 → complex-49=7²: now served by the W=8 v2=0
    # padding-trick subsystem (B1 base + odd-M rem∈{1,3} tails), 0.80→1.63. The bare primes 11/13/19 (no
    # radix to pad-tail) are now served by the D2 BP-W8 direct prime leaf: DCT-I n=12 (cplx-11, 0.56→1.21),
    # DST-I n=12 (cplx-13, 0.68→1.84), DCT-I n=96 (cplx-95=5·19, 0.74→2.80) — all clear the gate.
    med(b) = median(b.times)
    ratio(pk, fk, n) = (x = randn(Float32, n); pp = plan_r2r(x, pk); pf = FFTW.plan_r2r(copy(x), fk; flags = FFTW.MEASURE); y = similar(x);
        med(@benchmark $pf * z setup = (z = copy($x))) / med(@benchmark mul!($y, $pp, $x)))
    if Base.JLOptions().check_bounds == 0
        @test ratio(RODFT00, FFTW.RODFT00, 48) ≥ 0.96            # inner cplx-49=7² — W=8 padding trick
        @test ratio(REDFT00, FFTW.REDFT00, 12) ≥ 0.96           # inner cplx-11 (bare prime — D2 BP-W8 leaf, 0.56→1.21)
        @test ratio(RODFT00, FFTW.RODFT00, 12) ≥ 0.96           # inner cplx-13 (bare prime — D2 BP-W8 leaf, 0.68→1.84)
        @test ratio(REDFT00, FFTW.REDFT00, 96) ≥ 0.96           # inner cplx-95=5·19 (bare prime 19 — D2, 0.74→2.80)
    else
        for (pk, fk, n) in ((RODFT00, FFTW.RODFT00, 48), (REDFT00, FFTW.REDFT00, 12),
                            (RODFT00, FFTW.RODFT00, 12), (REDFT00, FFTW.REDFT00, 96))
            @test_skip ratio(pk, fk, n) ≥ 0.96
        end
    end
end

@testitem "r2r all 8 kinds — public API bit-exact vs FFTW" begin
    using PureFFT, FFTW, LinearAlgebra
    kinds = ((REDFT00,FFTW.REDFT00),(REDFT01,FFTW.REDFT01),(REDFT10,FFTW.REDFT10),(REDFT11,FFTW.REDFT11),
             (RODFT00,FFTW.RODFT00),(RODFT01,FFTW.RODFT01),(RODFT10,FFTW.RODFT10),(RODFT11,FFTW.RODFT11))
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-3
    for T in (Float64,Float32), (pk,fk) in kinds, n in (2,3,4,5,8,9,16,17)
        x = randn(T,n); ref = T.(FFTW.r2r(Float64.(x), fk))
        sc = max(1, maximum(abs.(Float64.(x)))*n)
        # functional r2r
        @test maximum(abs.(Float64.(r2r(x,pk)) .- Float64.(ref)))/sc < tol(T)
        # plan * x  and  mul!
        p = plan_r2r(x, pk); y = similar(x)
        @test maximum(abs.(Float64.(p*x) .- Float64.(ref)))/sc < tol(T)
        mul!(y, p, x); @test maximum(abs.(Float64.(y) .- Float64.(ref)))/sc < tol(T)
    end
end

@testitem "DST-I (RODFT00) bit-exact vs FFTW + naive + self-inverse (F64+F32, even+odd N)" begin
    using PureFFT, FFTW, ErrorTypes
    tol(::Type{Float64})=1e-12; tol(::Type{Float32})=1f-4
    naive_dst1(x)=(N=length(x); [2*sum(x[j+1]*sin(pi*(j+1)*(k+1)/(N+1)) for j in 0:N-1) for k in 0:N-1])
    for T in (Float64, Float32), n in (1,2,3,4,5,8,9,16,17,32)
        x = randn(T, n)
        y = unwrap(PureFFT.tryr2r(x, RODFT00))
        @test maximum(abs.(y .- FFTW.r2r(x, FFTW.RODFT00)))/max(1,maximum(abs.(x))*n) < tol(T)
        @test maximum(abs.(y .- T.(naive_dst1(Float64.(x)))))/max(1,maximum(abs.(x))*n) < tol(T)  # independent ref
        # RODFT00 self-inverse up to 2(N+1)
        @test maximum(abs.(unwrap(PureFFT.tryr2r(y, RODFT00)) .- 2*(n+1) .* x))/max(1,maximum(abs.(x))*n) < tol(T)
    end
    # inv / \ : self-inverse with 1/(2(N+1)) scale recovers x
    for T in (Float64, Float32), n in (4, 8, 17)
        x = randn(T, n); p = plan_r2r(x, RODFT00)
        @test maximum(abs.((p \ (p*x)) .- x))/max(1,maximum(abs.(x))) < tol(T)
    end
    # tryplan_r2r returns Ok for RODFT00
    @test !ErrorTypes.is_error(PureFFT.tryplan_r2r(randn(8), RODFT00))
    # size < 1 returns Err
    @test ErrorTypes.is_error(PureFFT.tryplan_r2r(Float64[], RODFT00))
end
