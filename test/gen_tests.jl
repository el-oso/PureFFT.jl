# Gated correctness for the @generated AVX building blocks now wired into PureFFT.AvxRadix:
#   * gen_transpose_packed  (src/gen/transpose.jl) — generalizes avx_transpose{5,7,9}_packed
#   * gen_pp_codelet!       (src/gen/colgen.jl)    — the P² codelet; B25/B49 now delegate to it
# Bit-exactness vs a reference DFT is the robust regression guard.
#
# NOTE — the original standalone probes (colgen_tests.jl / gen_transpose_tests.jl) also asserted
# @code_native instruction-count PARITY (generated ≤ hand, Δ≤2). That was validated manually on this
# Julia/LLVM (B25/B49: identical insns; transpose: within 1–2) but is NOT gated here: exact instruction
# counts shift across LLVM versions, so a Δ≤2 assertion would be flaky in CI. Bit-exactness is the
# version-independent invariant; the op-count parity lives in git history (commits 138cb83 / 362658d).
# B25/B49 hand-kernel comparison removed (P0.5): butterfly25!/49! retired; B25/B49 now use gen_pp_codelet!.

@testitem "Generated packed transpose ≡ hand avx_transpose{5,7,9}_packed (bit-exact)" begin
    A = PureFFT.AvxRadix   # SIMD's Vec/getindex are already loaded via PureFFT — no test-env SIMD dep needed
    randv() = A.V4f((randn(), randn(), randn(), randn()))
    lanes(t) = [ntuple(i -> v[i], 4) for v in t]
    hand(rs::NTuple{5, A.V4f}) = A.avx_transpose5_packed(rs...)
    hand(rs::NTuple{7, A.V4f}) = A.avx_transpose7_packed(rs...)
    hand(rs::NTuple{9, A.V4f}) = A.avx_transpose9_packed(rs...)
    @testset "N=$N" for N in (5, 7, 9)
        rs = ntuple(_ -> randv(), N)
        @test lanes(A.gen_transpose_packed(rs)) == lanes(hand(rs))   # exact lane equality
    end
end

@testitem "Generated column-packed codelet (gen_pp_codelet!) ≡ reference DFT (bit-exact)" begin
    A = PureFFT.AvxRadix
    # Same twiddle bundles B25/B49 now build (butterfly25!/49! retired in P0.5; B25/B49 use gen_pp_codelet!).
    colbf_tw(P, fwd)    = ntuple(a -> A.avx_broadcast_twiddle(a, P, fwd), (P - 1) ÷ 2)
    colbf_tw_lo(P, fwd) = map(A.avx_lo, colbf_tw(P, fwd))
    chunk_tw(P, fwd)    = ntuple(g -> ntuple(r -> A.avx_mixedradix_twiddle_chunk(2g - 1, r, P * P, fwd), P - 1),
                                 (P - 1) ÷ 2)
    ndft(x) = [sum(x[j + 1] * cispi(-2 * j * k / length(x)) for j in 0:(length(x) - 1)) for k in 0:(length(x) - 1)]

    @testset "P=$P (n=$(P)²)" for P in (5, 7)
        n = P * P; fwd = true
        tch = chunk_tw(P, fwd); tc = colbf_tw(P, fwd); tcl = colbf_tw_lo(P, fwd)
        x = [ComplexF64(randn(), randn()) for _ in 1:n]
        go = similar(x)
        A.gen_pp_codelet!(go, x, 0, tch, tc, tcl)
        @test maximum(abs.(go .- ndft(x))) / maximum(abs.(ndft(x))) ≤ 1e-12   # ≡ reference DFT
    end
end

@testitem "Generated composite column butterfly (avx_colbf_composite) ≡ hand cb4 (bit-exact)" begin
    # Phase 1 radix-4 PoC: avx_colbf_composite with Val((2,2)) + empty twiddles must reproduce the hand
    # avx_column_butterfly4 EXACTLY (proves the 2-factor leaf-call + interleave + trivial id/rot90 classify).
    A = PureFFT.AvxRadix
    randv() = A.V4f((randn(), randn(), randn(), randn()))
    lanes(t) = [ntuple(i -> v[i], 4) for v in t]
    rot = A._ROT90_FWD
    rs = ntuple(_ -> randv(), 4)
    hand = A.avx_column_butterfly4(rs..., rot)
    gen = A.avx_colbf_composite(rs, Val((2, 2)), (), rot, nothing)
    @test lanes(gen) == lanes(hand)     # exact lane equality (bit-for-bit)
end

@testitem "Composite column butterflies (avx_column_butterfly{8}) ≡ reference DFT (per-lane)" begin
    # Phase 1: the composite column butterflies now forward to avx_colbf_composite. A size-R column
    # butterfly is an independent size-R DFT over the R registers, per complex-lane (V4f = 2 complex:
    # lanes (0,1) and (2,3)), natural output order. Reference DFT is the hand-kernel-independent guard.
    A = PureFFT.AvxRadix
    rot = A._ROT90_FWD
    randv() = A.V4f((randn(), randn(), randn(), randn()))
    col(rs, o) = [Complex(rs[j][o + 1], rs[j][o + 2]) for j in 1:length(rs)]
    dft(c) = [sum(c[j + 1] * cispi(-2 * j * k / length(c)) for j in 0:(length(c) - 1)) for k in 0:(length(c) - 1)]
    function ref(rs)
        R = length(rs); DA = dft(col(rs, 0)); DB = dft(col(rs, 2))
        [A.V4f((real(DA[k]), imag(DA[k]), real(DB[k]), imag(DB[k]))) for k in 1:R]
    end
    maxrel(t, r) = maximum(maximum(abs.(ntuple(i -> t[k][i] - r[k][i], 4))) for k in 1:length(t)) /
                   maximum(maximum(abs.(ntuple(i -> r[k][i], 4))) for k in 1:length(t))
    bcast(idx, len) = A.avx_broadcast_twiddle(idx, len, true)
    @testset "R=8" begin
        rs = ntuple(_ -> randv(), 8)
        @test maxrel(collect(A.avx_column_butterfly8(rs..., rot)), ref(rs)) ≤ 1e-13
    end
    @testset "R=9" begin
        rs = ntuple(_ -> randv(), 9)
        tw1 = bcast(1, 9); tw2 = bcast(2, 9); tw3 = bcast(4, 9); bf3 = bcast(1, 3)
        @test maxrel(collect(A.avx_column_butterfly9(rs..., tw1, tw2, tw3, bf3)), ref(rs)) ≤ 1e-13
    end
    @testset "R=16 (register form, Val((4,4)))" begin
        # 4×4 DIT: tws = (W16^1, W16^3) — the minimal set the classifier consumes (half-plane reduction
        # expresses W16^9 as neg(W16^1)). Validates the full classifier: id/neg/rot90/bf8/mul_complex.
        rs = ntuple(_ -> randv(), 16)
        tws = (bcast(1, 16), bcast(3, 16))
        @test maxrel(collect(A.avx_colbf_composite(rs, Val((4, 4)), tws, rot, nothing)), ref(rs)) ≤ 1e-13
    end
    # Ports (register form): composite leaves (size-8 recursion) + quadrant twiddle reduction. tws built
    # from composite_tws_exponents (the shared slot-order source), so no manual ordering. bf3 for size-3 leaves.
    tws_of(R1, R2) = ntuple(i -> bcast(A.composite_tws_exponents(R1, R2)[i], R1 * R2),
                            length(A.composite_tws_exponents(R1, R2)))
    @testset "R=24 (Val((8,3)) = 8×3, size-8 leaf recursion)" begin
        rs = ntuple(_ -> randv(), 24)
        @test maxrel(collect(A.avx_colbf_composite(rs, Val((8, 3)), tws_of(8, 3), rot, bcast(1, 3))), ref(rs)) ≤ 1e-13
    end
    @testset "R=32 (Val((4,8)) = 4×8, size-8 leaf recursion)" begin
        rs = ntuple(_ -> randv(), 32)
        @test maxrel(collect(A.avx_colbf_composite(rs, Val((4, 8)), tws_of(4, 8), rot, nothing)), ref(rs)) ≤ 1e-13
    end
end

@testitem "Generated radix-M DIT composite codelet (GenPPCompositePlan) ≡ reference DFT + round-trip" begin
    # The composite codelet (src/codelets.jl) runs a register radix-M DIT over the gen_pp P² codelet for
    # n = M·P². autoplan routes the measured-winning family — P ∈ {17,19,23,29,31}, M ∈ {2,4} (≈2–3× FFTW,
    # prior route Bluestein); P ∈ {11,13} and P³ are EXCLUDED. Correctness (fwd vs reference DFT + inverse
    # round-trip) and routing (wins for the family, never routes excluded sizes here).
    P = PureFFT
    ndft(x) = [sum(x[j + 1] * cispi(-2 * j * k / length(x)) for j in 0:(length(x) - 1)) for k in 0:(length(x) - 1)]
    fam = ((17, 2), (19, 2), (23, 2), (29, 2), (31, 2), (17, 4), (19, 4), (23, 4), (29, 4), (31, 4))
    @testset "n=$(M * p * p) (P=$p, M=$M)" for (p, M) in fam
        n = M * p * p
        x = [ComplexF64(randn(), randn()) for _ in 1:n]
        pl = P.GenPPCompositePlan(ComplexF64, n, p, M)
        y = copy(x); P.apply_unnormalized!(pl, y)
        @test maximum(abs.(y .- ndft(x))) / maximum(abs.(ndft(x))) ≤ 1e-12      # ≡ reference DFT
        pli = P.GenPPCompositePlan(ComplexF64, n, p, M; inverse = true)
        P.apply_unnormalized!(pli, y); y ./= n
        @test maximum(abs.(y .- x)) / maximum(abs.(x)) ≤ 1e-12                  # inverse round-trip
        @test P.autoplan(ComplexF64, n) isa P.GenPPCompositePlan                # routing: composite wins
    end
    @testset "excluded n=$n never routes to composite" for n in (242, 484, 338, 676, 121, 1331, 4913)
        @test !(P.autoplan(ComplexF64, n) isa P.GenPPCompositePlan)
    end
end
