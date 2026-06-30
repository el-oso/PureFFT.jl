# Gated correctness for the @generated AVX building blocks now wired into PureFFT.AvxRadix:
#   * gen_transpose_packed  (src/gen/transpose.jl) — generalizes avx_transpose{5,7,9}_packed
#   * gen_pp_codelet!       (src/gen/colgen.jl)    — generalizes hand butterfly25!/49!
# Bit-exactness vs the hand-written kernels (and vs a reference DFT) is the robust regression guard.
#
# NOTE — the original standalone probes (colgen_tests.jl / gen_transpose_tests.jl) also asserted
# @code_native instruction-count PARITY (generated ≤ hand, Δ≤2). That was validated manually on this
# Julia/LLVM (B25/B49: identical insns; transpose: within 1–2) but is NOT gated here: exact instruction
# counts shift across LLVM versions, so a Δ≤2 assertion would be flaky in CI. Bit-exactness is the
# version-independent invariant; the op-count parity lives in git history (commits 138cb83 / 362658d).

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

@testitem "Generated column-packed codelet ≡ hand butterfly25!/49! and reference DFT (bit-exact)" begin
    A = PureFFT.AvxRadix
    # Same twiddle bundles the B25/B49 structs build, generalized to any odd prime P.
    colbf_tw(P, fwd)    = ntuple(a -> A.avx_broadcast_twiddle(a, P, fwd), (P - 1) ÷ 2)
    colbf_tw_lo(P, fwd) = map(A.avx_lo, colbf_tw(P, fwd))
    chunk_tw(P, fwd)    = ntuple(g -> ntuple(r -> A.avx_mixedradix_twiddle_chunk(2g - 1, r, P * P, fwd), P - 1),
                                 (P - 1) ÷ 2)
    hand!(::Val{5}, out, inp, tch, tc, tcl) =
        A.butterfly25!(out, inp, 0, tch[1], tch[2], tc[1], tc[2], tcl[1], tcl[2])
    hand!(::Val{7}, out, inp, tch, tc, tcl) =
        A.butterfly49!(out, inp, 0, tch[1], tch[2], tch[3], tc[1], tc[2], tc[3], tcl[1], tcl[2], tcl[3])
    ndft(x) = [sum(x[j + 1] * cispi(-2 * j * k / length(x)) for j in 0:(length(x) - 1)) for k in 0:(length(x) - 1)]

    @testset "P=$P (n=$(P)²)" for P in (5, 7)
        n = P * P; fwd = true
        tch = chunk_tw(P, fwd); tc = colbf_tw(P, fwd); tcl = colbf_tw_lo(P, fwd)
        x = [ComplexF64(randn(), randn()) for _ in 1:n]
        go = similar(x); ho = similar(x)
        A.gen_pp_codelet!(go, x, 0, tch, tc, tcl)
        hand!(Val(P), ho, x, tch, tc, tcl)
        @test maximum(abs.(go .- ho)) / maximum(abs.(ho)) ≤ 1e-13     # generator ≡ hand kernel
        @test maximum(abs.(go .- ndft(x))) / maximum(abs.(ndft(x))) ≤ 1e-12   # ≡ reference DFT
    end
end
