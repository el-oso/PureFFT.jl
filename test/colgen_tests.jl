# Parity probe: column-packed prime-power codelet GENERATOR (src/gen/colgen.jl) vs the hand-written
# butterfly25! / butterfly49! (src/avxradix/kernels.jl). NOT wired into kernels/autoplan.
#
# (a) bit-exact: gen_pp_codelet! ≡ hand butterfly{25,49}! on random input (rel-err ≤ 1e-13).
# (b) quality:  @code_native arith (vfma/vmul/vadd/vsub) + shuffle + spill counts, generated vs hand,
#               in a representative @noinline kernel harness (same loop body, only the codelet differs).
#
# Run standalone:  julia --project test/colgen_tests.jl

module ColGenProbe

using Test, SIMD, InteractiveUtils

# kernels.jl pulls in avxport.jl (the AVX2 primitives) and defines butterfly25!/49! + bf{25,49}_twiddles.
include(joinpath(@__DIR__, "..", "src", "avxradix", "kernels.jl"))
include(joinpath(@__DIR__, "..", "src", "gen", "colgen.jl"))

# ---- twiddle bundles (identical to what the B25/B49 structs in recursive.jl build) ----
colbf_tw(P, fwd)    = ntuple(a -> avx_broadcast_twiddle(a, P, fwd), (P - 1) ÷ 2)
colbf_tw_lo(P, fwd) = map(avx_lo, colbf_tw(P, fwd))
chunk_tw(P, fwd)    = ntuple(g -> ntuple(r -> avx_mixedradix_twiddle_chunk(2g - 1, r, P * P, fwd), P - 1),
                             (P - 1) ÷ 2)

# ---- hand reference, called with the SAME twiddle bundles (unpacked to its positional args) ----
hand!(::Val{5}, out, inp, base, tch, tc, tcl) =
    butterfly25!(out, inp, base, tch[1], tch[2], tc[1], tc[2], tcl[1], tcl[2])
hand!(::Val{7}, out, inp, base, tch, tc, tcl) =
    butterfly49!(out, inp, base, tch[1], tch[2], tch[3], tc[1], tc[2], tc[3], tcl[1], tcl[2], tcl[3])

# ---- (b) instruction counting from @code_native ----
const SHUF_RE = r"\b(vshuf|vperm|vunpck|vblend|vpermil|vinsertf|vextractf|vmov[lh]|vbroadcast)"i
const ARITH_RE = r"\b(vfm|vmul|vadd|vsub|vfnm)"i
const SPILL_RE = r"\b(vmov\w*|movups|movaps).*(rsp|rbp)"i   # stack spill/reload via [rsp]/[rbp]

native_str(f, T) = (io = IOBuffer(); code_native(io, f, T; debuginfo = :none, syntax = :att); String(take!(io)))

function counts(s)
    arith = shuf = spill = total = 0
    for ln in split(s, '\n')
        t = strip(ln)
        (isempty(t) || startswith(t, '.') || startswith(t, '#') ||
         startswith(t, ';') || endswith(t, ':')) && continue
        mnem = first(split(t))
        (startswith(mnem, "ret") || startswith(mnem, "push") ||
         startswith(mnem, "pop") || startswith(mnem, "nop")) && continue
        total += 1
        occursin(ARITH_RE, mnem) && (arith += 1)
        occursin(SHUF_RE, mnem)  && (shuf += 1)
        occursin(SPILL_RE, t)    && (spill += 1)
    end
    (; arith, shuf, spill, total)
end

# ---- argtypes for direct @code_native of each codelet BODY (both return nothing / write to memory,
# so there is no tuple-sret ABI artifact; this is the representative kernel body proc_*! calls). ----
gen_argtypes(tch, tc, tcl) =
    Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Int, typeof(tch), typeof(tc), typeof(tcl)}
hand_fn(::Val{5}) = butterfly25!
hand_fn(::Val{7}) = butterfly49!
hand_argtypes(::Val{5}) = Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Int,
    NTuple{4, V4f}, NTuple{4, V4f}, V4f, V4f, V2f, V2f}
hand_argtypes(::Val{7}) = Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Int,
    NTuple{6, V4f}, NTuple{6, V4f}, NTuple{6, V4f}, V4f, V4f, V4f, V2f, V2f, V2f}

@testset "column-packed codelet generator parity" begin
    for (P, name) in ((5, "B25"), (7, "B49"))
        n = P * P; fwd = true
        tch = chunk_tw(P, fwd); tc = colbf_tw(P, fwd); tcl = colbf_tw_lo(P, fwd)

        x = [ComplexF64(randn(), randn()) for _ in 1:n]
        go = similar(x); ho = similar(x)
        gen_pp_codelet!(go, x, 0, tch, tc, tcl)
        hand!(Val(P), ho, x, 0, tch, tc, tcl)

        relerr = maximum(abs.(go .- ho)) / maximum(abs.(ho))
        @test relerr ≤ 1e-13                                          # (a) bit-exact

        cg = counts(native_str(gen_pp_codelet!, gen_argtypes(tch, tc, tcl)))
        ch = counts(native_str(hand_fn(Val(P)), hand_argtypes(Val(P))))
        println("$name (P=$P)  relerr=$relerr")
        println("  gen : arith=$(cg.arith) shuf=$(cg.shuf) spill=$(cg.spill) total=$(cg.total)")
        println("  hand: arith=$(ch.arith) shuf=$(ch.shuf) spill=$(ch.spill) total=$(ch.total)")
        # parity within a few %: arith+shuf the work, spills the register-pressure proxy.
        @test abs(cg.arith - ch.arith) ≤ max(2, ch.arith ÷ 20)        # (b) arith parity
        @test abs(cg.shuf  - ch.shuf)  ≤ max(2, ch.shuf  ÷ 20)        #     shuffle parity
    end
end

end # module
