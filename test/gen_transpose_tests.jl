# Parity probe: generated packed transpose (src/gen/transpose.jl) vs hand-written
# avx_transpose{5,7,9}_packed (src/avxradix/avxport.jl). NOT wired into kernels.
#
# (a) bit-exact: generated ≡ hand-written on random input (exact equality of all lanes).
# (b) quality:  @code_native shuffle/permute + total instruction counts, generated vs hand.
#
# Run standalone:  julia --project test/gen_transpose_tests.jl

module GenTransposeProbe

using Test, SIMD, InteractiveUtils

# Pull the AVX2 primitives + hand-written transposes (defined at file scope) into this module,
# then the generator on top — both see avx_unpacklo_complex / _blend03 / V4f.
include(joinpath(@__DIR__, "..", "src", "avxradix", "avxport.jl"))
include(joinpath(@__DIR__, "..", "src", "gen", "transpose.jl"))

randv() = V4f((randn(), randn(), randn(), randn()))
lanes(t) = [Tuple(v) for v in t]

# --- (b) instruction counting from @code_native ---
const SHUF_RE = r"\b(vshuf|vperm|vunpck|vblend|vpermil|vinsertf|vextractf|vmov[lh]|vbroadcast)"i

function native_str(f, argtypes)
    io = IOBuffer()
    code_native(io, f, argtypes; debuginfo = :none, syntax = :att)
    String(take!(io))
end

function count_insns(s)
    total = 0; shuf = 0
    for ln in split(s, '\n')
        t = strip(ln)
        (isempty(t) || startswith(t, '.') || startswith(t, '#') ||
         startswith(t, ';') || endswith(t, ':')) && continue
        mnem = first(split(t))
        startswith(mnem, "ret") || startswith(mnem, "push") ||
            startswith(mnem, "pop") || startswith(mnem, "nop") && continue
        total += 1
        occursin(SHUF_RE, mnem) && (shuf += 1)
    end
    (shuf, total)
end

# Hand wrappers as tuple-takers so gen and hand share ONE harness (the @generated gen already takes
# a tuple). For (a) bit-exactness we call these directly.
hand_packed(rs::NTuple{5, V4f}) = avx_transpose5_packed(rs...)
hand_packed(rs::NTuple{7, V4f}) = avx_transpose7_packed(rs...)
hand_packed(rs::NTuple{9, V4f}) = avx_transpose9_packed(rs...)

# Realistic INLINED harness mirroring kernel use: the transpose sits between in-register producers
# (a complex-multiply per row — like a column-butterfly's twiddled output) and an in-register
# consumer (combine the transposed rows), so the shuffle network actually materializes in registers
# rather than being elided into pure memory moves. `TF` is a compile-time-constant function ⇒ fully
# inlined; the body is @generated-unrolled with literal indices (CLAUDE.md rule 1). gen and hand run
# the byte-identical harness, so any instruction-count delta is purely the transpose body.
@inline @generated function harness!(::Val{TF}, out, inp, tw, ::Val{N}) where {TF, N}
    s = Any[]
    for i in 1:N   # producers: keep values register-resident via a complex multiply
        push!(s, :($(Symbol(:r, i)) = avx_mul_complex(avx_load_complex(inp, $(2 * (i - 1))), tw)))
    end
    push!(s, :(t = $TF(($([Symbol(:r, i) for i in 1:N]...),))))
    acc = :(t[1])                       # consumer: dependent complex-multiply chain over the rows.
    for i in 2:N                        # nonlinear + LLVM won't reassociate FP ⇒ can't DCE the
        acc = :(avx_mul_complex($acc, t[$i]))   # permutation; forces the shuffle network into regs.
    end
    push!(s, :(avx_store_complex!(out, 0, $acc)))
    push!(s, :(return nothing))
    Expr(:block, s...)
end
@noinline run_gen!(out, inp, tw, ::Val{N})  where {N} = harness!(Val(gen_transpose_packed), out, inp, tw, Val(N))
@noinline run_hand!(out, inp, tw, ::Val{N}) where {N} = harness!(Val(hand_packed), out, inp, tw, Val(N))

const CASES = [5, 7, 9]

@testset "generated packed transpose parity" begin
    for N in CASES
        rs = ntuple(_ -> randv(), N)
        g = gen_transpose_packed(rs); h = hand_packed(rs)
        @test lanes(g) == lanes(h)                      # (a) bit-exact (exact lane equality)

        T = Tuple{Vector{ComplexF64}, Vector{ComplexF64}, V4f, Val{N}}
        gs, gt = count_insns(native_str(run_gen!, T))
        hs, ht = count_insns(native_str(run_hand!, T))
        println("N=$N  bitexact=$(lanes(g)==lanes(h))  shuffle gen/hand=$gs/$hs  total gen/hand=$gt/$ht")
        @test abs(gs - hs) <= 2                          # (b) shuffle parity within ~1-2
        @test abs(gt - ht) <= 2                          # total parity within ~1-2
    end
end

end # module
