# Column-packed prime-power (P²) codelet GENERATOR (probe — NOT wired into kernels/autoplan).
#
# Composes the three already-parity-validated column-packed components into ONE size-parameterized
# @generated codelet, generalizing the hand-written butterfly25! (5²) / butterfly49! (7²)
# (src/avxradix/kernels.jl) to ANY odd prime P. The emitted body is byte-for-byte the same shape as
# butterfly25!'s dual-width structure:
#
#   load  (col 0 as P partial V2f, plus H=(P-1)/2 column-pair groups, each P×V4f)
#   → P-pt column butterflies down the rows   (avx_colbf_prime, width-generic: V2f for col 0, V4f groups)
#   → inter-stage twiddle multiply            (avx_mul_complex × MR twiddle chunks, rows 2..P only)
#   → dual-width packed transpose             (generated here — generalizes avx_transpose_5x5/7x7)
#   → P-pt column butterflies across          (avx_colbf_prime)
#   → natural-order store  X[P·k2 + k1]
#
# n = P². twcol/twcol_lo are the shared column-butterfly broadcast twiddles W_P^1..W_P^H (V4f / V2f);
# twchunk[g] are the per-group mixed-radix twiddle chunks (chunk(2g-1, r, P²)). These are exactly the
# fields the hand B25/B49 structs carry (recursive.jl): twcol=(t0,t1[,t2]), twcol_lo=(t0lo,..),
# twchunk=(tw1,tw2[,tw3]).
#
# NOTE on the transpose: butterfly25!/49! use the DUAL-WIDTH transpose avx_transpose_5x5/7x7 (col 0 is
# a half-width V2f set), NOT the full-width gen_transpose_packed (src/gen/transpose.jl, which generalizes
# the all-V4f avx_transpose5/7_packed). To compare like-for-like with the hand dual-width kernels this
# generator emits the dual-width network. gen_transpose_packed already proved the full-width network is
# generatable at parity (commit 138cb83); this proves the same for the dual-width one the prime-power
# codelets actually use.
#
# Generate-time only: the @generated body emits straight-line code with LITERAL tuple indices (CLAUDE.md
# rule 1 — never index a tuple with a runtime variable). Emitted code is concrete / trim-safe. Needs the
# avxradix AVX2 primitives (avx_colbf_prime, avx_mul_complex, avx_transpose_2x2, avx_merge, avx_lo,
# avx_hi, avx_load/store_*) in scope.

@inline @generated function gen_pp_codelet!(out, inp, base::Int,
                                    twchunk::NTuple{H, NTuple{M, V4f}},
                                    twcol::NTuple{H, V4f}, twcol_lo::NTuple{H, V2f}) where {H, M}
    P = 2H + 1
    M == P - 1 || error("gen_pp_codelet!: M ($M) must equal P-1 ($(P - 1)) for P=$P")
    s = Any[]

    # ---- 1. loads: col 0 partial (P×V2f) + H column-pair groups (P×V4f each) ----
    for j in 1:P
        push!(s, :($(Symbol(:a, j)) = avx_load_partial1(inp, base + $(P * (j - 1)))))
    end
    for g in 1:H, j in 1:P
        push!(s, :($(Symbol(:b, g, :_, j)) = avx_load_complex(inp, base + $(P * (j - 1) + (2g - 1)))))
    end

    # ---- 2. column butterflies down the P rows (shared twcol / twcol_lo) ----
    # call-site @inline: avx_colbf_prime is @generated but not @inline; the hand kernel's
    # avx_column_butterfly5/7 IS @inline, so force inlining here for a like-for-like body.
    push!(s, :(mid0 = @inline avx_colbf_prime($(Expr(:tuple, [Symbol(:a, j) for j in 1:P]...)), twcol_lo)))
    for g in 1:H
        push!(s, :($(Symbol(:mid, g)) =
            @inline avx_colbf_prime($(Expr(:tuple, [Symbol(:b, g, :_, j) for j in 1:P]...)), twcol)))
    end

    # ---- 3. inter-stage twiddle: col 0 untouched; group g rows 2..P scaled by twchunk[g] ----
    for j in 1:P
        push!(s, :($(Symbol(:c0_, j)) = mid0[$j]))
    end
    for g in 1:H
        push!(s, :($(Symbol(:c, g, :_, 1)) = $(Symbol(:mid, g))[1]))
        for j in 2:P
            push!(s, :($(Symbol(:c, g, :_, j)) =
                avx_mul_complex($(Symbol(:mid, g))[$j], twchunk[$g][$(j - 1)])))
        end
    end

    # ---- 4. dual-width packed transpose (generalizes avx_transpose_5x5/7x7) ----
    # T0 (P×V2f): col 0 of every row, spread out — c0_1, then lo/hi of each group's row 1.
    t0_elems = Any[Symbol(:c0_, 1)]
    for g in 1:H
        push!(t0_elems, :(avx_lo($(Symbol(:c, g, :_, 1)))))
        push!(t0_elems, :(avx_hi($(Symbol(:c, g, :_, 1)))))
    end
    push!(s, :(o0 = @inline avx_colbf_prime($(Expr(:tuple, t0_elems...)), twcol_lo)))
    # output group jblk (P×V4f): merge(col0 row-pair) + each group's 2×2-transposed row-pair.
    for jblk in 1:H
        elems = Any[:(avx_merge($(Symbol(:c0_, 2jblk)), $(Symbol(:c0_, 2jblk + 1))))]
        for g in 1:H
            tp = Symbol(:tp, jblk, :_, g)
            push!(s, :($tp = avx_transpose_2x2($(Symbol(:c, g, :_, 2jblk)), $(Symbol(:c, g, :_, 2jblk + 1)))))
            push!(elems, :($tp[1])); push!(elems, :($tp[2]))
        end
        # ---- 5. column butterflies across (the transposed rows) ----
        push!(s, :($(Symbol(:o, jblk)) = @inline avx_colbf_prime($(Expr(:tuple, elems...)), twcol)))
    end

    # ---- 6. natural-order store X[P·k + g] ----
    for k in 0:(P - 1)
        push!(s, :(avx_store_partial1!(out, base + $(P * k), o0[$(k + 1)])))
        for jblk in 1:H
            push!(s, :(avx_store_complex!(out, base + $(P * k + (2jblk - 1)), $(Symbol(:o, jblk))[$(k + 1)])))
        end
    end

    quote
        @inbounds begin
            $(s...)
        end
        return nothing
    end
end
