# Composite-radix column-butterfly GENERATOR (Phase 1 — systematize the hand avx_column_butterfly{4,8,9,
# 12,16,…}). Every hand composite column butterfly is a strictly-2-factor DIT:
#
#   stage 1:  R2 sub-butterflies of size R1, down the columns  (rs[g + k·R2], k=0..R1-1)
#   twiddle:  mid[g][j] *= W_R^(g·(j-1))          (inter-stage)
#   stage 2:  R1 sub-butterflies of size R2, across            (the twiddled g-column at each j)
#   interleave: out[l·R1 + j] = o_j[l]                          (l=0..R2-1, j=1..R1)
#
# So ONE size-parameterized @generated that (a) calls the size-R1/R2 leaf butterflies and (b) *classifies*
# each inter-stage twiddle g·(j-1) at generate time emits the BYTE-IDENTICAL instruction stream of the
# hand kernel by construction — par is near-free (same DAG, not a re-derivation), exactly as gen_pp_codelet!
# proved for the prime side. Verified by-hand vs cb4/cb8/cb9 (interleave + classify reproduce them exactly).
#
# Width-generic: the leaf butterflies + avx_rotate90/avx_mul_complex are all width-dispatched (V4f / Vec{8}),
# so the SAME body lights up the Vec{8} (AVX-512) path — no separate W=8 generator.
#
# Generate-time only: the body emits straight-line code with LITERAL tuple indices (CLAUDE.md rule 1 — never
# index a tuple with a runtime variable). The _leaf_call / _twiddle_expr helpers below run at specialization
# time (they build Exprs); Vector{Any} there is generate-time scratch, not runtime (trim-safe).

# --- generate-time helpers (run inside the @generated body; build Exprs) ---

# Emit the size-r leaf butterfly over the r input Exprs `ins`, as an r-tuple call. `rot`/`bf3` are the
# generator's arg symbols (the leaf that doesn't need one simply ignores it).
function _leaf_call(r::Int, ins::Vector, rot::Symbol, bf3::Symbol)
    if r == 2
        Expr(:call, :avx_butterfly2, ins...)
    elseif r == 3
        Expr(:call, :avx_column_butterfly3, ins..., bf3)
    elseif r == 4
        Expr(:call, :avx_column_butterfly4, ins..., rot)
    else
        error("avx_colbf_composite: leaf size r=$r not supported (have 2,3,4; composite recursion is Phase-1 later)")
    end
end

# Classify the inter-stage twiddle W_R^e applied to input Expr `x`, at generate time. Emits the trivial
# forms the hand kernels use (id / rot90) directly; a general W_R^e consumes the next `tws` slot. `slot` is
# a Ref{Int} counting consumed non-trivial twiddles (so callers thread the literal index). PoC (radix-4)
# only reaches id + rot90; radix-8/9/16 extend this (√½ bf8 forms, mul_complex constants).
function _twiddle_expr(x, e::Int, R::Int, rot::Symbol, slot::Ref{Int})
    er = mod(e, R)
    if er == 0
        x                                   # W^0 = 1
    elseif 4er == R                         # e = R/4 → W^(R/4) = -i = rotate90
        Expr(:call, :avx_rotate90, x, rot)
    else
        error("avx_colbf_composite: twiddle e=$e (R=$R) not yet classified (PoC handles id, R/4; " *
              "bf8/neg/mul_complex land with radix-8/9/16)")
    end
end

# rs: R = R1·R2 input registers (width-generic). Val{(R1,R2)} = the 2-factor split (generate-time). tws =
# the non-trivial inter-stage twiddle constants, in consumption order (empty for radix-4). rot = rotation90
# const; bf3 = the radix-3 twiddle const (each leaf uses only what it needs). Returns the R outputs as a tuple.
@inline @generated function avx_colbf_composite(rs::NTuple{R}, ::Val{F}, tws, rot, bf3) where {R, F}
    R1, R2 = F
    R1 * R2 == R || error("avx_colbf_composite: R1·R2 ($R1·$R2) ≠ R ($R)")
    stmts = Any[]
    slot = Ref(0)

    # stage 1: R2 leaves of size R1. group g (0-based) over rs[g + k·R2 + 1], k=0..R1-1 → mid[g+1][j].
    mid = Matrix{Symbol}(undef, R2, R1)
    for g in 0:R2 - 1
        ins = Any[:(rs[$(g + k * R2 + 1)]) for k in 0:R1 - 1]
        outs = [gensym("m$(g)_$j") for j in 1:R1]
        push!(stmts, :(($(outs...),) = $(_leaf_call(R1, ins, :rot, :bf3))))
        for j in 1:R1
            mid[g + 1, j] = outs[j]
        end
    end

    # inter-stage twiddle: mid[g+1][j] *= W_R^(g·(j-1)).
    tw = Matrix{Any}(undef, R2, R1)
    for g in 0:R2 - 1, j in 1:R1
        tw[g + 1, j] = _twiddle_expr(mid[g + 1, j], g * (j - 1), R, :rot, slot)
    end

    # stage 2: R1 leaves of size R2. group j over the g-column tw[*, j] → o[j][l].
    o = Matrix{Symbol}(undef, R1, R2)
    for j in 1:R1
        ins = Any[tw[g + 1, j] for g in 0:R2 - 1]
        outs = [gensym("o$(j)_$l") for l in 1:R2]
        push!(stmts, :(($(outs...),) = $(_leaf_call(R2, ins, :rot, :bf3))))
        for l in 1:R2
            o[j, l] = outs[l]
        end
    end

    # interleave: out[l·R1 + j] = o[j][l+1].
    outtuple = Vector{Symbol}(undef, R)
    for l in 0:R2 - 1, j in 1:R1
        outtuple[l * R1 + j] = o[j, l + 1]
    end

    quote
        @inbounds begin
            $(stmts...)
        end
        ($(outtuple...),)
    end
end
