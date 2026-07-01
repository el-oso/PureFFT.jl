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
# forms the hand kernels use directly (id / neg / rot90 / √½ bf8); a general W_R^e consumes a `tws` slot,
# REUSED per distinct reduced exponent so equal twiddles share one constant (matches the hand kernels'
# minimal twiddle set). `slotmap` maps a canonical lower-half exponent → its 1-based `tws` slot. Half-plane
# reduction: W^er = −W^(er−R/2) for er in (R/2, R), so the upper half reuses the lower-half constant via neg
# (this is how hand cb16 expresses W16^9 as neg(tw1), keeping tws = (W16^1, W16^3)).
function _twiddle_expr(x, e::Int, R::Int, rot::Symbol, slotmap::Dict{Int, Int})
    er = mod(e, R)
    er == 0   && return x                                     # W^0 = 1
    2er == R  && return Expr(:call, :avx_neg, x)              # e = R/2   → −1
    4er == R  && return Expr(:call, :avx_rotate90, x, rot)    # e = R/4   → −i (rotate90)
    8er == R  && return Expr(:call, :avx_bf8_tw1, x, rot)     # e = R/8   → √½·(rot90(x)+x)
    8er == 3R && return Expr(:call, :avx_bf8_tw3, x, rot)     # e = 3R/8  → √½·(rot90(x)−x)
    if 2er > R                                                # upper half: W^er = −W^(er−R/2)
        s = get!(slotmap, er - R ÷ 2, length(slotmap) + 1)
        Expr(:call, :avx_mul_complex, x, Expr(:call, :avx_neg, :(tws[$s])))
    else                                                      # general W_R^er (reuse slot per distinct er)
        s = get!(slotmap, er, length(slotmap) + 1)
        Expr(:call, :avx_mul_complex, x, :(tws[$s]))
    end
end

# rs: R = R1·R2 input registers (width-generic). Val{(R1,R2)} = the 2-factor split (generate-time). tws =
# the non-trivial inter-stage twiddle constants, in consumption order (empty for radix-4). rot = rotation90
# const; bf3 = the radix-3 twiddle const (each leaf uses only what it needs). Returns the R outputs as a tuple.
@inline @generated function avx_colbf_composite(rs::NTuple{R}, ::Val{F}, tws, rot, bf3) where {R, F}
    R1, R2 = F
    R1 * R2 == R || error("avx_colbf_composite: R1·R2 ($R1·$R2) ≠ R ($R)")
    stmts = Any[]
    slotmap = Dict{Int, Int}()   # canonical lower-half exponent → tws slot (reused per distinct exponent)

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
        tw[g + 1, j] = _twiddle_expr(mid[g + 1, j], g * (j - 1), R, :rot, slotmap)
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
