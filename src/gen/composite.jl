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
    elseif r == 8
        Expr(:call, :avx_column_butterfly8, ins..., rot)   # composite leaf: the size-8 butterfly (itself a
                                                           # generator forward, Val((4,2)), trivial inner tws)
    else
        error("avx_colbf_composite: leaf size r=$r not supported (have 2,3,4,8; size-9 leaf for radix-27 " *
              "needs inner W9 twiddles — added when 27 is ported)")
    end
end

# Canonical `tws` slot key for the inter-stage twiddle W_R^e: `nothing` when the twiddle is constant-free
# (id / rot90 / neg / bf8 — a decoration of x needing no stored constant), else the BASE exponent in
# (0, R/4) whose W_R constant is needed. QUADRANT reduction (for 4|R): write er = q·(R/4) + r; every
# quadrant reuses the base-region constant W_R^r via a rot90/neg decoration (W^(R/4)=−i, W^(R/2)=−1,
# W^(3R/4)=+i). r=0 (multiple of R/4) and r=R/8 (bf8 form) are constant-free. This reproduces the hand
# kernels' minimal sets exactly (cb32→{1,2,3,5,6,7}, cb16→{1,3}). Odd R (radix-9/27): no quadrant
# structure → each distinct nonzero exponent is its own constant (R/2→neg when R even-but-not-4|R).
# ONE source of truth, shared by _twiddle_expr (slot assignment) and composite_tws_exponents (runtime list).
function _tws_key(e::Int, R::Int)
    er = mod(e, R)
    er == 0 && return nothing
    if R % 4 == 0
        r = er % (R ÷ 4)
        (r == 0 || r == R ÷ 8) && return nothing     # multiple of R/4 (rot90/neg) or R/8 offset (bf8)
        r                                             # base constant W_R^r, r in (0,R/4)\{R/8}
    else
        2er == R ? nothing : er                       # R/2 → neg; else a distinct constant (odd radix)
    end
end

# The ordered distinct twiddle exponents the generator consumes as `tws` slots for a 2-factor split
# R=R1·R2, in the (g outer, j inner) order the generator visits them. A forwarder builds its `tws` from
# THIS list — `ntuple(i -> avx_broadcast_twiddle(exps[i], R1*R2, fwd), length(exps))` — so slot order
# always matches. (Trivial twiddles produce no slot; upper-half exponents reuse a lower-half slot.)
function composite_tws_exponents(R1::Int, R2::Int)
    R = R1 * R2
    seen = Set{Int}()
    exps = Int[]
    for g in 0:R2 - 1, j in 1:R1
        k = _tws_key(g * (j - 1), R)
        if !isnothing(k) && !(k in seen)
            push!(seen, k); push!(exps, k)
        end
    end
    exps
end

# Emit the inter-stage twiddle W_R^e applied to input Expr `x`, at generate time. Uses the quadrant
# decomposition (see _tws_key): constant-free forms (id/rot90/neg/bf8) emit inline; a base constant is
# decorated per quadrant (·1, ·rot90, ·neg, ·neg∘rot90). `mul` = avx_mul_complex(x, decorated-const).
function _twiddle_expr(x, e::Int, R::Int, rot::Symbol, slotmap::Dict{Int, Int})
    er = mod(e, R)
    er == 0 && return x                                       # W^0 = 1
    mul(c) = Expr(:call, :avx_mul_complex, x, c)              # x · (decorated constant)
    r90(v) = Expr(:call, :avx_rotate90, v, rot)              # · −i
    neg(v) = Expr(:call, :avx_neg, v)                        # · −1
    if R % 4 == 0
        Q = R ÷ 4; q = er ÷ Q; r = er % Q
        if r == 0                                             # pure quadrant rotation OF x (no constant)
            q == 1 && return r90(x)                          # W^(R/4)  = −i
            q == 2 && return neg(x)                          # W^(R/2)  = −1
            q == 3 && return neg(r90(x))                     # W^(3R/4) = +i
        elseif r == Q ÷ 2                                     # r = R/8 → bf8 form (no constant)
            q == 0 && return Expr(:call, :avx_bf8_tw1, x, rot)   # W^(R/8)
            q == 1 && return Expr(:call, :avx_bf8_tw3, x, rot)   # W^(3R/8)
            error("avx_colbf_composite: W^(5R/8)/W^(7R/8) bf8 forms not needed by in-scope ports (e=$e R=$R)")
        else                                                 # base constant W_R^r, decorated by quadrant
            c = :(tws[$(get!(slotmap, r, length(slotmap) + 1))])
            q == 0 && return mul(c)
            q == 1 && return mul(r90(c))                     # W^(R/4+r)  = −i·W^r
            q == 2 && return mul(neg(c))                     # W^(R/2+r)  = −W^r
            q == 3 && return mul(neg(r90(c)))                # W^(3R/4+r) = +i·W^r
        end
    else                                                     # odd R (no quadrant): distinct constants; R/2→neg
        2er == R && return neg(x)
        return mul(:(tws[$(get!(slotmap, er, length(slotmap) + 1))]))
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
