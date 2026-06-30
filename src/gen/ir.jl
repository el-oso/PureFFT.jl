# Minimal flat-Vector{Node} IR for the codelet generator (genfft analogue).
#
# GENERATE-TIME ONLY — never on the runtime hot path. The frontend builds the SAME split-real
# arithmetic as `_gen_dft_soa_mixed!` (codelets.jl), one CSE pass hash-conses identical nodes, and
# the emitter unparses to a straight-line `Expr` (split-real muladd form). This is the substrate
# for the make-or-break GATE experiment: does explicit CSE beat what LLVM already extracts?
#
# Granularity = real scalars (re/im already split). That is the level where a size-25 DFT has real
# CSE opportunities (cos/sin products shared across conjugate twiddles), so the experiment actually
# tests CSE rather than a no-op. The brief's complex `(wr,wi)` field is preserved in `Node.w`.

# --- node ----------------------------------------------------------------------------------------
const LOADR = 0x01   # a = 1-based input index into xr
const LOADI = 0x02   # a = 1-based input index into xi
const ADD   = 0x03   # t[a] + t[b]
const SUB   = 0x04   # t[a] - t[b]
const MULC  = 0x05   # t[a] * w[1]
const FMA   = 0x06   # muladd(t[a], w[1], t[b])
const FMS   = 0x07   # muladd(t[a], w[1], -t[b])

struct Node
    op::UInt8
    a::Int                 # operand-1 node index (or input slot for loads)
    b::Int                 # operand-2 node index (0 = none)
    w::NTuple{2, Float64}  # baked constant(s); w[1] is the multiplier (w[2] reserved / complex-form)
end

mutable struct Builder
    nodes::Vector{Node}
end
Builder() = Builder(Node[])
emit!(b::Builder, n::Node)::Int = (push!(b.nodes, n); length(b.nodes))

# --- frontend: size-R split-real DFT, identical arithmetic to `_gen_dft_soa_mixed!` --------------
function _spf(R::Int)                       # smallest prime factor (R ≥ 2)
    R % 2 == 0 && return 2
    p = 3
    while p * p <= R
        R % p == 0 && return p
        p += 2
    end
    return R
end

# Build the IR for a size-R DFT of input node-pairs (insr,insi). `s` = -1 fwd / +1 inv.
# Mirrors `_gen_dft_soa_mixed!` op-for-op: add → 2 real ADDs; complex mul → 2 MULC + (FMS,FMA).
function build_soa_dft!(b::Builder, insr::Vector{Int}, insi::Vector{Int}, R::Int, s::Int)
    R == 1 && return (insr, insi)
    p = _spf(R)
    twiddle(e) = Complex{Float64}(cispi(s * 2 * mod(e, R) / R))
    add(ar, ai, br, bi) =
        (emit!(b, Node(ADD, ar, br, (0.0, 0.0))), emit!(b, Node(ADD, ai, bi, (0.0, 0.0))))
    function mul(ar, ai, w)
        isone(w) && return (ar, ai)
        wr = real(w); wi = imag(w)
        p_aiwi = emit!(b, Node(MULC, ai, 0, (wi, 0.0)))     # ai*wi
        r = emit!(b, Node(FMS, ar, p_aiwi, (wr, 0.0)))      # muladd(ar, wr, -(ai*wi))
        p_aiwr = emit!(b, Node(MULC, ai, 0, (wr, 0.0)))     # ai*wr
        i = emit!(b, Node(FMA, ar, p_aiwr, (wi, 0.0)))      # muladd(ar, wi,  ai*wr)
        return (r, i)
    end

    if p == R                                                # prime leaf: direct DFT
        or = Vector{Int}(undef, R); oi = Vector{Int}(undef, R)
        for k in 0:(R - 1)
            ar, ai = insr[1], insi[1]
            for j in 1:(R - 1)
                tr, ti = mul(insr[j + 1], insi[j + 1], twiddle(j * k))
                ar, ai = add(ar, ai, tr, ti)
            end
            or[k + 1] = ar; oi[k + 1] = ai
        end
        return (or, oi)
    end

    m = R ÷ p
    Gr = Vector{Vector{Int}}(undef, p); Gi = Vector{Vector{Int}}(undef, p)
    for a in 0:(p - 1)
        Gr[a + 1], Gi[a + 1] = build_soa_dft!(
            b, Int[insr[a + p * j + 1] for j in 0:(m - 1)],
               Int[insi[a + p * j + 1] for j in 0:(m - 1)], m, s)
    end
    or = Vector{Int}(undef, R); oi = Vector{Int}(undef, R)
    for k in 0:(R - 1)
        r = k % m; ar, ai = Gr[1][r + 1], Gi[1][r + 1]
        for a in 1:(p - 1)
            tr, ti = mul(Gr[a + 1][r + 1], Gi[a + 1][r + 1], twiddle(a * k))
            ar, ai = add(ar, ai, tr, ti)
        end
        or[k + 1] = ar; oi[k + 1] = ai
    end
    return (or, oi)
end

# Build the full size-R IR + return (nodes, outr, outi). insr/insi load slots = 1..R.
function build_dft_ir(R::Int, s::Int)
    b = Builder()
    insr = Int[emit!(b, Node(LOADR, j, 0, (0.0, 0.0))) for j in 1:R]
    insi = Int[emit!(b, Node(LOADI, j, 0, (0.0, 0.0))) for j in 1:R]
    outr, outi = build_soa_dft!(b, insr, insi, R, s)
    return (b.nodes, outr, outi)
end

# --- CSE pass: Vector{Node} → Vector{Node} (hash-cons identical nodes) ----------------------------
# Same op + same (canonicalised) operands + same constant ⇒ one node. Loads keep their input slot.
function cse(old::Vector{Node})
    seen = Dict{Node, Int}()
    new = Node[]
    remap = Vector{Int}(undef, length(old))
    for i in eachindex(old)
        n = old[i]
        canon = if n.op == LOADR || n.op == LOADI
            n
        else
            Node(n.op, remap[n.a], n.b == 0 ? 0 : remap[n.b], n.w)
        end
        idx = get(seen, canon, 0)
        if idx == 0
            push!(new, canon); idx = length(new); seen[canon] = idx
        end
        remap[i] = idx
    end
    return new, remap
end

# --- emitter: Vector{Node} → Expr (split-real, muladd form) ---------------------------------------
_sym(i::Int) = Symbol("t", i)

function emit_block(nodes::Vector{Node}, outr::Vector{Int}, outi::Vector{Int})
    stmts = Any[]
    for i in eachindex(nodes)
        n = nodes[i]; t = _sym(i)
        e = if n.op == LOADR
            :(@inbounds xr[$(n.a)])
        elseif n.op == LOADI
            :(@inbounds xi[$(n.a)])
        elseif n.op == ADD
            :($(_sym(n.a)) + $(_sym(n.b)))
        elseif n.op == SUB
            :($(_sym(n.a)) - $(_sym(n.b)))
        elseif n.op == MULC
            :($(_sym(n.a)) * $(n.w[1]))
        elseif n.op == FMA
            :(muladd($(_sym(n.a)), $(n.w[1]), $(_sym(n.b))))
        else # FMS
            :(muladd($(_sym(n.a)), $(n.w[1]), -$(_sym(n.b))))
        end
        push!(stmts, :($t = $e))
    end
    for k in eachindex(outr)
        push!(stmts, :(@inbounds outr[$k] = $(_sym(outr[k]))))
        push!(stmts, :(@inbounds outi[$k] = $(_sym(outi[k]))))
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end
