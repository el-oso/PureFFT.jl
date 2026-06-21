# AvxMixedRadixPlan — wraps the AVX2 mixed-radix kernel tree (src/avxradix) as an AbstractFFTPlan, so
# non-pow2 sizes whose radix tree we can build (2·3·5-smooth via base B36 + radix 3/4/5/6/8/9/12) can be
# routed through it. Float64-only; the constructor returns `nothing` for any unsupported size/type so the
# caller falls back to the existing path. Hot path (apply_unnormalized! → AvxRadix.applyplan!) is
# concrete/dispatch-free (K is the concrete kernel-tree type).

struct AvxMixedRadixPlan{T, K} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    rp::AvxRadix.RPlan{K}
end

plan_length(p::AvxMixedRadixPlan)::Int = p.n
plan_inverse(p::AvxMixedRadixPlan)::Bool = p.inverse
function apply_unnormalized!(p::AvxMixedRadixPlan, x::AbstractVector)
    AvxRadix.applyplan!(p.rp, x)
    return x
end

"""
    AvxMixedRadixPlan(Complex{T}, n; inverse=false) -> AvxMixedRadixPlan or nothing

Build an AVX2 mixed-radix plan for length `n`, or `nothing` if `T` is not `Float64` or the size is
outside the currently-supported coverage (then the caller falls back).
"""
function AvxMixedRadixPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    T === Float64 || return nothing
    tree = AvxRadix.plan_tree(Int(n), !inverse)
    isnothing(tree) && return nothing
    return AvxMixedRadixPlan{T, typeof(tree.k)}(Int(n), inverse, tree)
end

"""
    AvxMixedRadixPlanW8(Complex{T}, n; inverse=false) -> AvxMixedRadixPlan or nothing

AVX-512 (Vec{8}) variant for W=8-clean sizes (n = 2^(6+3a+2b)·3^b). `nothing` otherwise. Only beats the
W=4 path on small compute-bound sizes, so `autoplan` times it and keeps it only when it wins.
"""
function AvxMixedRadixPlanW8(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    T === Float64 || return nothing
    tree = AvxRadix.plan_tree_w8(Int(n), !inverse)
    isnothing(tree) && return nothing
    return AvxMixedRadixPlan{T, typeof(tree.k)}(Int(n), inverse, tree)
end
