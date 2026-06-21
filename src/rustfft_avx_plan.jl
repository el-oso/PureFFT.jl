# RustFFTAvxPlan — wraps the faithful-port keystone (src/rustport) as an AbstractFFTPlan, so non-pow2
# sizes whose rust-style radix tree we can build (2·3·5-smooth via base B36 + radix 3/4/5/6/8/9/12) can
# be routed through it. f64-only; the constructor returns `nothing` for any unsupported size/type so the
# caller falls back to the existing path. Hot path (apply_unnormalized! → RustPort.applyplan!) is
# concrete/dispatch-free (K is the concrete kernel-tree type).

struct RustFFTAvxPlan{T, K} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    rp::RustPort.RPlan{K}
end

plan_length(p::RustFFTAvxPlan)::Int = p.n
plan_inverse(p::RustFFTAvxPlan)::Bool = p.inverse
function apply_unnormalized!(p::RustFFTAvxPlan, x::AbstractVector)
    RustPort.applyplan!(p.rp, x)
    return x
end

"""
    RustFFTAvxPlan(Complex{T}, n; inverse=false) -> RustFFTAvxPlan or nothing

Build a faithful-port plan for length `n`, or `nothing` if `T` is not `Float64` or the size is outside
the currently-ported coverage (then the caller falls back).
"""
function RustFFTAvxPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    T === Float64 || return nothing
    tree = RustPort.plan_tree(Int(n), !inverse)
    isnothing(tree) && return nothing
    return RustFFTAvxPlan{T, typeof(tree.k)}(Int(n), inverse, tree)
end
