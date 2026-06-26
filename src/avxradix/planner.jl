# AVX mixed-radix planner: the radix-stack logic (plan_power12_power6 + the plan_mixed_radix push
# order). First cut: base = Butterfly36 (consumes 2^2·3^2); radixes from {3,4,5,6,8,9,12} (the
# 8/9/12/6 preference + 4/5/3 leftovers). Returns nothing for sizes needing unsupported bases/radixes
# (16/2/7/11 or non-B36 base) → caller falls back to the existing path.
include(joinpath(@__DIR__, "recursive.jl"))

function factor235(n::Int)
    p2 = p3 = p5 = 0; m = n
    while m % 2 == 0; m ÷= 2; p2 += 1; end
    while m % 3 == 0; m ÷= 3; p3 += 1; end
    while m % 5 == 0; m ÷= 5; p5 += 1; end
    (p2, p3, p5, m)
end

# plan_power12_power6: divide radix factors into 8^n·9^m·12^k·6^j, minimize j then maximize k
function plan_power12_power6(p2::Int, p3::Int)
    max12 = min(p2 ÷ 2, p3)
    req6 = Union{Int, Nothing}[nothing, nothing, nothing, nothing]   # req6[s+1] = largest power_twelve given 6^s
    for ht in 0:max12
        h2 = p2 - 2ht; h3 = p3 - ht
        s = (h2 % 3, h3 % 2)
        sixes = s == (0, 0) ? 0 : s == (1, 1) ? 1 : s == (2, 0) ? 2 : s == (0, 1) ? 3 : nothing
        if !isnothing(sixes) && sixes <= h2 && sixes <= h3
            req6[sixes + 1] = ht
        end
    end
    pt = 0; ps = 0
    for i in 0:3
        v = req6[i + 1]
        if !isnothing(v) && v >= pt; pt = v; ps = i; end
    end
    if p2 == 1 && p3 > 0; ps = 1; end
    if p2 > 1 && p3 == 1 && pt == 0; ps = 1; end
    (pt, ps)
end

# plan_mixed_radix push order (innermost first). p2/p3/p5 = factors AFTER the base. nothing if unsupported.
function plan_radixes(p2::Int, p3::Int, p5::Int)
    radixes = Int[]
    pt, ps = plan_power12_power6(p2, p3)
    q2 = p2 - 2pt - ps; q3 = p3 - pt - ps
    if q2 % 3 == 1 && q2 > 1
        push!(radixes, 16); q2 -= 4                  # need MR16 — unsupported
    end
    append!(radixes, fill(12, pt))
    append!(radixes, fill(9, q3 ÷ 2))
    append!(radixes, fill(8, q2 ÷ 3))
    append!(radixes, fill(6, ps))
    append!(radixes, fill(5, p5))
    q2 % 3 == 2 && push!(radixes, 4)
    q3 % 2 == 1 && push!(radixes, 3)
    q2 % 3 == 1 && push!(radixes, 2)                 # need MR2 — unsupported
    any(r -> r in (2, 16), radixes) && return nothing
    radixes
end

_wrap(r::Int, k::Kernel, fwd::Bool) = r == 3 ? MR3(k, fwd) : r == 4 ? MR4(k, fwd) : r == 5 ? MR5(k, fwd) :
    r == 6 ? MR6(k, fwd) : r == 8 ? MR8(k, fwd) : r == 9 ? MR9(k, fwd) : r == 12 ? MR12(k, fwd) : error("radix $r")

_build_tree(base::Kernel, radixes, fwd::Bool) = RPlan(foldl((k, r) -> _wrap(r, k, fwd), radixes; init = base))

# build a kernel tree for n (or nothing if unsupported). fwd = forward.
function plan_tree(n::Int, fwd::Bool=true)
    p2, p3, p5, rest = factor235(n)
    rest == 1 || return nothing
    p3 >= 2 || return nothing                          # need 3^2 for a base (B36 or B18)
    # Prefer base B36 = 2^2·3^2 when 2^2 is available (consumes two 2s + two 3s).
    if p2 >= 2
        rx = plan_radixes(p2 - 2, p3 - 2, p5)
        isnothing(rx) || return _build_tree(B36(fwd), rx, fwd)
    end
    # Base B18 = 2·3^2 (one 2 + two 3s) — covers 2^odd·3^2·5^c whose B36 leftover would need an unsupported
    # radix-2 (e.g. 90 = B18·5, 360 = B18·4·5). Faithful port of rustfft Butterfly18Avx64.
    if p2 >= 1
        rx18 = plan_radixes(p2 - 1, p3 - 2, p5)
        isnothing(rx18) || return _build_tree(B18(fwd), rx18, fwd)
    end
    return nothing
end
