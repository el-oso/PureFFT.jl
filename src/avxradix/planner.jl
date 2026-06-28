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

# pure-pow2 kernel of exponent e (2^e), or nothing. B16/B64 leaves for e∈{4,6}; for e≥8 the rustfft "8xn"
# scheme: monolithic B256/B512 base (by e mod 3, so the leftover is a clean product of 8s + at most one 4,
# no radix-2) + a radix-8/4 chain. Used by the pure-pow2 (e≥8) route AND the single-factor-of-3 route below.
function _pow2_kernel(e::Int, fwd::Bool)
    e == 4 && return B16(fwd)
    e == 5 && return B32(fwd)
    e == 6 && return B64(fwd)
    e >= 8 || return nothing
    base, m = e % 3 == 0 ? (B512(fwd), e - 9) : (B256(fwd), e - 8)
    m % 3 == 1 && return nothing                        # never happens for this base choice, but stay safe
    k::Kernel = base
    for _ in 1:(m ÷ 3)
        k = MR8(k, fwd)
    end
    m % 3 == 2 && (k = MR4(k, fwd))
    return k
end

# build a kernel tree for n (or nothing if unsupported). fwd = forward.
function plan_tree(n::Int, fwd::Bool=true)
    p2, p3, p5, rest = factor235(n)
    p7 = 0; while rest % 7 == 0; rest ÷= 7; p7 += 1; end
    rest == 1 || return nothing
    # ---- single factor of 7 (2^a·7): no 7^2 base, so carry the lone 7 in ONE radix-7 pass (MR7, reusing
    # avx_column_butterfly7) over a pure-pow2 leaf. 112=MR7(B16), 224=MR7(B32), 448=MR7(B64), 1792=MR7(B256).
    if p7 == 1 && p3 == 0 && p5 == 0
        inner = _pow2_kernel(p2, fwd)
        return isnothing(inner) ? nothing : RPlan(MR7(inner, fwd))
    end
    p7 == 0 || return nothing                           # 7^2+ / 7·(3,5) unsupported → fall back
    # ---- single factor of 5 (2^a·5): carry the lone 5 in ONE radix-5 pass (MR5) over a pow2 leaf.
    # 80=MR5(B16), 160=MR5(B32), 320=MR5(B64), 1280=MR5(B256). (B32 leaf supplies the 2^5 block.)
    if p5 == 1 && p3 == 0
        inner = _pow2_kernel(p2, fwd)
        return isnothing(inner) ? nothing : RPlan(MR5(inner, fwd))
    end
    # ---- 2^a·3·5 (lone 3 and lone 5): pow2 leaf + one radix-3 then one radix-5 pass.
    # 240=MR5(MR3(B16)), 480=MR5(MR3(B32)), 960=MR5(MR3(B64)).
    if p3 == 1 && p5 == 1
        inner = _pow2_kernel(p2, fwd)
        return isnothing(inner) ? nothing : RPlan(MR5(MR3(inner, fwd), fwd))
    end
    # Pure power of two ≥ 256: B256/B512 base + radix-8/4 chain (e≥8 only; smaller pure pow2 keep the
    # existing FFT-specific path — leave p3==0 behavior unchanged).
    if p3 == 0 && p5 == 0
        p2 >= 8 || return nothing
        k = _pow2_kernel(p2, fwd)
        return isnothing(k) ? nothing : RPlan(k)
    end
    # Single factor of 3 (2^a·3): no 3^2 base (B36/B18) available, so these fell to the slow generic path.
    # Use a pure-pow2 leaf/chain that absorbs the 2s and carry the lone 3 in ONE mixed-radix pass folding up
    # to 2 leftover 2s (MR3/MR6/MR12 = 3·2^j). Matches FFTW's "wide pow2 codelet × one odd codelet" shape.
    # Smallest j → largest pow2 inner: 48=MR3(B16), 96=MR6(B16), 192=MR3(B64), 384=MR6(B64), 768=MR3(B256)…
    if p3 == 1 && p5 == 0
        for j in 0:2
            p2 - j == 5 && continue                     # skip B32 leaf: B16(4×4)+MR6 beats B32(4×8)+MR3 at 96 (measured)
            inner = _pow2_kernel(p2 - j, fwd)
            isnothing(inner) && continue
            outer = j == 0 ? MR3(inner, fwd) : j == 1 ? MR6(inner, fwd) : MR12(inner, fwd)
            return RPlan(outer)
        end
        return nothing
    end
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
