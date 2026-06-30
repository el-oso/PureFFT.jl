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

_wrap(r::Int, k::Kernel, fwd::Bool) = r == 3 ? MR3(k, fwd) : r == 4 ? MR4(k, fwd) : r == 5 ? MRPrime(5, k, fwd) :
    r == 6 ? MR6(k, fwd) : r == 8 ? MR8(k, fwd) : r == 9 ? MR9(k, fwd) : r == 12 ? MR12(k, fwd) : error("radix $r")

_build_tree_k(base::Kernel, radixes, fwd::Bool) = foldl((k, r) -> _wrap(r, k, fwd), radixes; init = base)
_build_tree(base::Kernel, radixes, fwd::Bool) = RPlan(_build_tree_k(base, radixes, fwd))

# pure-pow2 kernel of exponent e (2^e), or nothing. B16/B64 leaves for e∈{4,6}; for e≥8 the rustfft "8xn"
# scheme: monolithic B256/B512 base (by e mod 3, so the leftover is a clean product of 8s + at most one 4,
# no radix-2) + a radix-8/4 chain. Used by the pure-pow2 (e≥8) route AND the single-factor-of-3 route below.
function _pow2_kernel(e::Int, fwd::Bool)
    e == 3 && return B8(fwd)
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

# 2^p2·5^p5 (p2∈{1,2,7}, no pow2 leaf): odd 5-power core (B25 5² base + MR5 for the rest, or BP5 for a
# lone 5) wrapped in p2 MR2 radix-2 passes (odd-M tail handles the odd core; M is even above the first).
function _pow2x_pow5_kernel(p2::Int, p5::Int, fwd::Bool)
    k::Kernel = p5 >= 2 ? B25(fwd) : BP(5, fwd)
    for _ in 1:(p5 >= 2 ? p5 - 2 : p5 - 1); k = MRPrime(5, k, fwd); end
    for _ in 1:p2; k = MR2(k, fwd); end
    k
end

# Smooth 2^p2·3^p3·5^p5 kernel (no RPlan wrap), or nothing. Used both for pure-smooth plans and as the
# inner core under a single-prime (7/13) wrap. Small pure-pow2 leaves (B8..B64) are allowed here (the pure-
# smooth public path gates those off via p2>=8 to preserve the dedicated pow2 routing).
function _smooth235_kernel(p2::Int, p3::Int, p5::Int, fwd::Bool)
    # 2^a·5^m (any power of 5, no 3s): even pow2 leaf + m radix-5 passes. MR5's inner length M is always
    # even, so the chain composes. 80=MR5(B16), 1000=MR5³(B8), 10000=MR5⁴(B16).
    if p5 >= 1 && p3 == 0
        inner = _pow2_kernel(p2, fwd)
        if isnothing(inner)                             # p2∈{1,2,7}: no pow2 leaf. Root the 5-power on
            return _pow2x_pow5_kernel(p2, p5, fwd)      # B25/BP5 + carry the 2s as MR2 passes. 250=2·5³, 500=4·5³.
        end
        return foldl((k, _) -> MRPrime(5, k, fwd), 1:p5; init = inner)
    end
    # 2^a·3·5 (lone 3 and lone 5): pow2 leaf + one radix-3 then one radix-5. 240=MR5(MR3(B16)).
    if p3 == 1 && p5 == 1
        inner = _pow2_kernel(p2, fwd)
        return isnothing(inner) ? nothing : MRPrime(5, MR3(inner, fwd), fwd)
    end
    # Pure power of two: B8..B512 leaf (+ 8xn chain for e≥8). The pure-smooth caller gates p2 for these.
    if p3 == 0 && p5 == 0
        return _pow2_kernel(p2, fwd)
    end
    # Single factor of 3 (2^a·3): pow2 leaf/chain absorbing the 2s + one MR3/MR6/MR12 (=3·2^j) for the lone 3.
    # Smallest j → largest pow2 inner: 48=MR3(B16), 96=MR6(B16), 192=MR3(B64), 768=MR3(B256)…
    if p3 == 1 && p5 == 0
        for j in 0:2
            p2 - j == 5 && continue                     # skip B32 leaf: B16(4×4)+MR6 beats B32(4×8)+MR3 at 96 (measured)
            inner = _pow2_kernel(p2 - j, fwd)
            isnothing(inner) && continue
            return j == 0 ? MR3(inner, fwd) : j == 1 ? MR6(inner, fwd) : MR12(inner, fwd)
        end
        return nothing
    end
    # Pure power of 3 (3^p3, no 2s/5s): B9 base (3²) + radix-9 passes (2 threes each) + one radix-3 for an
    # odd power. Every inner length here is odd, so the MR9/MR3 cross-passes use the partial-V2f odd-column
    # tail. 81=MR9(B9), 729=MR9²(B9), 6561=MR9³(B9); 27=MR3(B9), 2187=MR3(MR9²(B9)).
    if p2 == 0 && p5 == 0 && p3 >= 2
        rem = p3 - 2
        k::Kernel = B9(fwd)
        for _ in 1:(rem ÷ 2); k = MR9(k, fwd); end
        isodd(rem) && (k = MR3(k, fwd))
        return k
    end
    p3 >= 2 || return nothing                          # need 3^2 for a base (B36 or B18)
    # Prefer base B36 = 2^2·3^2 when 2^2 is available (consumes two 2s + two 3s).
    if p2 >= 2
        rx = plan_radixes(p2 - 2, p3 - 2, p5)
        isnothing(rx) || return _build_tree_k(B36(fwd), rx, fwd)
    end
    # Base B18 = 2·3^2 (one 2 + two 3s) — covers 2^odd·3^2·5^c whose B36 leftover would need radix-2.
    if p2 >= 1
        rx18 = plan_radixes(p2 - 1, p3 - 2, p5)
        isnothing(rx18) || return _build_tree_k(B18(fwd), rx18, fwd)
    end
    return nothing
end

# odd-prime leaf coverage: a residual prime carried as a BP leaf (>7, since 3/5/7 are SIMD radix passes).
# 43 = largest prime an MR3/MR5 wrap composes for the small odd r2r-inner sizes (33=3·11, 65=5·13, 95=5·19,
# 129=3·43). Above this the O(p²) leaf loses to Bluestein, so fall back (autoplan times it regardless).
const _BP_MAX = 43
function _isprime_odd(m::Int)
    m < 2 && return false
    iseven(m) && return m == 2
    d = 3
    while d * d <= m
        m % d == 0 && return false
        d += 2
    end
    return true
end

# Odd-n radix tree: n = 3^a·5^b·7^c·q with q∈{1, one supported odd prime ≤43}. Innermost leaf = B9 (3²)
# when a≥2, else a prime leaf BP{q/5/7/3}; wrap MR9/MR3 (3s) + MR5 (5s) + MR7 (7s) — each has the odd-M
# partial-V2f tail (radix-3/9 from commit 3aebc8b, radix-5/7 here), so every (odd) inner length composes.
# Covers pure powers (25=MR5(BP5), 49=MR7(BP7), 63=MR7(B9), 125, 3ⁿ) and residual-prime composites
# (11,13 leaves; 33,65,95,129). nothing if the residual is an unsupported prime/prime-power.
function _odd_tree(n::Int, fwd::Bool)
    n == 1 && return nothing
    a = b = c = 0; m = n
    while m % 3 == 0; m ÷= 3; a += 1; end
    while m % 5 == 0; m ÷= 5; b += 1; end
    while m % 7 == 0; m ÷= 7; c += 1; end
    if m == 169                                          # 13² residual: BP13 leaf + one radix-13 pass (odd M=13)
        k::Kernel = MRPrime(13, BP(13, fwd), fwd)
    elseif m != 1                                        # residual non-{3,5,7} prime ≤43 as the leaf
        (m <= _BP_MAX && _isprime_odd(m)) || return nothing
        k = BP(m, fwd)
    elseif a >= 2                                        # B9 (3²) base when ≥2 threes (radix-9 preferred)
        k = B9(fwd); a -= 2
    elseif b >= 2                                        # B25 (5²) base: 25, 125=MR5(B25), 625=MR5²(B25)
        k = B25(fwd); b -= 2
    elseif c >= 2                                        # B49 (7²) base: 49, 343=MR7(B49)
        k = B49(fwd); c -= 2
    elseif c >= 1
        k = BP(7, fwd); c -= 1
    elseif b >= 1
        k = BP(5, fwd); b -= 1
    elseif a == 1
        k = BP(3, fwd); a -= 1
    else
        return nothing
    end
    while a >= 2; k = MR9(k, fwd); a -= 2; end
    a == 1 && (k = MR3(k, fwd))
    for _ in 1:b; k = MRPrime(5, k, fwd); end
    for _ in 1:c; k = MRPrime(7, k, fwd); end
    return k
end

# Carry leftover smooth 2^p2·3^p3·5^p5 factors over an EVEN-M kernel using the fast radixes. M stays even
# throughout (the caller guarantees an even base), so no odd tail is needed and any radix is safe. Combine
# 2·3→MR6 and 2³→MR8 where possible (a long MR2 chain is the slow signature); leftovers as MR5/MR3/MR4/MR2.
function _carry_even(k0::Kernel, p2::Int, p3::Int, p5::Int, fwd::Bool)
    k::Kernel = k0
    for _ in 1:p5; k = MRPrime(5, k, fwd); end
    while p2 >= 1 && p3 >= 1; k = MR6(k, fwd); p2 -= 1; p3 -= 1; end
    for _ in 1:p3; k = MR3(k, fwd); end
    while p2 >= 3; k = MR8(k, fwd); p2 -= 3; end
    p2 == 2 && (k = MR4(k, fwd))
    p2 == 1 && (k = MR2(k, fwd))
    k
end

# build a kernel tree for n (or nothing if unsupported). fwd = forward.
function plan_tree(n::Int, fwd::Bool=true)
    if isodd(n)                                          # all odd sizes route through the odd-prime tree
        ot = _odd_tree(n, fwd)
        return isnothing(ot) ? nothing : RPlan(ot)
    end
    p2, p3, p5, rest = factor235(n)
    p7 = 0;  while rest % 7  == 0; rest ÷= 7;  p7  += 1; end
    p13 = 0; while rest % 13 == 0; rest ÷= 13; p13 += 1; end
    rest == 1 || return nothing                         # any other prime (11, ≥17) → fall back
    (p7 <= 2 && p13 <= 2) || return nothing             # 7³+/13³+ unsupported → fall back

    # ---- pure smooth 2·3·5: keep the dedicated pow2 routing for small pure powers of two (p2<8 → fall back).
    if p7 == 0 && p13 == 0
        p3 == 0 && p5 == 0 && p2 < 8 && return nothing
        ks = _smooth235_kernel(p2, p3, p5, fwd)
        return isnothing(ks) ? nothing : RPlan(ks)
    end
    # a square (7²/13²) coexisting with the other prime (e.g. 7²·13, 7·13²) is unsupported → fall back
    ((p7 < 2 || p13 == 0) && (p13 < 2 || p7 == 0)) || return nothing

    # ---- single 7 and/or single 13: ONE radix-7/13 column pass each (MR7/MR13, the verified
    # avx_column_butterfly7/13) over a fast smooth 2·3·5 base when one exists (M even ⇒ no tail).
    # 112=MR7(B16), 65520=MR13(MR7(720)), 5040=MR7(720), 208=MR13(B16). p2≥1 always (n even).
    if p7 <= 1 && p13 <= 1
        inner = _smooth235_kernel(p2, p3, p5, fwd)
        if !isnothing(inner)
            k::Kernel = inner
            p7  == 1 && (k = MRPrime(7, k, fwd))
            p13 == 1 && (k = MRPrime(13, k, fwd))
            return RPlan(k)
        end
        # No fast smooth base (small 2^a·3^b·prime: 14/26/52/78/182). Root on the B2 leaf so each prime rides
        # the FAST column butterfly (MR7/MR13 over even M=2) instead of the generic O(p²) BP leaf, then carry
        # the rest. 26=MR13(B2), 52=MR2(MR13(B2)), 78=MR3(MR13(B2)), 14=MR7(B2), 182=MR13(MR7(B2)).
        kk::Kernel = B2(fwd)
        p7  == 1 && (kk = MRPrime(7, kk, fwd))
        p13 == 1 && (kk = MRPrime(13, kk, fwd))
        return RPlan(_carry_even(kk, p2 - 1, p3, p5, fwd))      # B2 consumed one factor of 2; M now even
    end

    # ---- 7² / 13² squares: a dedicated ODD base (B49 = the 7² codelet; MR13(BP13) = 13² with the new odd-M
    # tail) + a mandatory first MR2 to even-ize the odd M (the only odd-safe even-izer — MR4/6/8/12 drop the
    # odd last column), then the fast even carry. 98=MR2(B49), 196=MR2(MR2(B49)), 294=MR3(MR2(B49)),
    # 588=MR6(MR2(B49)), 169=MR13(BP13) [odd, via _odd_tree], 338=MR2(MR13(BP13)).
    core::Kernel = p7 == 2 ? B49(fwd) : MRPrime(13, BP(13, fwd), fwd)
    return RPlan(_carry_even(MR2(core, fwd), p2 - 1, p3, p5, fwd))
end
