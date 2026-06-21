# Stage 6: `@generated` straight-line DFT codelets — Julia's genfft-equivalent.
#
# `_codelet!` emits, at COMPILE time, a fully-unrolled size-R DFT: every input is a local,
# every twiddle a baked literal `Complex{T}` constant, every step an SSA assignment. LLVM then
# keeps the whole transform in registers (no recursion, no loop, no twiddle-table loads, W=1
# folded away) and FMA-fuses via `_cmul`. This is exactly what FFTW's `genfft`
# hand-written `Butterfly` structs provide; here it falls out of Julia metaprogramming — the
# concrete demonstration that "does Julia need to improve? no, it has @generated".
#
# Direction is a type parameter `Val{S}` (S = -1 forward, +1 inverse) so the twiddle literals
# are correct per direction while staying compile-time constant.

using SIMD: Vec, vload, vstore, shufflevector   # for the batched SoA codelet + SIMD transpose below

# FMA-fused complex multiply (Julia's Complex `*` won't contract to FMA on its own).
@inline function _cmul(o::Complex, w::Complex)
    return Complex(
        muladd(real(o), real(w), -imag(o) * imag(w)),
        muladd(real(o), imag(w), imag(o) * real(w)),
    )
end

# Symbolic radix-2 DIT builder: appends straight-line assignments computing the size-R DFT of
# input symbols `ins`, returns the R output symbols. `ctr` hands out unique local names.
function _gen_dft!(stmts::Vector{Any}, ins::Vector, R::Int, s, ::Type{T}, ctr) where {T}
    R == 1 && return ins
    half = R >> 1
    evin = Vector{Any}(undef, half)
    odin = Vector{Any}(undef, half)
    for i in 1:half
        evin[i] = ins[2i - 1]
        odin[i] = ins[2i]
    end
    ev = _gen_dft!(stmts, evin, half, s, T, ctr)
    od = _gen_dft!(stmts, odin, half, s, T, ctr)
    outs = Vector{Any}(undef, R)
    for k in 0:(half - 1)
        w = Complex{T}(cispi(s * 2k / R))
        lo = Symbol("v", ctr[]); ctr[] += 1
        hi = Symbol("v", ctr[]); ctr[] += 1
        if isone(w)
            push!(stmts, :($lo = $(ev[k + 1]) + $(od[k + 1])))
            push!(stmts, :($hi = $(ev[k + 1]) - $(od[k + 1])))
        else
            t = Symbol("v", ctr[]); ctr[] += 1
            push!(stmts, :($t = _cmul($(od[k + 1]), $w)))
            push!(stmts, :($lo = $(ev[k + 1]) + $t))
            push!(stmts, :($hi = $(ev[k + 1]) - $t))
        end
        outs[k + 1] = lo
        outs[k + 1 + half] = hi
    end
    return outs
end

"""
    _codelet!(out, oo, x, off, str, Val(R), Val(S))

Straight-line size-`R` DFT of the strided subsequence `x[off+1 :: str]` into `out[oo+1:oo+R]`,
generated at compile time. `S = -1` forward, `+1` inverse.
"""
@generated function _codelet!(out, oo, x, off, str, ::Val{R}, ::Val{S}) where {R, S}
    T = real(eltype(x))
    stmts = Any[]
    ins = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        sym = Symbol("a", j)
        ins[j + 1] = sym
        push!(stmts, :(@inbounds $sym = x[off + $j * str + 1]))
    end
    ctr = Ref(0)
    outs = _gen_dft!(stmts, ins, R, S, T, ctr)
    for k in 0:(R - 1)
        push!(stmts, :(@inbounds out[oo + $(k + 1)] = $(outs[k + 1])))
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

# --- dynamic mixed-radix codelet (any size) -------------------------------------------------
# `_codelet!`/`_gen_dft!` above are radix-2 (power-of-two only). `_gen_dft_mixed!` generalizes the
# straight-line generation to ANY length via Cooley-Tukey on the smallest prime factor: split
# R = p·m, recurse to p DFT-m's over the stride-p subsequences, then combine with baked W_R^{ak}
# twiddle literals. A prime leaf (p == R) is a direct R-point DFT. This is the genfft idea at
# Julia compile time — PureFFT can synthesize a tailored unrolled kernel for any size at plan time,
# covering cases (odd primes, prime powers) where typical FFT libraries fall back to generic mixed-radix.

# smallest prime factor of R (R ≥ 2)
function _smallest_prime_factor(R::Int)
    R % 2 == 0 && return 2
    p = 3
    while p * p <= R
        R % p == 0 && return p
        p += 2
    end
    return R
end

# emit `acc = sum of terms` as a left-folded chain of `+`; returns the accumulator symbol.
function _emit_sum!(stmts::Vector{Any}, terms::Vector, ctr)
    acc = terms[1]
    for t in @view terms[2:end]
        s = Symbol("v", ctr[]); ctr[] += 1
        push!(stmts, :($s = $acc + $t))
        acc = s
    end
    return acc
end

# Straight-line mixed-radix DFT of input symbols `ins` (length R) → R output symbols. `s` is the
# sign (-1 forward, +1 inverse); twiddles are compile-time `Complex{T}` literals.
function _gen_dft_mixed!(stmts::Vector{Any}, ins::Vector, R::Int, s, ::Type{T}, ctr) where {T}
    R == 1 && return ins
    p = _smallest_prime_factor(R)
    # twiddle literal W_R^{e} = exp(s·2πi·e/R); reduce exponent mod R for accuracy.
    twiddle(e) = Complex{T}(cispi(s * 2 * mod(e, R) / R))
    # multiply symbol `g` by literal `w`, eliding trivial ±1 / ±i.
    function mul_lit(g, w)
        isone(w) && return g
        t = Symbol("v", ctr[]); ctr[] += 1
        push!(stmts, :($t = _cmul($g, $w)))
        return t
    end

    if p == R                                   # prime leaf: direct DFT
        outs = Vector{Any}(undef, R)
        for k in 0:(R - 1)
            terms = Any[mul_lit(ins[j + 1], twiddle(j * k)) for j in 0:(R - 1)]
            outs[k + 1] = _emit_sum!(stmts, terms, ctr)
        end
        return outs
    end

    m = R ÷ p
    subs = Vector{Vector{Any}}(undef, p)        # p DFT-m's over stride-p subsequences
    for a in 0:(p - 1)
        sub = Any[ins[a + p * j + 1] for j in 0:(m - 1)]
        subs[a + 1] = _gen_dft_mixed!(stmts, sub, m, s, T, ctr)
    end
    outs = Vector{Any}(undef, R)                # combine: X[k] = Σ_a W_R^{ak} G_a[k mod m]
    for k in 0:(R - 1)
        r = k % m
        terms = Any[mul_lit(subs[a + 1][r + 1], twiddle(a * k)) for a in 0:(p - 1)]
        outs[k + 1] = _emit_sum!(stmts, terms, ctr)
    end
    return outs
end

"""
    _dft_codelet!(out, oo, x, off, str, Val(R), Val(S))

Straight-line size-`R` DFT (any `R`) of `x[off+1 :: str]` into `out[oo+1:oo+R]`, generated at
compile time by mixed-radix Cooley-Tukey. `S = -1` forward, `+1` inverse. In-place safe (all
inputs are read into locals before any output is written).
"""
@generated function _dft_codelet!(out, oo, x, off, str, ::Val{R}, ::Val{S}) where {R, S}
    T = real(eltype(x))
    stmts = Any[]
    ins = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        sym = Symbol("a", j)
        ins[j + 1] = sym
        push!(stmts, :(@inbounds $sym = x[off + $j * str + 1]))
    end
    ctr = Ref(0)
    outs = _gen_dft_mixed!(stmts, ins, R, S, T, ctr)
    for k in 0:(R - 1)
        push!(stmts, :(@inbounds out[oo + $(k + 1)] = $(outs[k + 1])))
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

# largest prime factor of `n` (n ≥ 1)
function _max_prime_factor(n::Int)
    n <= 1 && return 1
    m = 1
    f = 2
    k = n
    while f * f <= k
        while k % f == 0
            m = max(m, f)
            k ÷= f
        end
        f += 1
    end
    return max(m, k)
end

"""
    CodeletPlan{T,N} <: AbstractFFTPlan{T}

Runs the dynamically-generated mixed-radix codelet [`_dft_codelet!`](@ref) for size `N`. The size
is a type parameter so the `@generated` codelet specializes and the hot path stays dispatch-free.
Best for small sizes whose largest prime factor is small (smooth / prime-power); `autoplan` routes
non-power-of-two sizes here when they qualify, else to Bluestein.
"""
struct CodeletPlan{T, N} <: AbstractFFTPlan{T}
    inverse::Bool
end
CodeletPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T} =
    CodeletPlan{T, Int(n)}(inverse)

plan_length(::CodeletPlan{T, N}) where {T, N} = N::Int
plan_inverse(p::CodeletPlan)::Bool = p.inverse
function apply_unnormalized!(p::CodeletPlan{T, N}, x::AbstractVector) where {T, N}
    if p.inverse
        _dft_codelet!(x, 0, x, 0, 1, Val(N), Val(1))
    else
        _dft_codelet!(x, 0, x, 0, 1, Val(N), Val(-1))
    end
    return x
end

# --- batched SoA mixed-radix codelet (the vectorized engine for the four-step executor) ------
# Mixed-radix straight-line builder in SPLIT (re/im) form: each value is a (re,im) symbol pair and
# every op is real arithmetic with scalar twiddle literals → pure FMA, ZERO shuffles. When the
# symbols are SIMD vectors over a batch of independent transforms, this vectorizes perfectly (the
# "vector rank" FFTW uses). Mirrors `_gen_dft_mixed!`.
function _gen_dft_soa_mixed!(stmts, insr::Vector, insi::Vector, R::Int, s, ::Type{T}, ctr, pfx) where {T}
    R == 1 && return (insr, insi)
    p = _smallest_prime_factor(R)
    nm() = (r = Symbol(pfx, "r", ctr[]); i = Symbol(pfx, "i", ctr[]); ctr[] += 1; (r, i))
    add(ar, ai, br, bi) = ((r, i) = nm(); push!(stmts, :($r = $ar + $br)); push!(stmts, :($i = $ai + $bi)); (r, i))
    function mul(ar, ai, w)
        isone(w) && return (ar, ai)
        wr = real(w); wi = imag(w); (r, i) = nm()
        push!(stmts, :($r = muladd($ar, $wr, -$ai * $wi)))
        push!(stmts, :($i = muladd($ar, $wi, $ai * $wr)))
        return (r, i)
    end
    twiddle(e) = Complex{T}(cispi(s * 2 * mod(e, R) / R))
    if p == R                                   # prime leaf: direct DFT
        or = Vector{Any}(undef, R); oi = Vector{Any}(undef, R)
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
    Gr = Vector{Vector{Any}}(undef, p); Gi = Vector{Vector{Any}}(undef, p)
    for a in 0:(p - 1)
        Gr[a + 1], Gi[a + 1] = _gen_dft_soa_mixed!(
            stmts, Any[insr[a + p * j + 1] for j in 0:(m - 1)], Any[insi[a + p * j + 1] for j in 0:(m - 1)], m, s, T, ctr, pfx,
        )
    end
    or = Vector{Any}(undef, R); oi = Vector{Any}(undef, R)
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

"""
    _dft_codelet_soa_batched!(outr, outi, xr, xi, width, Val(R), Val(S))

Apply a size-`R` DFT to each of `width` independent transforms held in split (re/im) arrays, where
transform `t`'s element `j` is at `[j*width + t]`. Vectorized over `t` with `Vec{W}` (W = 8 for
Float64); a final idempotent overlapping vector covers a `width` not divisible by W. Requires
`width ≥ W`. Shuffle-free (split layout). This is the per-pass engine of [`FourStepCodeletPlan`](@ref).
"""
@generated function _dft_codelet_soa_batched!(outr, outi, xr, xi, width::Int, ::Val{R}, ::Val{S}) where {R, S}
    T = eltype(xr)
    W = 64 ÷ sizeof(T)
    es = sizeof(T)
    # vector body: W transforms per Vec, gathered/scattered contiguously over the batch index `b`
    vb = Any[]
    vr = Vector{Any}(undef, R); vi = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        a = Symbol("vinr", j); b = Symbol("vini", j)   # input names distinct from generator's "vr"/"vi"
        vr[j + 1] = a; vi[j + 1] = b
        push!(vb, :(@inbounds $a = vload(Vec{$W, $T}, pr + (($j * width + b) * $es))))
        push!(vb, :(@inbounds $b = vload(Vec{$W, $T}, pii + (($j * width + b) * $es))))
    end
    vor, voi = _gen_dft_soa_mixed!(vb, vr, vi, R, S, T, Ref(0), "v")
    for k in 0:(R - 1)
        push!(vb, :(@inbounds vstore($(vor[k + 1]), por + (($k * width + b) * $es))))
        push!(vb, :(@inbounds vstore($(voi[k + 1]), poi + (($k * width + b) * $es))))
    end
    # scalar remainder body: one transform `t` at a time (covers width % W, and width < W)
    sb = Any[]
    sr = Vector{Any}(undef, R); si = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        a = Symbol("sinr", j); b = Symbol("sini", j)   # input names distinct from generator's "sr"/"si"
        sr[j + 1] = a; si[j + 1] = b
        push!(sb, :(@inbounds $a = xr[$j * width + t + 1]))
        push!(sb, :(@inbounds $b = xi[$j * width + t + 1]))
    end
    sor, soi = _gen_dft_soa_mixed!(sb, sr, si, R, S, T, Ref(0), "s")
    for k in 0:(R - 1)
        push!(sb, :(@inbounds outr[$k * width + t + 1] = $(sor[k + 1])))
        push!(sb, :(@inbounds outi[$k * width + t + 1] = $(soi[k + 1])))
    end
    quote
        GC.@preserve xr xi outr outi begin
            pr = pointer(xr); pii = pointer(xi); por = pointer(outr); poi = pointer(outi)
            b = 0
            while b + $W <= width
                $(vb...)
                b += $W
            end
        end
        @inbounds for t in (width - (width % $W)):(width - 1)
            $(sb...)
        end
        return nothing
    end
end

"""
    _dft_codelet_soa_batched_tw!(outr, outi, xr, xi, twr, twi, width, Val(R), Val(S))

Like [`_dft_codelet_soa_batched!`](@ref) but multiplies each output by a per-element twiddle
(`twr`/`twi`, same `[k*width+t]` layout as the output) on store — fusing the four-step / mixed-radix
twiddle pass into the codelet so there is no separate full-array twiddle pass. The per-pass engine of
[`RecursiveMixedRadixPlan`](@ref).
"""
@generated function _dft_codelet_soa_batched_tw!(outr, outi, xr, xi, twr, twi, width::Int, ::Val{R}, ::Val{S}) where {R, S}
    T = eltype(xr)
    W = 64 ÷ sizeof(T)
    es = sizeof(T)
    vb = Any[]
    vr = Vector{Any}(undef, R); vi = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        a = Symbol("vinr", j); b = Symbol("vini", j)
        vr[j + 1] = a; vi[j + 1] = b
        push!(vb, :(@inbounds $a = vload(Vec{$W, $T}, pr + (($j * width + b) * $es))))
        push!(vb, :(@inbounds $b = vload(Vec{$W, $T}, pii + (($j * width + b) * $es))))
    end
    vor, voi = _gen_dft_soa_mixed!(vb, vr, vi, R, S, T, Ref(0), "v")
    for k in 0:(R - 1)
        tr = Symbol("vtr", k); ti = Symbol("vti", k)
        push!(vb, :(@inbounds $tr = vload(Vec{$W, $T}, ptr + (($k * width + b) * $es))))
        push!(vb, :(@inbounds $ti = vload(Vec{$W, $T}, pti + (($k * width + b) * $es))))
        push!(vb, :(@inbounds vstore($(vor[k + 1]) * $tr - $(voi[k + 1]) * $ti, por + (($k * width + b) * $es))))
        push!(vb, :(@inbounds vstore($(vor[k + 1]) * $ti + $(voi[k + 1]) * $tr, poi + (($k * width + b) * $es))))
    end
    sb = Any[]
    sr = Vector{Any}(undef, R); si = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        a = Symbol("sinr", j); b = Symbol("sini", j)
        sr[j + 1] = a; si[j + 1] = b
        push!(sb, :(@inbounds $a = xr[$j * width + t + 1]))
        push!(sb, :(@inbounds $b = xi[$j * width + t + 1]))
    end
    sor, soi = _gen_dft_soa_mixed!(sb, sr, si, R, S, T, Ref(0), "s")
    for k in 0:(R - 1)
        push!(sb, :(@inbounds wr = twr[$k * width + t + 1]))
        push!(sb, :(@inbounds wi = twi[$k * width + t + 1]))
        push!(sb, :(@inbounds outr[$k * width + t + 1] = $(sor[k + 1]) * wr - $(soi[k + 1]) * wi))
        push!(sb, :(@inbounds outi[$k * width + t + 1] = $(sor[k + 1]) * wi + $(soi[k + 1]) * wr))
    end
    quote
        GC.@preserve xr xi outr outi twr twi begin
            pr = pointer(xr); pii = pointer(xi); por = pointer(outr); poi = pointer(outi)
            ptr = pointer(twr); pti = pointer(twi)
            b = 0
            while b + $W <= width
                $(vb...)
                b += $W
            end
        end
        @inbounds for t in (width - (width % $W)):(width - 1)
            $(sb...)
        end
        return nothing
    end
end

# --- SIMD register-tiled SoA transpose (the four-step's reorder) -----------------------------
# The naive scalar strided transpose was the four-step's dominant overhead (~3μs at n=5760). This
# does it in 8×8 register tiles (contiguous SIMD loads/stores + an in-register 8×8 transpose), with
# a scalar remainder for dims not divisible by 8 → ~1.6× faster, the main lever toward FFTW parity.

# in-register 8×8 Float64 transpose (3-stage unpack network)
@inline function _transpose8(r0, r1, r2, r3, r4, r5, r6, r7)
    ul(a, b) = shufflevector(a, b, Val((0, 8, 2, 10, 4, 12, 6, 14)))
    uh(a, b) = shufflevector(a, b, Val((1, 9, 3, 11, 5, 13, 7, 15)))
    a0 = ul(r0, r1); a1 = uh(r0, r1); a2 = ul(r2, r3); a3 = uh(r2, r3)
    a4 = ul(r4, r5); a5 = uh(r4, r5); a6 = ul(r6, r7); a7 = uh(r6, r7)
    bl(a, b) = shufflevector(a, b, Val((0, 1, 8, 9, 4, 5, 12, 13)))
    bh(a, b) = shufflevector(a, b, Val((2, 3, 10, 11, 6, 7, 14, 15)))
    b0 = bl(a0, a2); b2 = bh(a0, a2); b1 = bl(a1, a3); b3 = bh(a1, a3)
    b4 = bl(a4, a6); b6 = bh(a4, a6); b5 = bl(a5, a7); b7 = bh(a5, a7)
    cl(a, b) = shufflevector(a, b, Val((0, 1, 2, 3, 8, 9, 10, 11)))
    ch(a, b) = shufflevector(a, b, Val((4, 5, 6, 7, 12, 13, 14, 15)))
    return (cl(b0, b4), cl(b1, b5), cl(b2, b6), cl(b3, b7), ch(b0, b4), ch(b1, b5), ch(b2, b6), ch(b3, b7))
end

# Generic scalar fallback (e.g. Float32 — its 16-wide transpose network differs; correctness first).
function _transpose_soa!(br::AbstractVector, bi, ar, ai, n1::Int, n2::Int)
    @inbounds for k1 in 0:(n1 - 1), i2 in 0:(n2 - 1)
        br[i2 * n1 + k1 + 1] = ar[k1 * n2 + i2 + 1]
        bi[i2 * n1 + k1 + 1] = ai[k1 * n2 + i2 + 1]
    end
    return
end

# br[i2*n1+k1] = ar[k1*n2+i2] (and bi←ai), SoA, Float64. 8×8 SIMD tiles + scalar edges.
function _transpose_soa!(br::Vector{Float64}, bi::Vector{Float64}, ar::Vector{Float64}, ai::Vector{Float64}, n1::Int, n2::Int)
    @inline vl(p, off) = vload(Vec{8, Float64}, p + off * 8)
    GC.@preserve br bi ar ai begin
        par = pointer(ar); pai = pointer(ai); pbr = pointer(br); pbi = pointer(bi)
        k1 = 0
        @inbounds while k1 + 8 <= n1
            i2 = 0
            while i2 + 8 <= n2
                o = k1 * n2 + i2
                R0, R1, R2, R3, R4, R5, R6, R7 = _transpose8(vl(par, o), vl(par, o + n2), vl(par, o + 2n2), vl(par, o + 3n2), vl(par, o + 4n2), vl(par, o + 5n2), vl(par, o + 6n2), vl(par, o + 7n2))
                I0, I1, I2, I3, I4, I5, I6, I7 = _transpose8(vl(pai, o), vl(pai, o + n2), vl(pai, o + 2n2), vl(pai, o + 3n2), vl(pai, o + 4n2), vl(pai, o + 5n2), vl(pai, o + 6n2), vl(pai, o + 7n2))
                q = i2 * n1 + k1
                vstore(R0, pbr + q * 8); vstore(R1, pbr + (q + n1) * 8); vstore(R2, pbr + (q + 2n1) * 8); vstore(R3, pbr + (q + 3n1) * 8)
                vstore(R4, pbr + (q + 4n1) * 8); vstore(R5, pbr + (q + 5n1) * 8); vstore(R6, pbr + (q + 6n1) * 8); vstore(R7, pbr + (q + 7n1) * 8)
                vstore(I0, pbi + q * 8); vstore(I1, pbi + (q + n1) * 8); vstore(I2, pbi + (q + 2n1) * 8); vstore(I3, pbi + (q + 3n1) * 8)
                vstore(I4, pbi + (q + 4n1) * 8); vstore(I5, pbi + (q + 5n1) * 8); vstore(I6, pbi + (q + 6n1) * 8); vstore(I7, pbi + (q + 7n1) * 8)
                i2 += 8
            end
            @inbounds for ii in i2:(n2 - 1), r in 0:7        # remainder columns
                br[ii * n1 + (k1 + r) + 1] = ar[(k1 + r) * n2 + ii + 1]
                bi[ii * n1 + (k1 + r) + 1] = ai[(k1 + r) * n2 + ii + 1]
            end
            k1 += 8
        end
        @inbounds for kk in k1:(n1 - 1), i2 in 0:(n2 - 1)    # remainder rows
            br[i2 * n1 + kk + 1] = ar[kk * n2 + i2 + 1]
            bi[i2 * n1 + kk + 1] = ai[kk * n2 + i2 + 1]
        end
    end
    return
end

# --- split-layout (SoA) variant -------------------------------------------------------------
# Same builder, but each value is a (real, imag) symbol pair; twiddle real/imag parts are baked
# literals and the complex multiply is two `muladd`s → pure-real straight-line code, no shuffles.
function _gen_dft_soa!(stmts, insr::Vector, insi::Vector, R::Int, s, ::Type{T}, ctr) where {T}
    R == 1 && return (insr, insi)
    half = R >> 1
    evr = Vector{Any}(undef, half); evi = Vector{Any}(undef, half)
    odr = Vector{Any}(undef, half); odi = Vector{Any}(undef, half)
    for i in 1:half
        evr[i] = insr[2i - 1]; evi[i] = insi[2i - 1]
        odr[i] = insr[2i]; odi[i] = insi[2i]
    end
    er, ei = _gen_dft_soa!(stmts, evr, evi, half, s, T, ctr)
    fr, fi = _gen_dft_soa!(stmts, odr, odi, half, s, T, ctr)
    outr = Vector{Any}(undef, R); outi = Vector{Any}(undef, R)
    for k in 0:(half - 1)
        w = Complex{T}(cispi(s * 2k / R))
        wr = real(w); wi = imag(w)
        lor = Symbol("r", ctr[]); loi = Symbol("i", ctr[]); ctr[] += 1
        hir = Symbol("r", ctr[]); hii = Symbol("i", ctr[]); ctr[] += 1
        if isone(w)
            push!(stmts, :($lor = $(er[k + 1]) + $(fr[k + 1])))
            push!(stmts, :($loi = $(ei[k + 1]) + $(fi[k + 1])))
            push!(stmts, :($hir = $(er[k + 1]) - $(fr[k + 1])))
            push!(stmts, :($hii = $(ei[k + 1]) - $(fi[k + 1])))
        else
            tr = Symbol("r", ctr[]); ti = Symbol("i", ctr[]); ctr[] += 1
            push!(stmts, :($tr = muladd($(fr[k + 1]), $wr, -$(fi[k + 1]) * $wi)))
            push!(stmts, :($ti = muladd($(fr[k + 1]), $wi, $(fi[k + 1]) * $wr)))
            push!(stmts, :($lor = $(er[k + 1]) + $tr))
            push!(stmts, :($loi = $(ei[k + 1]) + $ti))
            push!(stmts, :($hir = $(er[k + 1]) - $tr))
            push!(stmts, :($hii = $(ei[k + 1]) - $ti))
        end
        outr[k + 1] = lor; outi[k + 1] = loi
        outr[k + 1 + half] = hir; outi[k + 1 + half] = hii
    end
    return outr, outi
end

@generated function _codelet_soa!(
        outr, outi, oo, xr, xi, off, str, ::Val{R}, ::Val{S}
    ) where {R, S}
    T = eltype(xr)
    stmts = Any[]
    insr = Vector{Any}(undef, R); insi = Vector{Any}(undef, R)
    for j in 0:(R - 1)
        sr = Symbol("ar", j); si = Symbol("ai", j)
        insr[j + 1] = sr; insi[j + 1] = si
        push!(stmts, :(@inbounds $sr = xr[off + $j * str + 1]))
        push!(stmts, :(@inbounds $si = xi[off + $j * str + 1]))
    end
    ctr = Ref(0)
    or_, oi = _gen_dft_soa!(stmts, insr, insi, R, S, T, ctr)
    for k in 0:(R - 1)
        push!(stmts, :(@inbounds outr[oo + $(k + 1)] = $(or_[k + 1])))
        push!(stmts, :(@inbounds outi[oo + $(k + 1)] = $(oi[k + 1])))
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end
