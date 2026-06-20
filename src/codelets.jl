# Stage 6: `@generated` straight-line DFT codelets — Julia's genfft-equivalent.
#
# `_codelet!` emits, at COMPILE time, a fully-unrolled size-R DFT: every input is a local,
# every twiddle a baked literal `Complex{T}` constant, every step an SSA assignment. LLVM then
# keeps the whole transform in registers (no recursion, no loop, no twiddle-table loads, W=1
# folded away) and FMA-fuses via `_cmul`. This is exactly what FFTW's `genfft` and rustfft's
# hand-written `Butterfly` structs provide; here it falls out of Julia metaprogramming — the
# concrete demonstration that "does Julia need to improve? no, it has @generated".
#
# Direction is a type parameter `Val{S}` (S = -1 forward, +1 inverse) so the twiddle literals
# are correct per direction while staying compile-time constant.

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
