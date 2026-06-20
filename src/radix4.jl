# Faithful port of rustfft's Radix4 algorithm (src/algorithm/radix4.rs) — the power-of-two
# workhorse, scalar (LLVM autovectorizes; matches Rust's scalar codegen per bench/lang_compare).
# This is the same-algorithm artifact for the full-package comparison against the rustfft crate,
# and the structural base for the AVX round (its Butterfly4 cross-pass is what gets SIMD-ized).
#
# N = 2ⁿ = 4^k · base, base ∈ {1,2,4,8,16,32} (32 if n odd, 16 if n even, for n≥4; else 2ⁿ):
#   1. bit-reversed transpose (radix-4 digit reversal of the width index) → contiguous base leaves
#   2. base butterflies: a straight-line size-`base` DFT per leaf (reuses @generated `_codelet!`)
#   3. log₄ Butterfly4 cross-passes with precomputed layered twiddles
# Correctness is validated against FFTW (not byte-matching rustfft's internal layout).

# choose base exponent like rustfft: 0..3 → 2ⁿ; n≥4 → 16 (even) or 32 (odd)
function _radix4_base(n_log2::Int)
    n_log2 <= 3 && return 1 << n_log2
    return iseven(n_log2) ? 16 : 32
end

@inline function _digitrev4(x::Int, k::Int)
    r = 0
    @inbounds for _ in 1:k
        r = (r << 2) | (x & 3)
        x >>= 2
    end
    return r
end

# multiply by ∓i (the radix-4 twist): forward S=-1 → ·(-i); inverse S=+1 → ·(+i)
@inline _twist(z::Complex, ::Val{-1}) = Complex(imag(z), -real(z))
@inline _twist(z::Complex, ::Val{1}) = Complex(-imag(z), real(z))

# bit-reversed transpose: out[xi*base + y] = x[y*width + digitrev4(xi)]
function _radix4_reorder!(out, x, base::Int, width::Int, k::Int)
    @inbounds for xi in 0:(width - 1)
        xr = _digitrev4(xi, k)
        for y in 0:(base - 1)
            out[xi * base + y + 1] = x[y * width + xr + 1]
        end
    end
    return
end

# Cache-blocked PLAIN transpose (no digit reversal): dst[w*base + y] = x[y*width + w].
# Contiguous reads of each x row-segment (vs the scattered stride-`width` gather of the naive
# reorder). The digit reversal is folded — for free — into the base butterfly's source offset.
function _radix4_transpose!(dst, x, base::Int, width::Int)
    blk = max(1, _L1_TILE ÷ base)
    @inbounds for wt in 0:blk:(width - 1)
        we = min(wt + blk, width)
        for y in 0:(base - 1)
            yo = y * width
            @simd for w in wt:(we - 1)
                dst[w * base + y + 1] = x[yo + w + 1]
            end
        end
    end
    return
end

# base butterflies reading the digit-reversed block from `src`, writing the natural block to `dst`
@inline function _base_butterflies_dr!(dst, src, base::Int, width::Int, k::Int, ::Val{S}) where {S}
    @match base begin
        32 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(dst, xi * 32, src, _digitrev4(xi, k) * 32, 1, Val(32), Val(S))
            end
        )
        16 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(dst, xi * 16, src, _digitrev4(xi, k) * 16, 1, Val(16), Val(S))
            end
        )
        8 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(dst, xi * 8, src, _digitrev4(xi, k) * 8, 1, Val(8), Val(S))
            end
        )
        4 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(dst, xi * 4, src, _digitrev4(xi, k) * 4, 1, Val(4), Val(S))
            end
        )
        2 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(dst, xi * 2, src, _digitrev4(xi, k) * 2, 1, Val(2), Val(S))
            end
        )
        _ => nothing
    end
    return
end

# type-stable base-butterfly dispatch (literal Vals → no dynamic dispatch)
@inline function _base_butterflies!(out, base::Int, width::Int, ::Val{S}) where {S}
    @match base begin
        32 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(out, xi * 32, out, xi * 32, 1, Val(32), Val(S))
            end
        )
        16 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(out, xi * 16, out, xi * 16, 1, Val(16), Val(S))
            end
        )
        8 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(out, xi * 8, out, xi * 8, 1, Val(8), Val(S))
            end
        )
        4 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(out, xi * 4, out, xi * 4, 1, Val(4), Val(S))
            end
        )
        2 => @inbounds(
            for xi in 0:(width - 1)
                _codelet!(out, xi * 2, out, xi * 2, 1, Val(2), Val(S))
            end
        )
        _ => nothing   # base == 1: identity
    end
    return
end

# layered twiddles: one Vector{Complex} per cross-pass, length 3L = [W_cur^j ; W_cur^2j ; W_cur^3j]
function _radix4_layers(::Type{Complex{T}}, n::Int, base::Int; inverse::Bool) where {T}
    s = inverse ? 2.0 : -2.0
    layers = Vector{Vector{Complex{T}}}()
    L = base
    while L < n
        cur = 4L
        v = Vector{Complex{T}}(undef, 3L)
        @inbounds for j in 0:(L - 1)
            v[j + 1] = Complex{T}(cispi(s * j / cur))
            v[L + j + 1] = Complex{T}(cispi(s * 2j / cur))
            v[2L + j + 1] = Complex{T}(cispi(s * 3j / cur))
        end
        push!(layers, v)
        L = cur
    end
    return layers
end

function _radix4_cross!(out, base::Int, n::Int, layers, ::Val{S}) where {S}
    L = base
    p = 1
    @inbounds while L < n
        cur = 4L
        tw = layers[p]
        for blk in 0:cur:(n - 1)
            @simd ivdep for j in 0:(L - 1)
                i0 = blk + j
                a = out[i0 + 1]
                b = _cmul(out[i0 + L + 1], tw[j + 1])
                c = _cmul(out[i0 + 2L + 1], tw[L + j + 1])
                d = _cmul(out[i0 + 3L + 1], tw[2L + j + 1])
                t0 = a + c
                t1 = a - c
                t2 = b + d
                t3 = _twist(b - d, Val(S))
                out[i0 + 1] = t0 + t2
                out[i0 + L + 1] = t1 + t3
                out[i0 + 2L + 1] = t0 - t2
                out[i0 + 3L + 1] = t1 - t3
            end
        end
        L = cur
        p += 1
    end
    return
end

# ---- SoA (split re/im) Radix4: the AVX-aimed path ------------------------------------------
# Same algorithm, but data and twiddles in separate real arrays so the Butterfly4 cross-pass is
# pure-real, unit-stride, `@simd ivdep` → shuffle-free AVX-512 (the pattern that hit ~38-57
# GFLOP/s as a batched kernel). Base butterflies reuse the generated SoA codelets.

function _radix4_layers_soa(::Type{T}, n::Int, base::Int; inverse::Bool) where {T}
    s = inverse ? 2.0 : -2.0
    lr = Vector{Vector{T}}()
    li = Vector{Vector{T}}()
    L = base
    while L < n
        cur = 4L
        vr = Vector{T}(undef, 3L)
        vi = Vector{T}(undef, 3L)
        @inbounds for j in 0:(L - 1)
            w1 = cispi(s * j / cur); vr[j + 1] = T(real(w1)); vi[j + 1] = T(imag(w1))
            w2 = cispi(s * 2j / cur); vr[L + j + 1] = T(real(w2)); vi[L + j + 1] = T(imag(w2))
            w3 = cispi(s * 3j / cur); vr[2L + j + 1] = T(real(w3)); vi[2L + j + 1] = T(imag(w3))
        end
        push!(lr, vr); push!(li, vi)
        L = cur
    end
    return lr, li
end

# Cache-blocked transpose AoS x → SoA (dr,di), fusing the deinterleave (split) into the pass.
function _radix4_transpose_soa!(dr, di, x, base::Int, width::Int)
    blk = max(1, _L1_TILE ÷ base)
    @inbounds for wt in 0:blk:(width - 1)
        we = min(wt + blk, width)
        for y in 0:(base - 1)
            yo = y * width
            @simd for w in wt:(we - 1)
                z = x[yo + w + 1]
                dr[w * base + y + 1] = real(z)
                di[w * base + y + 1] = imag(z)
            end
        end
    end
    return
end

# SoA base butterflies reading the digit-reversed block from (sr,si), writing to (outr,outi)
@inline function _base_butterflies_dr_soa!(
        outr, outi, sr, si, base::Int, width::Int, k::Int, ::Val{S}
    ) where {S}
    @match base begin
        32 => @inbounds(
            for xi in 0:(width - 1)
                _codelet_soa!(outr, outi, xi * 32, sr, si, _digitrev4(xi, k) * 32, 1, Val(32), Val(S))
            end
        )
        16 => @inbounds(
            for xi in 0:(width - 1)
                _codelet_soa!(outr, outi, xi * 16, sr, si, _digitrev4(xi, k) * 16, 1, Val(16), Val(S))
            end
        )
        8 => @inbounds(
            for xi in 0:(width - 1)
                _codelet_soa!(outr, outi, xi * 8, sr, si, _digitrev4(xi, k) * 8, 1, Val(8), Val(S))
            end
        )
        4 => @inbounds(
            for xi in 0:(width - 1)
                _codelet_soa!(outr, outi, xi * 4, sr, si, _digitrev4(xi, k) * 4, 1, Val(4), Val(S))
            end
        )
        2 => @inbounds(
            for xi in 0:(width - 1)
                _codelet_soa!(outr, outi, xi * 2, sr, si, _digitrev4(xi, k) * 2, 1, Val(2), Val(S))
            end
        )
        _ => nothing
    end
    return
end

function _radix4_reorder_soa!(outr, outi, xr, xi, base::Int, width::Int, k::Int)
    @inbounds for xi_ in 0:(width - 1)
        xrv = _digitrev4(xi_, k)
        for y in 0:(base - 1)
            outr[xi_ * base + y + 1] = xr[y * width + xrv + 1]
            outi[xi_ * base + y + 1] = xi[y * width + xrv + 1]
        end
    end
    return
end

@inline function _base_butterflies_soa!(outr, outi, base::Int, width::Int, ::Val{S}) where {S}
    @match base begin
        32 => @inbounds(
            for xi_ in 0:(width - 1)
                _codelet_soa!(outr, outi, xi_ * 32, outr, outi, xi_ * 32, 1, Val(32), Val(S))
            end
        )
        16 => @inbounds(
            for xi_ in 0:(width - 1)
                _codelet_soa!(outr, outi, xi_ * 16, outr, outi, xi_ * 16, 1, Val(16), Val(S))
            end
        )
        8 => @inbounds(
            for xi_ in 0:(width - 1)
                _codelet_soa!(outr, outi, xi_ * 8, outr, outi, xi_ * 8, 1, Val(8), Val(S))
            end
        )
        4 => @inbounds(
            for xi_ in 0:(width - 1)
                _codelet_soa!(outr, outi, xi_ * 4, outr, outi, xi_ * 4, 1, Val(4), Val(S))
            end
        )
        2 => @inbounds(
            for xi_ in 0:(width - 1)
                _codelet_soa!(outr, outi, xi_ * 2, outr, outi, xi_ * 2, 1, Val(2), Val(S))
            end
        )
        _ => nothing
    end
    return
end

function _radix4_cross_soa!(outr, outi, base::Int, n::Int, lr, li, ::Val{S}) where {S}
    L = base
    p = 1
    @inbounds while L < n
        cur = 4L
        twr = lr[p]; twi = li[p]
        for blk in 0:cur:(n - 1)
            @simd ivdep for j in 0:(L - 1)
                i0 = blk + j
                ar = outr[i0 + 1]; ai = outi[i0 + 1]
                b0r = outr[i0 + L + 1]; b0i = outi[i0 + L + 1]
                wr1 = twr[j + 1]; wi1 = twi[j + 1]
                br = muladd(b0r, wr1, -b0i * wi1); bi = muladd(b0r, wi1, b0i * wr1)
                c0r = outr[i0 + 2L + 1]; c0i = outi[i0 + 2L + 1]
                wr2 = twr[L + j + 1]; wi2 = twi[L + j + 1]
                cr = muladd(c0r, wr2, -c0i * wi2); ci = muladd(c0r, wi2, c0i * wr2)
                d0r = outr[i0 + 3L + 1]; d0i = outi[i0 + 3L + 1]
                wr3 = twr[2L + j + 1]; wi3 = twi[2L + j + 1]
                dr = muladd(d0r, wr3, -d0i * wi3); di = muladd(d0r, wi3, d0i * wr3)
                t0r = ar + cr; t0i = ai + ci
                t1r = ar - cr; t1i = ai - ci
                t2r = br + dr; t2i = bi + di
                bdr = br - dr; bdi = bi - di
                if S == -1            # ·(-i)
                    t3r = bdi; t3i = -bdr
                else                  # ·(+i)
                    t3r = -bdi; t3i = bdr
                end
                outr[i0 + 1] = t0r + t2r; outi[i0 + 1] = t0i + t2i
                outr[i0 + L + 1] = t1r + t3r; outi[i0 + L + 1] = t1i + t3i
                outr[i0 + 2L + 1] = t0r - t2r; outi[i0 + 2L + 1] = t0i - t2i
                outr[i0 + 3L + 1] = t1r - t3r; outi[i0 + 3L + 1] = t1i - t3i
            end
        end
        L = cur
        p += 1
    end
    return
end

"""
    Radix4SoAPlan{T} <: AbstractFFTPlan{T}

SoA (split re/im) Radix4 — the AVX-aimed variant (`:radix4simd`). Shuffle-free vectorized
Butterfly4 cross-pass; allocation-free.
"""
struct Radix4SoAPlan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    base::Int
    k::Int
    lr::Vector{Vector{T}}
    li::Vector{Vector{T}}
    xr::Vector{T}
    xi::Vector{T}
    outr::Vector{T}
    outi::Vector{T}
end

function Radix4SoAPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ispow2(n) || throw(ArgumentError(":radix4simd supports power-of-two sizes only; got n=$n"))
    nl = trailing_zeros(Int(n))
    base = _radix4_base(nl)
    k = (nl - trailing_zeros(base)) ÷ 2
    lr, li = _radix4_layers_soa(T, Int(n), base; inverse)
    z = () -> Vector{T}(undef, Int(n))
    return Radix4SoAPlan{T}(Int(n), inverse, base, k, lr, li, z(), z(), z(), z())
end

plan_length(p::Radix4SoAPlan)::Int = p.n
plan_inverse(p::Radix4SoAPlan)::Bool = p.inverse

function apply_unnormalized!(p::Radix4SoAPlan{T}, x::AbstractVector) where {T}
    n = p.n
    n <= 1 && return x
    xr, xi, outr, outi = p.xr, p.xi, p.outr, p.outi
    width = n ÷ p.base
    # cache-blocked transpose AoS x → SoA (xr,xi), fusing the split; base butterflies read the
    # digit-reversed block into (outr,outi); shuffle-free SoA cross-passes; merge back to x.
    _radix4_transpose_soa!(xr, xi, x, p.base, width)
    if p.inverse
        _base_butterflies_dr_soa!(outr, outi, xr, xi, p.base, width, p.k, Val(1))
        _radix4_cross_soa!(outr, outi, p.base, n, p.lr, p.li, Val(1))
    else
        _base_butterflies_dr_soa!(outr, outi, xr, xi, p.base, width, p.k, Val(-1))
        _radix4_cross_soa!(outr, outi, p.base, n, p.lr, p.li, Val(-1))
    end
    @inbounds @simd for i in 1:n
        x[i] = Complex(outr[i], outi[i])
    end
    return x
end

"""
    Radix4Plan{T} <: AbstractFFTPlan{T}

Faithful port of rustfft's Radix4 (power-of-two `n`). Preallocated layered twiddles + scratch →
allocation-free. AoS scalar (autovectorized via `@simd ivdep`); see also [`Radix4SoAPlan`](@ref).
"""
struct Radix4Plan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    base::Int
    k::Int
    layers::Vector{Vector{Complex{T}}}
    scratch::Vector{Complex{T}}
end

function Radix4Plan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ispow2(n) || throw(ArgumentError(":radix4 supports power-of-two sizes only; got n=$n"))
    nl = trailing_zeros(Int(n))
    base = _radix4_base(nl)
    k = (nl - trailing_zeros(base)) ÷ 2
    layers = _radix4_layers(Complex{T}, Int(n), base; inverse)
    return Radix4Plan{T}(Int(n), inverse, base, k, layers, Vector{Complex{T}}(undef, Int(n)))
end

plan_length(p::Radix4Plan)::Int = p.n
plan_inverse(p::Radix4Plan)::Bool = p.inverse

function apply_unnormalized!(p::Radix4Plan{T}, x::AbstractVector) where {T}
    n = p.n
    n <= 1 && return x
    scr = p.scratch
    width = n ÷ p.base
    # cache-blocked plain transpose into scratch, then base butterflies read the digit-reversed
    # block (folding the reorder in for free) and write back to x; cross-passes finish in x.
    _radix4_transpose!(scr, x, p.base, width)
    if p.inverse
        _base_butterflies_dr!(x, scr, p.base, width, p.k, Val(1))
        _radix4_cross!(x, p.base, n, p.layers, Val(1))
    else
        _base_butterflies_dr!(x, scr, p.base, width, p.k, Val(-1))
        _radix4_cross!(x, p.base, n, p.layers, Val(-1))
    end
    return x
end
