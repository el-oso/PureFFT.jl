# Stage 5: split-layout (SoA) recursive FFT.
#
# Same cache-oblivious recursive DIT as Stage 4, but real and imaginary parts live in
# SEPARATE arrays. The combine loop is then pure real arithmetic — the autovectorizer fills
# AVX-512 lanes with no re/im deinterleave shuffles, which measured ~1.42× faster than the
# AoS combine in isolation. Cost: deinterleave on entry + interleave on exit (two O(n)
# passes; later fused into the first/last codelet). A dedicated `SoAPlan` carries the split
# buffers and split twiddles, and satisfies the `AbstractFFTPlan` contract.

@inline _soacmul(pr, pii, wr, wi) = (muladd(pr, wr, -pii * wi), muladd(pr, wi, pii * wr))

"""
    split_twiddles(T, n; inverse=false) -> (twr, twi)

Per-level twiddles split into real/imag arrays. `twr[L][k+1] = real(W_{2^L}^k)` etc.
"""
function split_twiddles(::Type{T}, n::Integer; inverse::Bool = false) where {T}
    nst = trailing_zeros(Int(n))
    twr = Vector{Vector{T}}(undef, nst)
    twi = Vector{Vector{T}}(undef, nst)
    s = inverse ? 2.0 : -2.0
    len = 2
    for si in 1:nst
        half = len >> 1
        vr = Vector{T}(undef, half)
        vi = Vector{T}(undef, half)
        @inbounds for j in 0:(half - 1)
            w = cispi(s * j / len)
            vr[j + 1] = T(real(w))
            vi[j + 1] = T(imag(w))
        end
        twr[si] = vr
        twi[si] = vi
        len <<= 1
    end
    return twr, twi
end

# Generated SoA codelet leaves (base ≤ 32), then a pure-real shuffle-free combine.
@inline function _leaf_soa!(orr, ori, oo, xr, xi, off, str, n, ::Val{S}) where {S}
    if n == 32
        _codelet_soa!(orr, ori, oo, xr, xi, off, str, Val(32), Val(S))
    elseif n == 16
        _codelet_soa!(orr, ori, oo, xr, xi, off, str, Val(16), Val(S))
    elseif n == 8
        _codelet_soa!(orr, ori, oo, xr, xi, off, str, Val(8), Val(S))
    elseif n == 4
        _codelet_soa!(orr, ori, oo, xr, xi, off, str, Val(4), Val(S))
    elseif n == 2
        _codelet_soa!(orr, ori, oo, xr, xi, off, str, Val(2), Val(S))
    else # n == 1
        @inbounds (orr[oo + 1] = xr[off + 1]; ori[oo + 1] = xi[off + 1])
    end
    return
end

function _ditrec_soa!(orr, ori, oo, xr, xi, off, str, n, L, twr, twi, ::Val{S}) where {S}
    if n <= RECURSE_BASE
        _leaf_soa!(orr, ori, oo, xr, xi, off, str, n, Val(S))
        return
    end
    n2 = n >> 1
    s2 = str << 1
    _ditrec_soa!(orr, ori, oo, xr, xi, off, s2, n2, L - 1, twr, twi, Val(S))
    _ditrec_soa!(orr, ori, oo + n2, xr, xi, off + str, s2, n2, L - 1, twr, twi, Val(S))
    wr = twr[L]
    wi = twi[L]
    @inbounds @simd for k in 1:n2
        er = orr[oo + k]; ei = ori[oo + k]
        pr = orr[oo + n2 + k]; pii = ori[oo + n2 + k]
        tr = muladd(pr, wr[k], -pii * wi[k])
        ti = muladd(pr, wi[k], pii * wr[k])
        orr[oo + k] = er + tr; ori[oo + k] = ei + ti
        orr[oo + n2 + k] = er - tr; ori[oo + n2 + k] = ei - ti
    end
    return
end

"""
    SoAPlan{T} <: AbstractFFTPlan{T}

Split-layout recursive FFT plan (power-of-two `n`). Preallocates split input/output buffers
and split twiddles; the transform is allocation-free.
"""
struct SoAPlan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    twr::Vector{Vector{T}}
    twi::Vector{Vector{T}}
    xr::Vector{T}
    xi::Vector{T}
    outr::Vector{T}
    outi::Vector{T}
end

function SoAPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ispow2(n) || throw(ArgumentError(":soa supports power-of-two sizes only; got n=$n"))
    twr, twi = split_twiddles(T, n; inverse)
    z = () -> Vector{T}(undef, Int(n))
    return SoAPlan{T}(Int(n), inverse, twr, twi, z(), z(), z(), z())
end

plan_length(p::SoAPlan)::Int = p.n
plan_inverse(p::SoAPlan)::Bool = p.inverse

function apply_unnormalized!(p::SoAPlan{T}, x::AbstractVector) where {T}
    n = p.n
    n <= 1 && return x
    xr, xi, orr, ori = p.xr, p.xi, p.outr, p.outi
    @inbounds @simd for i in 1:n          # deinterleave AoS → SoA
        xr[i] = real(x[i]); xi[i] = imag(x[i])
    end
    L = trailing_zeros(n)
    if p.inverse
        _ditrec_soa!(orr, ori, 0, xr, xi, 0, 1, n, L, p.twr, p.twi, Val(1))
    else
        _ditrec_soa!(orr, ori, 0, xr, xi, 0, 1, n, L, p.twr, p.twi, Val(-1))
    end
    @inbounds @simd for i in 1:n          # interleave SoA → AoS
        x[i] = Complex(orr[i], ori[i])
    end
    return x
end
