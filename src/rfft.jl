# Real-input FFT (rfft / irfft) using the half-complex trick.
#
# FORWARD (rfft):
#   For real input x of length n (even), pack pairs into m=n/2 complex values
#     z[j] = x[2j] + i·x[2j+1],  j = 0 … m-1
#   run ONE size-m complex FFT Z = fft(z), then recombine via:
#     Xe[k] = (Z[k] + conj(Z[m-k])) / 2        (even-index DFT half)
#     Xo[k] = (Z[k] - conj(Z[m-k])) / (2i)     (odd-index DFT half)
#     X[k]  = Xe[k] + W_n^k · Xo[k]            k = 0 … m,  W_n^k = exp(-2πik/n)
#   Boundary cases (real-valued):  X[0] = Re(Z[0])+Im(Z[0]),  X[m] = Re(Z[0])-Im(Z[0])
#
# INVERSE (irfft):
#   Given X[0..m], reconstruct Z[0..m-1] using the inverse recombination:
#     Z[0] = ((X[0]+X[m])/2,  (X[0]-X[m])/2)   (boundary k=0)
#     Z[k] = (X[k] + conj(X[m-k])) / 2
#           + i * (X[k] - conj(X[m-k])) / (2 * W_n^k)   for k = 1..m-1
#   Note: X[m-k] here refers to stored rfft bin m-k (NOT conj(X[k])) — z is complex.
#   Apply size-m complex IFFT (via conj-FFT-conj trick, normalise by 1/m), unpack.
#   Normalization: divides by m (= n/2), matching FFTW's irfft convention
#   irfft(rfft(x), n) == x.
#
# Recombination twiddles W_n^k for k=0..m are precomputed in the plan; work buffers
# are preallocated so the hot path is allocation-free.

import AbstractFFTs

"""
    RealFFTPlan{T, P}

Plan for an efficient real-input forward FFT of even length `n`.
Holds a size-`n÷2` inner complex plan (concrete type P), the precomputed twiddle table, and
preallocated work buffers — the hot path performs zero heap allocations.
"""
struct RealFFTPlan{T, P <: AbstractFFTPlan{T}}
    n::Int                        # real input length (even)
    inner::P                      # size-m = n/2 complex FFT plan (concrete)
    twiddles::Vector{Complex{T}}  # W_n^k for k = 0 .. m  (index k+1)
    zbuf::Vector{Complex{T}}      # packing buffer, length m
    outbuf::Vector{Complex{T}}    # recombined output, length m+1
end

"""
    RealIFFTPlan{T, P}

Plan for the inverse real FFT (irfft): complex half-spectrum → real, length `n`.
"""
struct RealIFFTPlan{T, P <: AbstractFFTPlan{T}}
    n::Int
    inner::P                      # size-m inverse complex FFT plan (concrete)
    twiddles::Vector{Complex{T}}  # conj(W_n^k) for k = 0 .. m
    zbuf::Vector{Complex{T}}      # inverse-recombined complex, length m
end

# ── Plan constructors ────────────────────────────────────────────────────────

"""
    plan_prfft(x::AbstractVector{<:Real}) -> RealFFTPlan
    plan_prfft(T, n)

Build a forward real FFT plan for real inputs of even length `n`.
"""
function plan_prfft(::Type{T}, n::Integer) where {T <: AbstractFloat}
    iseven(n) || throw(ArgumentError("plan_prfft requires even n; got n=$n"))
    m = n ÷ 2
    inner = plan_pfft(Complex{T}, m; variant = :fast, inverse = false)
    twiddles = _rfft_twiddles(T, n)
    zbuf = Vector{Complex{T}}(undef, m)
    outbuf = Vector{Complex{T}}(undef, m + 1)
    return RealFFTPlan{T, typeof(inner)}(Int(n), inner, twiddles, zbuf, outbuf)
end

plan_prfft(x::AbstractVector{<:Real}) = plan_prfft(float(eltype(x)), length(x))

"""
    plan_pirfft(X::AbstractVector{<:Complex}, n) -> RealIFFTPlan
    plan_pirfft(T, n)

Build an inverse real FFT plan (irfft): half-spectrum of length `n÷2+1` → real length `n`.
"""
function plan_pirfft(::Type{T}, n::Integer) where {T <: AbstractFloat}
    iseven(n) || throw(ArgumentError("plan_pirfft requires even n; got n=$n"))
    m = n ÷ 2
    # Use the forward (unnormalized) inner plan: the conj-FFT-conj trick gives the
    # unnormalized IFFT; we apply the 1/m normalization ourselves during unpack.
    inner = plan_pfft(Complex{T}, m; variant = :fast, inverse = false)
    twiddles = _rfft_twiddles(T, n)   # forward twiddles W_n^k; irfft uses their conj
    zbuf = Vector{Complex{T}}(undef, m)
    return RealIFFTPlan{T, typeof(inner)}(Int(n), inner, twiddles, zbuf)
end

plan_pirfft(X::AbstractVector{<:Complex}, n::Integer) =
    plan_pirfft(real(float(eltype(X))), n)

# Precompute W_n^k = exp(-2πik/n) for k = 0 .. n/2  (n/2+1 values).
# Index convention: twiddles[k+1] = W_n^k.
function _rfft_twiddles(::Type{T}, n::Integer) where {T}
    m = n ÷ 2
    tbl = Vector{Complex{T}}(undef, m + 1)
    @inbounds for k in 0:m
        tbl[k + 1] = Complex{T}(cispi(-2.0 * k / n))
    end
    return tbl
end

# ── Forward rfft apply ───────────────────────────────────────────────────────

"""
    apply_rfft!(p::RealFFTPlan{T}, x::AbstractVector{T}, out::AbstractVector{Complex{T}})

In-place (into `out`) real FFT apply. `x` must have length `p.n`; `out` must have length
`p.n÷2 + 1`. No allocations after plan creation.
"""
function apply_rfft!(
        p::RealFFTPlan{T},
        x::AbstractVector{T},
        out::AbstractVector{Complex{T}},
    ) where {T}
    n = p.n
    m = n ÷ 2
    length(x) == n ||
        throw(DimensionMismatch("rfft plan length $n ≠ input length $(length(x))"))
    length(out) == m + 1 ||
        throw(DimensionMismatch("rfft output buffer must have length $(m + 1)"))

    z = p.zbuf
    tw = p.twiddles

    # Step 1: pack x[2j], x[2j+1] → z[j]
    @inbounds for j in 0:(m - 1)
        z[j + 1] = Complex{T}(x[2j + 1], x[2j + 2])
    end

    # Step 2: in-place size-m complex FFT (forward, unnormalized)
    apply_unnormalized!(p.inner, z)

    # Step 3: boundary k=0 and k=m
    z0 = z[1]
    @inbounds out[1] = Complex{T}(real(z0) + imag(z0), zero(T))
    @inbounds out[m + 1] = Complex{T}(real(z0) - imag(z0), zero(T))

    # Step 4: recombination for k = 1 .. m-1
    @inbounds for k in 1:(m - 1)
        zk = z[k + 1]
        zm_k = conj(z[m - k + 1])
        xe = (zk + zm_k) * T(0.5)
        xo = (zk - zm_k) * Complex{T}(zero(T), T(-0.5))  # (zk - zm_k)/(2i)
        out[k + 1] = xe + tw[k + 1] * xo
    end

    return out
end

# ── Inverse irfft apply ──────────────────────────────────────────────────────

"""
    apply_irfft!(p::RealIFFTPlan{T}, X::AbstractVector{Complex{T}}, out::AbstractVector{T})

In-place (into `out`) inverse real FFT. `X` must have length `p.n÷2+1`; `out` length `p.n`.
Normalizes by `1/(n÷2)` so that `irfft(rfft(x), n) == x` (FFTW convention).
"""
function apply_irfft!(
        p::RealIFFTPlan{T},
        X::AbstractVector{Complex{T}},
        out::AbstractVector{T},
    ) where {T}
    n = p.n
    m = n ÷ 2
    length(X) == m + 1 ||
        throw(DimensionMismatch("irfft plan expects $(m + 1) inputs; got $(length(X))"))
    length(out) == n ||
        throw(DimensionMismatch("irfft output buffer must have length $n"))

    z = p.zbuf
    tw = p.twiddles  # W_n^k = exp(-2πik/n), k = 0..m

    # Inverse recombination: reconstruct Z[k] from X[0..m].
    # The correct formula (derived from the 2×2 linear system for Z[k] and Z[m-k]):
    #   Z[k] = (X[k] + conj(X[m-k])) / 2 + im * (X[k] - conj(X[m-k])) / (2 * W_n^k)
    # Note: division by W_n^k = multiplication by conj(W_n^k) since |W_n^k| = 1.

    # k=0: X[0] = Re(Z[0])+Im(Z[0]),  X[m] = Re(Z[0])-Im(Z[0])
    x0 = real(X[1])
    xm = real(X[m + 1])
    @inbounds z[1] = Complex{T}((x0 + xm) * T(0.5), (x0 - xm) * T(0.5))

    @inbounds for k in 1:(m - 1)
        xk = X[k + 1]
        xmk = conj(X[m - k + 1])          # conj(X[m-k])
        sum_half = (xk + xmk) * T(0.5)
        diff = (xk - xmk)
        # im * diff / (2 * W_n^k) = im * diff * conj(W_n^k) / 2  (since |W_n^k|=1)
        # conj(W_n^k) = tw[k+1]^* stored as conj of twiddle
        # But tw[k+1] = W_n^k = exp(-2πik/n), so conj(tw[k+1]) = exp(+2πik/n)
        # im * diff * conj(tw[k+1]) / 2:
        w_conj = conj(tw[k + 1])           # conj(W_n^k) = exp(+2πik/n)
        # im * z = im * (a + ib) = -b + ia
        iw_diff = diff * w_conj            # temp: (X[k]-conj(X[m-k])) * conj(W_n^k)
        # multiply by im/2: real part of result = -imag(iw_diff)/2, imag = real(iw_diff)/2
        z[k + 1] = sum_half + Complex{T}(-imag(iw_diff) * T(0.5), real(iw_diff) * T(0.5))
    end

    # Compute unnormalized IFFT via the conj-FFT-conj identity:
    #   IFFT_unnorm(z) = conj(FFT(conj(z)))
    # Then divide by m to get the proper IFFT_m(Z) with 1/m normalization.
    # (The 1/m * 1/2 = 1/n factor arises from the packing.)
    @inbounds for j in 1:m
        z[j] = conj(z[j])
    end
    apply_unnormalized!(p.inner, z)
    # Unpack, conjugate, and normalize by 1/m simultaneously.
    invm = inv(T(m))
    @inbounds for j in 0:(m - 1)
        zj = conj(z[j + 1])
        out[2j + 1] = real(zj) * invm
        out[2j + 2] = imag(zj) * invm
    end

    return out
end

# ── High-level convenience API ───────────────────────────────────────────────

"""
    prfft(x::AbstractVector{<:Real}) -> Vector{Complex}

Forward real FFT: returns the `n÷2 + 1` unique (non-redundant) complex coefficients.
Equivalent to FFTW.rfft(x) (and AbstractFFTs.rfft via PureFFT).
"""
function prfft(x::AbstractVector{<:Real})
    T = float(eltype(x))
    n = length(x)
    p = plan_prfft(T, n)
    out = Vector{Complex{T}}(undef, n ÷ 2 + 1)
    return apply_rfft!(p, T.(x), out)
end

"""
    pirfft(X::AbstractVector{<:Complex}, n::Integer) -> Vector{<:Real}

Inverse real FFT: given `n÷2 + 1` complex coefficients, reconstruct the length-`n`
real signal. Normalizes by `1/(n÷2)` so that `pirfft(prfft(x), n) == x` (FFTW convention).
"""
function pirfft(X::AbstractVector{<:Complex}, n::Integer)
    T = real(float(eltype(X)))
    p = plan_pirfft(T, n)
    out = Vector{T}(undef, n)
    Xc = Complex{T}.(X)
    return apply_irfft!(p, Xc, out)
end

# ── AbstractFFTs integration ─────────────────────────────────────────────────
# Minimal wrappers so that AbstractFFTs.rfft / irfft route through PureFFT.

AbstractFFTs.rfft(x::AbstractVector{<:Real}) = prfft(x)
AbstractFFTs.irfft(X::AbstractVector{<:Complex}, n::Integer) = pirfft(X, n)
