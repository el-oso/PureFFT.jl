# Stage 8: Bluestein's algorithm (chirp-Z transform) for awkward sizes.
#
# A length-`n` DFT is rewritten as a length-`M` circular convolution by the identity
# jk = (j² + k² − (k−j)²)/2:
#
#   X[k] = w[k] · Σ_j (x[j]·w[j]) · v[k−j],   w[k] = exp(s·iπ k²/n),  v[m] = conj(w[m]),  s = ∓1
#
# The convolution is evaluated with three power-of-two FFTs (M = next_pow2(2n−1)) — the fast
# Radix4Avx path. This replaces the O(n²) direct-DFT that `mixedradix` falls back to on a large
# prime factor (the ~0 GFLOP/s cliff) with O(n log n). The kernel FFT `B = FFT(v)/M` is
# precomputed once at plan time (with 1/M folded in), so a transform is two FFTs + two scalings.

"""
    BluesteinPlan{T} <: AbstractFFTPlan{T}

Chirp-Z plan for an arbitrary length `n`. Holds the chirp `w` (length `n`), the precomputed
kernel spectrum `B` (length `M`, with the inverse-FFT 1/M folded in), forward/inverse
power-of-two convolution plans, and a reusable work buffer. Direction is fixed at plan time.
"""
struct BluesteinPlan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    M::Int
    w::Vector{Complex{T}}        # chirp: w[k] = exp(s·iπ k²/n), k = 0:n-1
    B::Vector{Complex{T}}        # FFT(kernel)/M, length M
    fwd::Radix4AvxPlan{T}        # length-M forward FFT (convolution)
    invp::Radix4AvxPlan{T}       # length-M inverse FFT (unnormalized)
    abuf::Vector{Complex{T}}     # work buffer, length M
end

# k²/n reduced mod 2 (in π units) keeps the chirp angle small for accuracy; widemul avoids
# Int overflow of k² for large n.
@inline _chirp_angle(::Type{T}, k::Int, n::Int) where {T} = T(Int(widemul(k, k) % (2n))) / T(n)

function BluesteinPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    n = Int(n)
    n >= 1 || throw(ArgumentError("BluesteinPlan needs n ≥ 1; got n=$n"))
    M = nextpow(2, max(2n - 1, 1))
    s = inverse ? one(T) : -one(T)

    w = Vector{Complex{T}}(undef, n)
    @inbounds for k in 0:(n - 1)
        w[k + 1] = cispi(s * _chirp_angle(T, k, n))
    end

    # kernel v[m] = conj(w[m]), placed at both ends of the length-M buffer (v is symmetric,
    # v[-m] = v[m]); the central [n, M-n] band stays zero (the convolution zero-padding).
    B = zeros(Complex{T}, M)
    @inbounds for m in 0:(n - 1)
        vm = conj(w[m + 1])
        B[m + 1] = vm
        m != 0 && (B[M - m + 1] = vm)
    end

    fwd = Radix4AvxPlan(Complex{T}, M; inverse = false)
    invp = Radix4AvxPlan(Complex{T}, M; inverse = true)

    apply_unnormalized!(fwd, B)              # B ← FFT(v)
    invM = one(T) / T(M)
    @inbounds @simd for i in eachindex(B)    # fold the inverse-FFT 1/M into the kernel
        B[i] *= invM
    end

    return BluesteinPlan{T}(n, inverse, M, w, B, fwd, invp, Vector{Complex{T}}(undef, M))
end

plan_length(p::BluesteinPlan)::Int = p.n
plan_inverse(p::BluesteinPlan)::Bool = p.inverse

function apply_unnormalized!(p::BluesteinPlan{T}, x::AbstractVector) where {T}
    n = p.n
    M = p.M
    a = p.abuf
    w = p.w
    @inbounds begin
        @simd for k in 1:n                   # a[k] = x[k]·w[k], zero-padded to M
            a[k] = x[k] * w[k]
        end
        for k in (n + 1):M
            a[k] = zero(Complex{T})
        end
    end
    apply_unnormalized!(p.fwd, a)            # a ← FFT(a)
    @inbounds @simd for i in 1:M             # pointwise multiply by the kernel spectrum
        a[i] *= p.B[i]
    end
    apply_unnormalized!(p.invp, a)           # a ← IFFT_unnorm(...) = (x·w) ⊛ v
    @inbounds @simd for k in 1:n             # X[k] = w[k]·conv[k]
        x[k] = w[k] * a[k]
    end
    return x
end
