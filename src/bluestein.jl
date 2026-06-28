# Stage 8: Bluestein's algorithm (chirp-Z transform) for awkward sizes.
#
# A length-`n` DFT is rewritten as a length-`M` circular convolution by the identity
# jk = (j² + k² − (k−j)²)/2:
#
#   X[k] = w[k] · Σ_j (x[j]·w[j]) · v[k−j],   w[k] = exp(s·iπ k²/n),  v[m] = conj(w[m]),  s = ∓1
#
# The convolution is evaluated with three length-M FFTs. M only needs to be ≥ 2n−1; rather than the
# next power of two (which can be 31% above that floor — e.g. n=99991 → M=262144 vs the 2n−1 floor of
# 199981), we pick the smaller of {next pow2, smallest 2·3·5-smooth ≥ 2n−1} by TIMING the forward FFT
# of each (the 2·3·5-smooth M, e.g. 200000=2⁶·5⁵, sits ~0% over the floor and now FFTs fast via the
# AvxMixedRadix radix-5 path — a ~1.2× win on the inner FFT, which dominates). This replaces the O(n²)
# direct-DFT that `mixedradix` falls back to on a large prime factor (the ~0 GFLOP/s cliff) with
# O(n log n). The kernel FFT `B = FFT(v)/M` is precomputed once at plan time (with 1/M folded in), so a
# transform is two FFTs + two scalings.

"""
    BluesteinPlan{T} <: AbstractFFTPlan{T}

Chirp-Z plan for an arbitrary length `n`. Holds the chirp `w` (length `n`), the precomputed
kernel spectrum `B` (length `M`, with the inverse-FFT 1/M folded in), forward/inverse
power-of-two convolution plans, and a reusable work buffer. Direction is fixed at plan time.
"""
# Parametric on the inner forward/inverse plan types (FP/IP) so `apply_unnormalized!` stays concrete /
# type-stable / alloc-free (StrictMode asserts this) even though which plan wins is chosen at plan time —
# pow2 → Radix4AvxPlan, smooth → AvxMixedRadixPlan. Same pattern as `BatchedRaderDim` in ndim.jl.
struct BluesteinPlan{T, FP, IP} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    M::Int
    w::Vector{Complex{T}}        # chirp: w[k] = exp(s·iπ k²/n), k = 0:n-1
    B::Vector{Complex{T}}        # FFT(kernel)/M, length M
    fwd::FP                      # length-M forward FFT (convolution)
    invp::IP                     # length-M inverse FFT (unnormalized)
    abuf::Vector{Complex{T}}     # work buffer, length M
end

# Candidate inner-FFT lengths M ≥ floorM: every 2·3·5-smooth integer in [floorM, 1.12·floorM] plus the
# next power of two. The smallest-smooth M isn't the fastest — it's often 5-heavy (e.g. 200000=2⁶·5⁵, a
# slow radix-5 tree); a slightly larger but more 2-heavy M (e.g. 207360=2⁹·3⁴·5) FFTs ~1.2× faster. We
# can't predict which wins, so the constructor times the forward FFT of each (cheap: ~a dozen candidates,
# all one-time plan cost) and keeps the fastest. (7-smooth is excluded — radix-7 trees never won and only
# tripled the candidate count / plan-build time.)
function _bluestein_Ms(floorM::Int, Mpow::Int)
    hi = min(Mpow, ceil(Int, 1.12 * floorM))
    out = Int[]
    p2 = 1
    while p2 <= hi
        p3 = p2
        while p3 <= hi
            p5 = p3
            while p5 <= hi
                p5 >= floorM && push!(out, p5)
                p5 *= 5
            end
            p3 *= 3
        end
        p2 *= 2
    end
    Mpow in out || push!(out, Mpow)
    return sort!(unique!(out))
end

# k²/n reduced mod 2 (in π units) keeps the chirp angle small for accuracy; widemul avoids
# Int overflow of k² for large n.
@inline _chirp_angle(::Type{T}, k::Int, n::Int) where {T} = T(Int(widemul(k, k) % (2n))) / T(n)

function BluesteinPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    n = Int(n)
    n >= 1 || throw(ArgumentError("BluesteinPlan needs n ≥ 1; got n=$n"))
    s = inverse ? one(T) : -one(T)

    # Pick M ≥ 2n−1 by timing the forward FFT of each smooth candidate (pow2 → Radix4Avx; 2·3·5·7-smooth
    # → AvxMixedRadix, `nothing` when unsupported, e.g. Float32 or no base-2 factor → skipped) and keeping
    # the fastest. The pow2 is always a candidate, so worst case is the previous behaviour (no regression).
    floorM = max(2n - 1, 1)
    Mpow = nextpow(2, floorM)
    local M::Int, fwd
    bestt = Inf
    for Mc in _bluestein_Ms(floorM, Mpow)             # ascending; smallest (near-floor) M tried first
        p = Mc == Mpow ? Radix4AvxPlan(Complex{T}, Mc; inverse = false) :
            AvxMixedRadixPlan(Complex{T}, Mc; inverse = false)   # `nothing` if unsupported (skip)
        isnothing(p) && continue
        t = _besttime(p, randn(Complex{T}, Mc))
        # Require a clear >3% margin to switch to a larger M: favours the smaller candidate and keeps the
        # noisy plan-time timings from flip-flopping between near-equal candidates (a few % apart).
        if t < bestt * 0.97
            bestt = t; M = Mc; fwd = p
        end
    end
    invp = M == Mpow ? Radix4AvxPlan(Complex{T}, M; inverse = true) :
           something(AvxMixedRadixPlan(Complex{T}, M; inverse = true))

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

    apply_unnormalized!(fwd, B)              # B ← FFT(v)
    invM = one(T) / T(M)
    @inbounds @simd for i in eachindex(B)    # fold the inverse-FFT 1/M into the kernel
        B[i] *= invM
    end

    return BluesteinPlan{T, typeof(fwd), typeof(invp)}(
        n, inverse, M, w, B, fwd, invp, Vector{Complex{T}}(undef, M))
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
