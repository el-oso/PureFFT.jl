# Stage 11: Rader's algorithm for prime-length DFTs.
#
# For a prime p, the size-p DFT (minus the k=0 / j=0 terms) is a cyclic convolution of length
# p-1, indexed by the discrete logarithm w.r.t. a primitive root g of (Z/p)*:
#
#   X[g^r] = x[0] + Σ_q x[g^{-q}] · ω^{g^{r-q}}   = x[0] + (a ⊛ b)[r],   ω = exp(s·2πi/p)
#
# with a_q = x[g^{-q}], b_q = ω^{g^q}. The length-(p-1) convolution is two FFTs of length p-1
# (the kernel FFT B = FFT(b) is precomputed). When p-1 is smooth this beats Bluestein (whose
# pow2 M ≈ 2-4p is larger): ~2× on primes with p-1 = 2^a·3^b. `autoplan` routes here only when
# that holds (p ≥ 128, largest prime factor of p-1 ≤ 5); otherwise Bluestein.

"""
    RaderPlan{T} <: AbstractFFTPlan{T}

Prime-length DFT via Rader's algorithm: a length-(p-1) cyclic convolution evaluated with the fast
`:fast` path (four-step / radix4avx for the smooth p-1). Direction fixed at plan time.
"""
struct RaderPlan{T, PF, PI} <: AbstractFFTPlan{T}
    p::Int
    inverse::Bool
    ain::Vector{Int}              # gather: a_q = x[ain[q]] (= x[g^{-q} mod p])
    kout::Vector{Int}            # scatter: X[kout[r]] = x[0] + conv[r] (= X[g^r mod p])
    B::Vector{Complex{T}}        # FFT(b), length p-1
    fwd::PF                      # length-(p-1) forward plan
    invp::PI                     # length-(p-1) inverse plan (normalizes by 1/(p-1))
    abuf::Vector{Complex{T}}     # work buffer, length p-1
end

# smallest primitive root of an odd prime p
function _primitive_root(p::Int)
    m = p - 1
    # distinct prime factors of p-1
    fs = Int[]
    k = m; f = 2
    while f * f <= k
        if k % f == 0
            push!(fs, f)
            while k % f == 0
                k ÷= f
            end
        end
        f += 1
    end
    k > 1 && push!(fs, k)
    for g in 2:(p - 1)
        if all(q -> powermod(g, m ÷ q, p) != 1, fs)
            return g
        end
    end
    error("no primitive root for p=$p")   # unreachable for prime p
end

function RaderPlan(::Type{Complex{T}}, p::Integer; inverse::Bool = false) where {T}
    p = Int(p)
    L = p - 1
    s = inverse ? one(T) : -one(T)
    g = _primitive_root(p)
    ginv = powermod(g, p - 2, p)
    ain = Vector{Int}(undef, L)        # a_q = x[g^{-q}], 0-based index g^{-q} → 1-based +1
    kout = Vector{Int}(undef, L)       # X[g^r]
    b = Vector{Complex{T}}(undef, L)
    @inbounds for q in 0:(L - 1)
        ain[q + 1] = powermod(ginv, q, p)
        kout[q + 1] = powermod(g, q, p)
        b[q + 1] = cispi(s * T(2 * powermod(g, q, p)) / T(p))
    end
    fwd = plan_pfft(Complex{T}, L; variant = :fast)
    invp = plan_pfft(Complex{T}, L; variant = :fast, inverse = true)
    B = copy(b)
    pfft!(B, fwd)                      # B = FFT(b); the inverse plan supplies the 1/L
    return RaderPlan{T, typeof(fwd), typeof(invp)}(p, inverse, ain, kout, B, fwd, invp, Vector{Complex{T}}(undef, L))
end

plan_length(p::RaderPlan)::Int = p.p
plan_inverse(p::RaderPlan)::Bool = p.inverse

function apply_unnormalized!(p::RaderPlan{T}, x::AbstractVector{Complex{T}}) where {T}
    L = p.p - 1
    a = p.abuf
    x0 = zero(Complex{T})
    @inbounds for i in 1:(p.p)
        x0 += x[i]                     # X[0] = Σ x
    end
    x1 = @inbounds x[1]                # x[0] term, before x is overwritten
    @inbounds for q in 1:L
        a[q] = x[p.ain[q] + 1]
    end
    pfft!(a, p.fwd)                    # A = FFT(a)
    @inbounds for i in 1:L
        a[i] *= p.B[i]                 # A · FFT(b)
    end
    pfft!(a, p.invp)                   # conv = IFFT(...) (normalized by 1/L)
    @inbounds begin
        x[1] = x0
        for r in 1:L
            x[p.kout[r] + 1] = x1 + a[r]
        end
    end
    return x
end
