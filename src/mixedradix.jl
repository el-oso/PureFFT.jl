# Stage 2: mixed-radix Cooley-Tukey.
#
# Splits n = r·m, computes r sub-DFTs of length m over stride-r subsequences, then
# combines with a twiddle multiply followed by an r-point DFT (the "butterfly"). Radix 4
# is preferred over radix 2 because a radix-4 step does the work of two radix-2 steps with
# fewer twiddle multiplies — this is the algorithmic gain Stage 2 isolates over Stage 1.
#
# This recursive form is allocating and correctness-first; it handles ARBITRARY n,
# including primes (which fall through to a direct r-point DFT). It is the oracle the
# in-place SIMD kernels of Stage 3 are checked against, and the substrate for Bluestein.

"""
    factorize(n) -> Vector{Int}

Factor `n` into a sequence of radices, preferring 4, then 2, then ascending odd primes.
A leftover prime > 2 is returned as a single large factor (handled by a direct DFT, or
Bluestein in Stage 3).
"""
function factorize(n::Integer)
    n = Int(n)
    factors = Int[]
    sizehint!(factors, 8sizeof(Int) - leading_zeros(n))   # ≤ log2(n) factors; one alloc
    for r in (4, 2)
        while n % r == 0
            push!(factors, r)
            n ÷= r
        end
    end
    p = 3
    while p * p <= n
        while n % p == 0
            push!(factors, p)
            n ÷= p
        end
        p += 2
    end
    n > 1 && push!(factors, n)
    return factors
end

"""
    mixedradix(x, factors, inverse) -> Vector

Recursive mixed-radix FFT of `x` using the given `factors` (product must equal
`length(x)`). Unnormalized; `inverse` only flips the twiddle sign.
"""
function mixedradix(
        x::AbstractVector{Complex{T}}, factors::AbstractVector{Int}, inverse::Bool
    ) where {T}
    n = length(x)
    if isempty(factors)            # n == 1
        return Complex{T}[x[i] for i in eachindex(x)]
    end
    r = factors[1]
    m = n ÷ r
    rest = @view factors[2:end]
    s = inverse ? 2.0 : -2.0

    # r sub-DFTs of length m over the stride-r subsequences x[j1::r]
    subs = Vector{Vector{Complex{T}}}(undef, r)
    @inbounds for j1 in 1:r
        subs[j1] = mixedradix(@view(x[j1:r:n]), rest, inverse)
    end

    out = Vector{Complex{T}}(undef, n)
    t = Vector{Complex{T}}(undef, r)
    @inbounds for b in 0:(m - 1)
        # twiddle-scaled gather: t[j1] = W_n^{s·j1·b} · subs[j1][b]
        for j1 in 0:(r - 1)
            t[j1 + 1] = subs[j1 + 1][b + 1] * Complex{T}(cispi(s * (j1 * b) / n))
        end
        # r-point DFT of t, scattered to out[a·m + b]
        for a in 0:(r - 1)
            acc = zero(Complex{T})
            for j1 in 0:(r - 1)
                acc += t[j1 + 1] * Complex{T}(cispi(s * (j1 * a) / r))
            end
            out[a * m + b + 1] = acc
        end
    end
    return out
end
