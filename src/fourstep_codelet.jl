# Stage 10: four-step executor with batched SoA codelets — the fast non-power-of-two path.
#
# n = n1·n2. Lay the input out (split re/im) as an n1×n2 matrix and:
#   1. size-n1 DFTs down the columns, batched over the n2 columns  (`_dft_codelet_soa_batched!`)
#   2. multiply by the W_n^{i2·k1} twiddle matrix
#   3. transpose n1×n2 → n2×n1
#   4. size-n2 DFTs down the columns, batched over the n1 columns
# Output lands in natural order. Both passes are the shuffle-free batched SoA codelet (the vector
# rank), so this is FFTW's mixed-radix-with-vectorized-butterflies approach — 2–4× over
# Bluestein on smooth sizes. Factors are encoded in the plan type so the codelets specialize.

# Largest codelet size used as a four-step factor (bounds generated-code size) and the SIMD batch
# width floor (the batched codelet needs width ≥ W).
const FOURSTEP_MAX_FACTOR = 128   # covers smooth n up to 128² = 16384
const FOURSTEP_MIN_FACTOR = 8

# Valid split bounds: each factor smooth (largest prime ≤ 7, so its codelet's prime leaf stays
# cheap) and in [FOURSTEP_MIN_FACTOR, FOURSTEP_MAX_FACTOR]. `autoplan` enumerates and times them
# (see `_foursplit_candidates` / `_best_foursplit_plan`); the balanced split is not always fastest.

"""
    FourStepCodeletPlan{T,N1,N2} <: AbstractFFTPlan{T}

Four-step FFT of length `N1·N2` using batched SoA codelets for both passes (sizes `N1`, `N2`
encoded in the type so the codelets specialize and the hot path is dispatch-free). Built by
`autoplan` for smooth composite non-power-of-two sizes; far faster than Bluestein.
"""
struct FourStepCodeletPlan{T, N1, N2} <: AbstractFFTPlan{T}
    inverse::Bool
    twr::Vector{T}              # W_n^{i2·k1} twiddle matrix, layout [k1*N2 + i2]
    twi::Vector{T}
    sr::Vector{T}              # scratch: split input / pass-2 output
    si::Vector{T}
    ar::Vector{T}              # scratch: pass-1 output (+ twiddled)
    ai::Vector{T}
    br::Vector{T}              # scratch: transposed
    bi::Vector{T}
end

function FourStepCodeletPlan(::Type{Complex{T}}, n1::Integer, n2::Integer; inverse::Bool = false) where {T}
    n1 = Int(n1); n2 = Int(n2); n = n1 * n2
    s = inverse ? 1 : -1
    twr = Vector{T}(undef, n); twi = Vector{T}(undef, n)
    @inbounds for k1 in 0:(n1 - 1), i2 in 0:(n2 - 1)
        w = cispi(T(s) * T(2 * mod(i2 * k1, n)) / T(n))
        twr[k1 * n2 + i2 + 1] = real(w)
        twi[k1 * n2 + i2 + 1] = imag(w)
    end
    z() = Vector{T}(undef, n)
    return FourStepCodeletPlan{T, n1, n2}(inverse, twr, twi, z(), z(), z(), z(), z(), z())
end

plan_length(::FourStepCodeletPlan{T, N1, N2}) where {T, N1, N2} = (N1 * N2)::Int
plan_inverse(p::FourStepCodeletPlan)::Bool = p.inverse

function apply_unnormalized!(p::FourStepCodeletPlan{T, N1, N2}, x::AbstractVector{Complex{T}}) where {T, N1, N2}
    n = N1 * N2
    sr = p.sr; si = p.si; ar = p.ar; ai = p.ai; br = p.br; bi = p.bi
    @inbounds @simd ivdep for i in 1:n          # split AoS → SoA
        sr[i] = real(x[i]); si[i] = imag(x[i])
    end
    V = p.inverse ? Val(1) : Val(-1)
    _dft_codelet_soa_batched!(ar, ai, sr, si, N2, Val(N1), V)            # pass 1: size-N1, width N2
    twr = p.twr; twi = p.twi
    @inbounds @simd ivdep for idx in 1:n        # twiddle by W_n^{i2·k1}
        a = ar[idx]; b = ai[idx]; wr = twr[idx]; wi = twi[idx]
        ar[idx] = a * wr - b * wi
        ai[idx] = a * wi + b * wr
    end
    _transpose_soa!(br, bi, ar, ai, N1, N2)   # SIMD register-tiled transpose [k1*N2+i2]→[i2*N1+k1]
    _dft_codelet_soa_batched!(sr, si, br, bi, N1, Val(N2), V)            # pass 2: size-N2, width N1
    @inbounds @simd ivdep for i in 1:n          # merge SoA → AoS (natural order)
        x[i] = Complex{T}(sr[i], si[i])
    end
    return x
end
