# Stage 7: cache-blocked four-step FFT — the path to FFTW-class throughput.
#
# Split N = N1·N2 and view the data as an N1×N2 matrix. Then:
#   1. batched FFT of size N2 (batch = N1, contiguous)   — SIMD across the batch, no shuffles
#   2. element-wise twiddle multiply (precomputed table)
#   3. cache-blocked transpose N1×N2 → N2×N1
#   4. batched FFT of size N1 (batch = N2, contiguous)
# The batched inner FFTs vectorize ACROSS independent transforms, so each SIMD lane holds a
# different transform — the shuffle-free, fully-vectorized codelet form that a single AoS/SoA
# recursive transform can't reach. A standalone batched kernel hits ~40–57 GFLOP/s here
# (≥ FFTW); wrapped in the four-step with a blocked transpose it lands ~1.4–1.6× of FFTW.
#
# Everything is SoA (split re/im) and works on preallocated buffers → allocation-free.

# Bit-reversal of the FFT-position index, shared across all `B` batched columns.
function _bitrev_rows!(vr, vi, B, R)
    j = 0
    @inbounds for i in 0:(R - 2)
        if i < j
            for b in 1:B
                vr[b + B * i], vr[b + B * j] = vr[b + B * j], vr[b + B * i]
                vi[b + B * i], vi[b + B * j] = vi[b + B * j], vi[b + B * i]
            end
        end
        m = R >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end
    return
end

# `B` simultaneous size-`R` DITs, laid out v[b + B*r] (batch b contiguous, position r stride B).
# `twr/twi` hold W_R^k for k = 0:R/2-1 (direction baked at build time).
function _batched_dit!(vr, vi, B, R, twr, twi)
    _bitrev_rows!(vr, vi, B, R)
    len = 2
    @inbounds while len <= R
        half = len >> 1
        stride = R ÷ len
        for base in 0:len:(R - 1)
            ti = 1
            for j in 0:(half - 1)
                wr = twr[ti]; wi = twi[ti]
                lo = B * (base + j); hi = B * (base + j + half)
                @simd for b in 1:B
                    ar = vr[lo + b]; ai = vi[lo + b]
                    br = vr[hi + b]; bi = vi[hi + b]
                    tr = muladd(br, wr, -bi * wi)
                    tci = muladd(br, wi, bi * wr)
                    vr[lo + b] = ar + tr; vi[lo + b] = ai + tci
                    vr[hi + b] = ar - tr; vi[hi + b] = ai - tci
                end
                ti += stride
            end
        end
        len <<= 1
    end
    return
end

# Cache-blocked transpose: src column-major N1×N2 (idx n1+N1*k2) → dst N2×N1 (idx k2+N2*n1).
# Kept as a pure blocked copy: the step-2 twiddle is a SEPARATE contiguous @simd pass, because
# fusing compute into this scattered-write loop measured slower (the writes don't vectorize).
function _btranspose!(dr, di, sr, si, N1, N2, blk = 32)
    @inbounds for jj in 0:blk:(N2 - 1), ii in 0:blk:(N1 - 1)
        for n1 in ii:(min(ii + blk, N1) - 1), k2 in jj:(min(jj + blk, N2) - 1)
            s = n1 + N1 * k2 + 1
            d = k2 + N2 * n1 + 1
            dr[d] = sr[s]; di[d] = si[s]
        end
    end
    return
end

_halfcos(R, s) = Float64[cospi(s * k / R) for k in 0:((R >> 1) - 1)]
_halfsin(R, s) = Float64[sinpi(s * k / R) for k in 0:((R >> 1) - 1)]

"""
    FourStepPlan{T} <: AbstractFFTPlan{T}

Cache-blocked four-step FFT plan for power-of-two `n = n1·n2`. All twiddles and buffers are
preallocated; the transform is allocation-free. Best for medium/large `n`.
"""
struct FourStepPlan{T} <: AbstractFFTPlan{T}
    n::Int
    inverse::Bool
    n1::Int
    n2::Int
    w1r::Vector{T}; w1i::Vector{T}     # size-n1 batched-FFT twiddles
    w2r::Vector{T}; w2i::Vector{T}     # size-n2 batched-FFT twiddles
    t2r::Vector{T}; t2i::Vector{T}     # step-2 twiddle matrix W_N^{n1 k2}
    xr::Vector{T}; xi::Vector{T}
    yr::Vector{T}; yi::Vector{T}
end

function FourStepPlan(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    ispow2(n) || throw(ArgumentError(":fourstep needs a power-of-two size; got n=$n"))
    n >= 16 || throw(ArgumentError(":fourstep needs n ≥ 16; got n=$n"))
    L = trailing_zeros(Int(n))
    n1 = 1 << (L >> 1)
    n2 = Int(n) ÷ n1
    s = inverse ? 2.0 : -2.0
    t2r = Vector{T}(undef, Int(n)); t2i = Vector{T}(undef, Int(n))
    @inbounds for k2 in 0:(n2 - 1), m in 0:(n1 - 1)
        w = cispi(s * m * k2 / n)
        t2r[m + n1 * k2 + 1] = T(real(w))
        t2i[m + n1 * k2 + 1] = T(imag(w))
    end
    z = () -> Vector{T}(undef, Int(n))
    return FourStepPlan{T}(
        Int(n), inverse, n1, n2,
        T.(_halfcos(n1, s)), T.(_halfsin(n1, s)), T.(_halfcos(n2, s)), T.(_halfsin(n2, s)),
        t2r, t2i, z(), z(), z(), z(),
    )
end

plan_length(p::FourStepPlan)::Int = p.n
plan_inverse(p::FourStepPlan)::Bool = p.inverse

function apply_unnormalized!(p::FourStepPlan{T}, x::AbstractVector) where {T}
    n, n1, n2 = p.n, p.n1, p.n2
    xr, xi, yr, yi = p.xr, p.xi, p.yr, p.yi
    @inbounds @simd for i in 1:n
        xr[i] = real(x[i]); xi[i] = imag(x[i])
    end
    _batched_dit!(xr, xi, n1, n2, p.w2r, p.w2i)           # step 1
    @inbounds @simd for i in 1:n                          # step 2: twiddle (contiguous, vectorized)
        ar = xr[i]; ai = xi[i]; wr = p.t2r[i]; wi = p.t2i[i]
        xr[i] = muladd(ar, wr, -ai * wi)
        xi[i] = muladd(ar, wi, ai * wr)
    end
    _btranspose!(yr, yi, xr, xi, n1, n2)                  # step 3: blocked transpose
    _batched_dit!(yr, yi, n2, n1, p.w1r, p.w1i)           # step 4
    @inbounds @simd for i in 1:n
        x[i] = Complex(yr[i], yi[i])
    end
    return x
end
