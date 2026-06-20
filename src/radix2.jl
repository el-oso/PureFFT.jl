# Stage 1: radix-2 Cooley-Tukey, the scalar correctness baseline.
#
# Two forms:
#   * radix2_dit! — in-place, iterative, decimation-in-time, precomputed twiddles.
#     This is the "real" baseline that later stages and SIMD variants are measured against.
#   * radix2_rec  — recursive, allocating, twiddles computed on the fly. The simplest
#     possible reference; handy as an independent correctness oracle in tests.
#
# Neither applies the 1/n inverse normalization; that is the plan's job.

"""
    bitreverse!(x)

In-place bit-reversal permutation of `x` (length must be a power of two).
"""
function bitreverse!(x::AbstractVector)
    n = length(x)
    j = 0
    @inbounds for i in 0:(n - 2)
        if i < j
            x[i + 1], x[j + 1] = x[j + 1], x[i + 1]
        end
        m = n >> 1
        while m >= 1 && j >= m
            j -= m
            m >>= 1
        end
        j += m
    end
    return x
end

"""
    radix2_dit!(x, tw)

In-place iterative radix-2 DIT FFT. `tw` is a twiddle table of length `n÷2`
(see [`twiddle_table`](@ref)); its `inverse` flag selects forward/inverse.
"""
function radix2_dit!(x::AbstractVector{Complex{T}}, tw::AbstractVector{Complex{T}}) where {T}
    n = length(x)
    n <= 1 && return x
    bitreverse!(x)
    len = 2
    @inbounds while len <= n
        half = len >> 1
        stride = n ÷ len
        for base in 0:len:(n - 1)
            tidx = 1
            for j in 0:(half - 1)
                w = tw[tidx]
                u = x[base + j + 1]
                v = x[base + j + half + 1] * w
                x[base + j + 1] = u + v
                x[base + j + half + 1] = u - v
                tidx += stride
            end
        end
        len <<= 1
    end
    return x
end

"""
    radix2_rec(x, inverse) -> Vector

Recursive, allocating radix-2 FFT. Independent reference implementation.
"""
function radix2_rec(x::AbstractVector{Complex{T}}, inverse::Bool) where {T}
    n = length(x)
    n == 1 && return Complex{T}[x[1]]
    even = radix2_rec(@view(x[1:2:end]), inverse)
    odd = radix2_rec(@view(x[2:2:end]), inverse)
    s = inverse ? 2.0 : -2.0
    half = n >> 1
    out = Vector{Complex{T}}(undef, n)
    @inbounds for k in 0:(half - 1)
        w = Complex{T}(cispi(s * k / n))
        t = w * odd[k + 1]
        out[k + 1] = even[k + 1] + t
        out[k + half + 1] = even[k + 1] - t
    end
    return out
end
