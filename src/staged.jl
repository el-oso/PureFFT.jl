# Stage 3 substrate: a radix-2 DIT laid out for SIMD.
#
# The trick that makes vectorization pay is a UNIT-STRIDE inner loop. We precompute, per
# stage, a contiguous twiddle vector `W_len^j` for j = 0:half-1, so the inner butterfly
# loop reads x[base+1 .. base+half], x[base+half+1 .. base+len] and the twiddles all
# contiguously. The SIMD variants (Base @simd, SIMD.jl, llvmcall) differ ONLY in how this
# inner loop is expressed; the staging and twiddles are shared, so any speed difference is
# attributable to the kernel, not the algorithm.

"""
    staged_twiddles(Complex{T}, n; inverse=false) -> Vector{Vector{Complex{T}}}

One contiguous twiddle vector per radix-2 stage. `stages[s][j+1] = W_len^j` with
`len = 2^s`, `half = len÷2`, `j = 0:half-1`.
"""
function staged_twiddles(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    nstages = trailing_zeros(Int(n))         # n == 2^nstages; count known up front
    stages = Vector{Vector{Complex{T}}}(undef, nstages)
    s = inverse ? 2.0 : -2.0
    len = 2
    for si in 1:nstages
        half = len >> 1
        v = Vector{Complex{T}}(undef, half)
        @inbounds for j in 0:(half - 1)
            v[j + 1] = Complex{T}(cispi(s * j / len))
        end
        stages[si] = v
        len <<= 1
    end
    return stages
end

"""
    radix2_staged!(x, stages)

Scalar staged radix-2 reference (no SIMD). Correctness oracle for the SIMD kernels.
"""
function radix2_staged!(x::AbstractVector{Complex{T}}, stages) where {T}
    n = length(x)
    n <= 1 && return x
    bitreverse!(x)
    len = 2
    @inbounds for twv in stages
        half = len >> 1
        for base in 0:len:(n - 1)
            for j in 0:(half - 1)
                w = twv[j + 1]
                u = x[base + j + 1]
                v = x[base + j + half + 1] * w
                x[base + j + 1] = u + v
                x[base + j + half + 1] = u - v
            end
        end
        len <<= 1
    end
    return x
end

"""
    radix2_base_simd!(x, stages)

Variant A — pure Base Julia. Identical to [`radix2_staged!`](@ref) but with `@simd` on the
unit-stride inner loop, asking LLVM to autovectorize the contiguous complex butterflies
with no external package. This is the purest test of "LLVM does it".
"""
function radix2_base_simd!(x::AbstractVector{Complex{T}}, stages) where {T}
    n = length(x)
    n <= 1 && return x
    bitreverse!(x)
    len = 2
    @inbounds for twv in stages
        half = len >> 1
        for base in 0:len:(n - 1)
            lo = base
            hi = base + half
            @simd for j in 1:half
                w = twv[j]
                u = x[lo + j]
                v = x[hi + j] * w
                x[lo + j] = u + v
                x[hi + j] = u - v
            end
        end
        len <<= 1
    end
    return x
end
