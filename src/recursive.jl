# Stage 4/6: cache-oblivious recursive Cooley-Tukey (decimation-in-time) with `@generated`
# straight-line codelets as leaves.
#
# Divide-and-conquer keeps each sub-transform cache-resident (radix-2's flaw was streaming the
# whole array every pass). The decimation is encoded in (offset, stride) into the original
# input, so no bit-reversal pass; output is written contiguously to scratch, then copied back.
# Base cases for n ≤ 32 dispatch to the compile-time-generated register-blocked codelets
# (`src/codelets.jl`), which removes recursion overhead and twiddle-table loads near the
# leaves — where most of the tree's nodes live. Direction is the compile-time `Val{S}`.

const RECURSE_BASE = 32   # n ≤ this → single generated codelet

# Type-stable base-case dispatch: each branch passes a LITERAL Val, so no dynamic dispatch.
@inline function _leaf!(out, oo, x, off, str, n, ::Val{S}) where {S}
    @match n begin
        32 => _codelet!(out, oo, x, off, str, Val(32), Val(S))
        16 => _codelet!(out, oo, x, off, str, Val(16), Val(S))
        8 => _codelet!(out, oo, x, off, str, Val(8), Val(S))
        4 => _codelet!(out, oo, x, off, str, Val(4), Val(S))
        2 => _codelet!(out, oo, x, off, str, Val(2), Val(S))
        _ => (@inbounds out[oo + 1] = x[off + 1])   # n == 1
    end
    return
end

# `2^L == n`; `stages[L][k+1] == W_n^k` (built with the matching direction).
function _ditrec!(out, oo, x, off, str, n, L, stages, ::Val{S}) where {S}
    if n <= RECURSE_BASE
        _leaf!(out, oo, x, off, str, n, Val(S))
        return
    end
    n2 = n >> 1
    s2 = str << 1
    _ditrec!(out, oo, x, off, s2, n2, L - 1, stages, Val(S))            # even
    _ditrec!(out, oo + n2, x, off + str, s2, n2, L - 1, stages, Val(S)) # odd
    tw = stages[L]
    @inbounds @simd for k in 1:n2
        e = out[oo + k]
        o = _cmul(out[oo + n2 + k], tw[k])
        out[oo + k] = e + o
        out[oo + n2 + k] = e - o
    end
    return
end

"""
    recursive_fft!(x, scratch, stages, inverse)

Cache-oblivious recursive radix-2 FFT (power-of-two `length(x)`) with generated codelet leaves.
`scratch` matches `length(x)`; `stages` are per-level twiddles ([`staged_twiddles`](@ref)) built
with `inverse`. One runtime branch on `inverse` picks the compile-time direction.
"""
function recursive_fft!(x::AbstractVector{Complex{T}}, scratch, stages, inverse::Bool) where {T}
    n = length(x)
    n <= 1 && return x
    L = trailing_zeros(n)        # n == 2^L
    if inverse
        _ditrec!(scratch, 0, x, 0, 1, n, L, stages, Val(1))
    else
        _ditrec!(scratch, 0, x, 0, 1, n, L, stages, Val(-1))
    end
    copyto!(x, scratch)
    return x
end
