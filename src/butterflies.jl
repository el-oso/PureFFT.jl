# Hardcoded small-N DFT codelets ("butterflies"). These are the leaves the mixed-radix
# planner (Stage 2) composes, and the kernels the SIMD variants (Stage 3) re-express.
# Stage 1 only needs the radix-2 butterfly; the rest land here as the scalar reference
# that later stages must match numerically.

# 2-point DFT: [a+b, a-b]
@inline butterfly2(a::T, b::T) where {T} = (a + b, a - b)

# 4-point DFT. `tw` is the sign of the imaginary twist: -im forward, +im inverse.
@inline function butterfly4(a::T, b::T, c::T, d::T, twi::T) where {T}
    # two radix-2 stages
    t0 = a + c
    t1 = a - c
    t2 = b + d
    t3 = (b - d) * twi   # multiply by ∓i
    return (t0 + t2, t1 + t3, t0 - t2, t1 - t3)
end
