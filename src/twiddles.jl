# Precomputed twiddle-factor tables. Accuracy matters: we always compute the angles
# in Float64 via `cispi` (which uses sinpi/cospi and avoids the range-reduction error
# of a plain `exp(im*theta)`), then narrow to the requested element type. Twiddles are
# computed once per plan, never inside the transform hot loop.

"""
    twiddle_table(Complex{T}, n; inverse=false) -> Vector{Complex{T}}

Table `W[k+1] = exp(s*2πi*k/n)` for `k = 0:(n÷2 - 1)`, with `s = -1` forward,
`s = +1` inverse. Indexing this with a stride yields every twiddle a radix-2
decimation-in-time FFT of length `n` needs.
"""
function twiddle_table(::Type{Complex{T}}, n::Integer; inverse::Bool = false) where {T}
    half = max(n >> 1, 1)
    tbl = Vector{Complex{T}}(undef, half)
    s = inverse ? 2.0 : -2.0
    @inbounds for k in 0:(half - 1)
        tbl[k + 1] = Complex{T}(cispi(s * k / n))
    end
    return tbl
end
