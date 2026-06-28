# Recursive (multi-factor) mixed-radix FFT — the parity path for smooth non-power-of-two sizes.
#
# The 2-factor four-step is forced into huge codelets for large n (e.g. n=5760 → 80×72), which spill
# registers; small codelets are far more efficient (R≈8 ≈55 GF/s vs R≈40 ≈36). This decomposes n into
# SEVERAL small factors and runs a batch-all mixed-radix: at each level a size-N1 batched SoA codelet
# (width = M·batch, so always wide/efficient) with the four-step twiddle FUSED into its output store,
# then a transpose. This is the structure (small butterflies + batched columns + transposes);
# it reaches ~0.85–0.91× FFTW on larger smooth sizes where the 2-factor path / Bluestein fall short.
#
# Layout (DIT, digit j = j2 + M·j1 with j1 the high digit): transform b, element j at data[j*batch+b].
# Each level: codelet_tw (size N1, width M·batch) → twiddle W_cur^{k1·j2} (fused) → transpose
# [k1,j2,b]→[j2,k1,b]. Factors are a type parameter so every Val(R) codelet specializes.

"""
    RecursiveMixedRadixPlan{T,FACS} <: AbstractFFTPlan{T}

Multi-factor mixed-radix FFT of length `prod(FACS)` (FACS an `NTuple` of small smooth factors).
Built by `autoplan` for smooth composite non-power-of-two sizes where it beats the 2-factor four-step
(the autotuner times it and the alternatives and keeps the fastest).
"""
struct RecursiveMixedRadixPlan{T, FACS} <: AbstractFFTPlan{T}
    inverse::Bool
    twr::Vector{Vector{T}}      # per-level fused twiddles (length K-1), each [k1*w + j2*batch + b]
    twi::Vector{Vector{T}}
    ar::Vector{T}; ai::Vector{T}     # SoA ping-pong scratch
    br::Vector{T}; bi::Vector{T}
end

# Balanced factorization of n into smooth factors each ≤ maxf (recursively pick a divisor near the
# k-th root). Returns nothing if n can't be split into small smooth factors.
function _recursive_factors(n::Int; maxf::Int = 30)
    n <= maxf && return Int[n]
    k = max(2, ceil(Int, log(n) / log(maxf)))
    target = round(Int, n^(1 / k))
    best = 0; bestdist = typemax(Int)
    for d in 2:maxf
        (n % d == 0 && _max_prime_factor(d) <= 7) || continue
        dist = abs(d - target)
        if dist < bestdist
            bestdist = dist; best = d
        end
    end
    best == 0 && return nothing
    rest = _recursive_factors(n ÷ best; maxf = maxf)
    isnothing(rest) && return nothing
    return pushfirst!(rest, best)
end

function RecursiveMixedRadixPlan(::Type{Complex{T}}, facs::AbstractVector{<:Integer}; inverse::Bool = false) where {T}
    facs = Int.(facs)
    n = prod(facs)
    s = inverse ? 1 : -1
    twr = Vector{T}[]; twi = Vector{T}[]
    cur = n; batch = 1
    for i in 1:(length(facs) - 1)
        N1 = facs[i]; M = cur ÷ N1; w = M * batch
        tr = Vector{T}(undef, n); ti = Vector{T}(undef, n)
        @inbounds for k1 in 0:(N1 - 1), j2 in 0:(M - 1)
            ww = cispi(T(s) * T(2 * mod(k1 * j2, cur)) / T(cur))
            base = k1 * w + j2 * batch
            for b in 1:batch
                tr[base + b] = real(ww); ti[base + b] = imag(ww)
            end
        end
        push!(twr, tr); push!(twi, ti)
        cur = M; batch *= N1
    end
    z() = Vector{T}(undef, n)
    return RecursiveMixedRadixPlan{T, (facs...,)}(inverse, twr, twi, z(), z(), z(), z())
end

plan_length(::RecursiveMixedRadixPlan{T, FACS}) where {T, FACS} = prod(FACS)::Int
plan_inverse(p::RecursiveMixedRadixPlan)::Bool = p.inverse

# block transpose [k1,j2,b] → [j2,k1,b] (batch>1: contiguous batch-chunk copies, vectorized over b)
@inline function _transpose_blocked!(ar, ai, br, bi, N1::Int, M::Int, batch::Int)
    @inbounds for k1 in 0:(N1 - 1), j2 in 0:(M - 1)
        s = (k1 * M + j2) * batch; d = (j2 * N1 + k1) * batch
        @simd ivdep for b in 1:batch
            ar[d + b] = br[s + b]; ai[d + b] = bi[s + b]
        end
    end
    return
end

@generated function apply_unnormalized!(p::RecursiveMixedRadixPlan{T, FACS}, x::AbstractVector{Complex{T}}) where {T, FACS}
    facs = collect(FACS); n = prod(facs); K = length(facs)
    body = Any[]
    push!(body, :(ar = p.ar; ai = p.ai; br = p.br; bi = p.bi))
    push!(body, :(V = p.inverse ? Val(1) : Val(-1)))
    push!(body, quote
        @inbounds @simd ivdep for i in 1:$n
            ar[i] = real(x[i]); ai[i] = imag(x[i])     # split AoS → SoA (A = ar/ai)
        end
    end)
    cur = n; batch = 1
    for i in 1:(K - 1)
        N1 = facs[i]; M = cur ÷ N1; w = M * batch
        push!(body, :(_dft_codelet_soa_batched_tw!(br, bi, ar, ai, p.twr[$i], p.twi[$i], Val($w), Val($N1), V)))  # A→B (+ fused twiddle)
        if batch == 1
            push!(body, :(_transpose_soa!(ar, ai, br, bi, $N1, $M)))      # B→A
        else
            push!(body, :(_transpose_blocked!(ar, ai, br, bi, $N1, $M, $batch)))
        end
        cur = M; batch *= N1
    end
    Nl = facs[K]
    push!(body, :(_dft_codelet_soa_batched!(br, bi, ar, ai, Val($batch), Val($Nl), V)))   # leaf A→B
    push!(body, quote
        @inbounds @simd ivdep for i in 1:$n
            x[i] = Complex{T}(br[i], bi[i])             # merge SoA → AoS (result in B)
        end
        return x
    end)
    return Expr(:block, body...)
end
