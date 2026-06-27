# Real-to-real transforms (DCT / DST) — the 8 FFTW r2r kinds. FFTW's reodft reduction math
# (same-size real FFT + pre/post twiddle for II/III/IV; 2(N∓1) extension for I), implemented with
# Julia specialization (kind as a type parameter ⇒ concrete/dispatch-free plans).
import ErrorTypes
import ErrorTypes: Result, Ok, Err, @unwrap_or

# ---- kind singletons (exact FFTW names) ----
abstract type R2RKind end
struct REDFT00_T <: R2RKind end   # DCT-I
struct REDFT10_T <: R2RKind end   # DCT-II  ("the DCT")
struct REDFT01_T <: R2RKind end   # DCT-III ("the IDCT")
struct REDFT11_T <: R2RKind end   # DCT-IV
struct RODFT00_T <: R2RKind end   # DST-I
struct RODFT10_T <: R2RKind end   # DST-II
struct RODFT01_T <: R2RKind end   # DST-III
struct RODFT11_T <: R2RKind end   # DST-IV
const REDFT00 = REDFT00_T(); const REDFT10 = REDFT10_T(); const REDFT01 = REDFT01_T(); const REDFT11 = REDFT11_T()
const RODFT00 = RODFT00_T(); const RODFT10 = RODFT10_T(); const RODFT01 = RODFT01_T(); const RODFT11 = RODFT11_T()

# ---- error type (Result-first core; throwing shims added in Task 6) ----
@enum R2RErrKind ERR_UNSUPPORTED_KIND ERR_SIZE_TOO_SMALL ERR_BAD_ELTYPE
struct R2RError
    kind::R2RErrKind
    msg::String
end
Base.show(io::IO, e::R2RError) = print(io, "R2RError(", e.kind, "): ", e.msg)

# ---- plan struct ----
# K = kind singleton type; T = Float64/Float32; P = inner plan type.
# Preallocated buffers ⇒ zero-alloc apply. scale = 1 for r2r; ortho factor for dct.
struct R2RPlan{K, T, P}
    n::Int
    inner::P
    pre::Vector{Complex{T}}     # pre-twiddles (kind-specific; may be empty)
    post::Vector{Complex{T}}    # post-twiddles
    rbuf::Vector{T}             # real work buffer
    cbuf::Vector{Complex{T}}    # half-spectrum / complex work buffer
    scale::T
end

# natural inner-FFT size per kind (Phase 1: II/III use size n)
_natural_size(::Union{REDFT10_T, REDFT01_T}, n::Int) = n

# Phase-1 support set. Returns Ok(plan) or Err(R2RError). Per-kind builders arrive in Tasks 3–5;
# this skeleton dispatches and returns Err for any unsupported kind.
function tryplan_r2r(x::AbstractVector{<:Real}, kind::R2RKind)
    T = float(eltype(x))
    n = length(x)
    return _build_r2r(kind, T, n)
end

# fallthrough: any kind without a concrete _build_r2r method is unsupported (Phase 1)
_build_r2r(kind::R2RKind, ::Type{T}, n::Int) where {T} =
    Result{R2RPlan, R2RError}(Err(R2RError(ERR_UNSUPPORTED_KIND, "kind $(kind) not implemented yet")))

# ── DCT-II (REDFT10) — even-N real-FFT route (Makhoul) ───────────────────────
# FFTW REDFT10 (unnormalized): y_k = 2·Σ_j x_j·cos(π(2j+1)k/(2N)), k=0..N-1.
# DCT-II post-twiddle: W_k = exp(-iπk/2N), k = 0..n-1.
_dct_post_tw(::Type{T}, n) where {T} = Complex{T}[cispi(-T(k) / (2n)) for k in 0:(n - 1)]

# Naive cosine-sum reference (Float64 inside; for tests).
_dct2_naive(x) = [2 * sum(x[j + 1] * cospi((2j + 1) * k / (2length(x))) for j in 0:length(x) - 1) for k in 0:length(x) - 1]

function _build_r2r(::REDFT10_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT10 needs n≥1")))
    if iseven(n)
        inner = plan_prfft(T, n)                       # length-n real FFT
        post  = _dct_post_tw(T, n)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan  = R2RPlan{REDFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, rbuf, cbuf, one(T))
        return Result{R2RPlan, R2RError}(Ok(plan))
    else
        return _build_r2r_dct2_odd(T, n)               # Task 4
    end
end

# Apply (even N): reorder x → rbuf (even samples up, odd samples reversed), real FFT → cbuf
# half-spectrum, y_k = 2·Re(post_k·V_k) with Hermitian extension V_k = conj(V_{n-k}) for k > n/2.
function _apply!(p::R2RPlan{REDFT10_T, T}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T}
    n = p.n; m = n ÷ 2; v = p.rbuf; V = p.cbuf; W = p.post
    @inbounds for j in 0:(m - 1)
        v[j + 1] = T(x[2j + 1])      # x[2j]
        v[n - j] = T(x[2j + 2])      # x[2j+1] reversed into the tail
    end
    apply_rfft!(p.inner, v, V)             # V[1..m+1] = half-spectrum
    @inbounds for k in 0:(n - 1)
        Vk = k <= m ? V[k + 1] : conj(V[n - k + 1])
        y[k + 1] = T(2) * real(W[k + 1] * Vk)
    end
    return y
end

# Generic apply entry + tryr2r (per-kind _apply! dispatched on the plan's K).
function tryr2r(x::AbstractVector{<:Real}, kind::R2RKind)
    r = tryplan_r2r(x, kind)
    ErrorTypes.is_error(r) && return Result{Vector, R2RError}(Err(ErrorTypes.unwrap_error(r)))
    p = ErrorTypes.unwrap(r)
    T = eltype(p.rbuf)
    y = Vector{T}(undef, p.n)
    _apply!(p, y, x)
    return Result{Vector, R2RError}(Ok(y))
end
