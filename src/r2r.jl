# Real-to-real transforms (DCT / DST) — the 8 FFTW r2r kinds. FFTW's reodft reduction math
# (same-size real FFT + pre/post twiddle for II/III/IV; 2(N∓1) extension for I), implemented with
# Julia specialization (kind as a type parameter ⇒ concrete/dispatch-free plans).
import ErrorTypes
import ErrorTypes: Result, Ok, Err, @unwrap_or, unwrap_error
import LinearAlgebra

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
# Preallocated buffers ⇒ zero-alloc apply.
struct R2RPlan{K, T, P}
    n::Int
    inner::P
    pre::Vector{Complex{T}}     # pre-twiddles (kind-specific; may be empty)
    post::Vector{Complex{T}}    # post-twiddles
    rbuf::Vector{T}             # real work buffer
    cbuf::Vector{Complex{T}}    # half-spectrum / complex work buffer
end

# Guard: _apply! dispatch on P <: RealFFTPlan / RealIFFTPlan vs P <: AbstractFFTPlan requires the
# real-FFT plans NOT to be <: AbstractFFTPlan (keeps the two method bodies disjoint at compile time).
@assert !(RealFFTPlan <: AbstractFFTPlan) && !(RealIFFTPlan <: AbstractFFTPlan) "r2r route dispatch assumes the real-FFT plans are not <: AbstractFFTPlan"

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


function _build_r2r(::REDFT10_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT10 needs n≥1")))
    if iseven(n)
        inner = plan_prfft(T, n)                       # length-n real FFT
        post  = _dct_post_tw(T, n)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan  = R2RPlan{REDFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, rbuf, cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    else
        return _build_r2r_dct2_odd(T, n)               # Task 4
    end
end

# ── DCT-II (REDFT10) — odd-N complex-FFT fallback ────────────────────────────
# Same Makhoul reorder as even N (even samples ascending, odd samples reversed into the tail), but the
# inner transform is a length-n COMPLEX FFT and V is the FULL spectrum (no Hermitian shortcut). Documented
# ~2× slower than the even-N real-FFT route — correctness only, below the parity gate. Route is selected by
# DISPATCH on the inner plan type P (complex plan <: AbstractFFTPlan; the even-N RealFFTPlan is not), so each
# _apply! body stays monomorphic / @test_opt-clean.
function _build_r2r_dct2_odd(::Type{T}, n::Int) where {T}
    inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = false)
    post  = _dct_post_tw(T, n)
    cbuf  = Vector{Complex{T}}(undef, n)               # full complex spectrum
    plan  = R2RPlan{REDFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, T[], cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply (odd N): inner is a complex plan (P<:AbstractFFTPlan). Reorder x → full complex buffer, length-n
# complex FFT, y_k = 2·Re(post_k·V_k) over the full spectrum.
function _apply!(p::R2RPlan{REDFT10_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; V = p.cbuf; W = p.post
    @inbounds for j in 0:((n - 1) ÷ 2)
        V[j + 1] = Complex{T}(T(x[2j + 1]), zero(T))   # even samples ascending into the front
    end
    @inbounds for j in 0:(n ÷ 2 - 1)
        V[n - j] = Complex{T}(T(x[2j + 2]), zero(T))   # odd samples reversed into the tail
    end
    apply_unnormalized!(p.inner, V)
    @inbounds for k in 0:(n - 1)
        y[k + 1] = T(2) * real(W[k + 1] * V[k + 1])
    end
    return y
end

# Apply (even N): reorder x → rbuf (even samples up, odd samples reversed), real FFT → cbuf
# half-spectrum, y_k = 2·Re(post_k·V_k) with Hermitian extension V_k = conj(V_{n-k}) for k > n/2.
function _apply!(p::R2RPlan{REDFT10_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealFFTPlan}
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

# ── DCT-III (REDFT01) — structural inverse of DCT-II ─────────────────────────
# FFTW REDFT01 (unnormalized): y_k = x_0 + 2·Σ_{j=1}^{N-1} x_j·cos(πj(2k+1)/(2N)), k=0..N-1.
# Inverse of REDFT10: build a (Hermitian) half/full spectrum V from x using conj(W_k) (W_k =
# exp(-iπk/2N), reused via _dct_post_tw), inverse-FFT, then UNDO the Makhoul even/odd reorder.
# Derivation (V = DFT of the reordered DCT-II input v): with A_k = W_k·V_k, REDFT10 gives
#   x_k = Re(A_k)=2·Re(A_k)/2  and  x_{N-k} = -2·Im(A_k)  ⇒  V_k = conj(W_k)·(x_k - i·x_{N-k})/2,
# V_0 = x_0/2. Scale: even-N apply_irfft! normalizes by 1/m (=2/N) → fold the unnormalized 2N
# back in by ×N here (2N·(…/2)=N·…); odd-N unnormalized inverse FFT needs only ×2 (2·(…/2)).
function _build_r2r(::REDFT01_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT01 needs n≥1")))
    if iseven(n)
        inner = plan_pirfft(T, n)
        pre   = _dct_post_tw(T, n)                     # W_k; III uses conj(W_k)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan  = R2RPlan{REDFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], rbuf, cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    else
        inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = true)
        pre   = _dct_post_tw(T, n)
        cbuf  = Vector{Complex{T}}(undef, n)           # full complex spectrum
        plan  = R2RPlan{REDFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], T[], cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    end
end

# Apply (even N): inner is an inverse real-FFT plan (P<:RealIFFTPlan). Build the scaled half-
# spectrum V[0..m] from x, apply_irfft! → reordered real v, then inverse-reorder into y.
function _apply!(p::R2RPlan{REDFT01_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealIFFTPlan}
    n = p.n; m = n ÷ 2; W = p.pre; V = p.cbuf; v = p.rbuf
    @inbounds V[1] = Complex{T}(T(n) * T(x[1]), zero(T))     # N·x_0  (= 2N·x_0/2)
    @inbounds for k in 1:m
        V[k + 1] = T(n) * conj(W[k + 1]) * Complex{T}(T(x[k + 1]), -T(x[n - k + 1]))
    end
    apply_irfft!(p.inner, V, v)                              # v = reordered time domain
    @inbounds for j in 0:(m - 1)
        y[2j + 1] = v[j + 1]                                 # even output samples up front
        y[2j + 2] = v[n - j]                                 # odd output samples from the tail
    end
    return y
end

# Apply (odd N): inner is an inverse complex plan (P<:AbstractFFTPlan). Build the full scaled
# spectrum V[0..N-1] (Hermitian), unnormalized inverse FFT, inverse-reorder into y.
function _apply!(p::R2RPlan{REDFT01_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; W = p.pre; V = p.cbuf
    @inbounds V[1] = Complex{T}(T(x[1]), zero(T))            # x_0  (= 2·x_0/2)
    @inbounds for k in 1:(n - 1)
        V[k + 1] = conj(W[k + 1]) * Complex{T}(T(x[k + 1]), -T(x[n - k + 1]))
    end
    apply_unnormalized!(p.inner, V)
    @inbounds for j in 0:((n - 1) ÷ 2)
        y[2j + 1] = real(V[j + 1])
    end
    @inbounds for j in 0:(n ÷ 2 - 1)
        y[2j + 2] = real(V[n - j])
    end
    return y
end

# ── DCT-IV (REDFT11) — size-N complex-FFT route (Makhoul-IV) ─────────────────
# FFTW REDFT11 (unnormalized): y_k = 2·Σ_j x_j·cos(π(2j+1)(2k+1)/(4N)), k=0..N-1.
# Reduction (any N, even or odd): use the DCT-II even/odd reorder WITH A SIGN FLIP on the
# reflected odd samples — v_p = x_{2m} at p=m, v_p = −x_{2m+1} at p=N−1−m — which folds both
# input groups into the single form Σ_p v_p·cos(π(4p+1)(2k+1)/(4N)). Expanding (4p+1)(2k+1)
# gives the 8pk term = a clean size-N kernel e^{−2πipk/N}, with separable pre/post twiddles:
#   pre_p = e^{−iπp/N},  post_k = e^{−iπ(2k+1)/(4N)},  y_k = 2·Re(post_k · FFT(v_p·pre_p)_k).
# (The naive single-FFT pre/post twiddle leaves an 8pk vs 4pk residual and is NOT DCT-IV — the
# reorder+sign is what halves the cross term.) Self-inverse: REDFT11·REDFT11 = 2N·I.
function _build_r2r(::REDFT11_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT11 needs n≥1")))
    inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = false)
    pre   = Complex{T}[cispi(-T(p) / T(n))         for p in 0:(n - 1)]   # e^{−iπ p /N}
    post  = Complex{T}[cispi(-T(2k + 1) / T(4n))   for k in 0:(n - 1)]   # e^{−iπ(2k+1)/4N}
    cbuf  = Vector{Complex{T}}(undef, n)
    plan  = R2RPlan{REDFT11_T, T, typeof(inner)}(n, inner, pre, post, T[], cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply: inner is a complex plan (P<:AbstractFFTPlan). Reorder+sign x → pre-twiddled complex
# buffer, length-n complex FFT, y_k = 2·Re(post_k·C_k).
function _apply!(p::R2RPlan{REDFT11_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; c = p.cbuf
    @inbounds for m in 0:((n + 1) ÷ 2 - 1)            # even samples ascending into the front
        c[m + 1] = p.pre[m + 1] * T(x[2m + 1])
    end
    @inbounds for m in 0:(n ÷ 2 - 1)                  # odd samples NEGATED, reversed into the tail
        c[n - m] = p.pre[n - m] * (-T(x[2m + 2]))
    end
    apply_unnormalized!(p.inner, c)                   # size-N FFT
    @inbounds for k in 0:(n - 1)
        y[k + 1] = T(2) * real(p.post[k + 1] * c[k + 1])
    end
    return y
end

# ── DST-IV (RODFT11) — size-N complex-FFT route, sine sibling of DCT-IV ───────
# FFTW RODFT11 (unnormalized): y_k = 2·Σ_j x_j·sin(π(2j+1)(2k+1)/(4N)), k=0..N-1.
# Same machinery as REDFT11 (identical pre/post twiddles, size-N complex FFT). The post-twiddled
# spectrum is S_k = post_k·FFT(v·pre)_k = Σ_p v_p·e^{−iπ(4p+1)(2k+1)/4N} for any reordered v. With
# the DCT-IV reorder but the EVEN (front) samples ALSO negated — v_m = −x_{2m} at p=m, v_{N−1−m} =
# −x_{2m+1} at p=N−1−m — the imaginary part folds both groups into +Σ_j x_j sin(π(2j+1)(2k+1)/4N):
#   y_k = 2·Im(post_k · FFT(v·pre)_k).  (Re would give the DCT-IV cosine; Im is the sine variant.)
# No index reversal needed — the front sign-flip is what makes Im the all-+ sine sum. Self-inverse:
# RODFT11·RODFT11 = 2N·I (like REDFT11).
function _build_r2r(::RODFT11_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "RODFT11 needs n≥1")))
    inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = false)
    pre   = Complex{T}[cispi(-T(p) / T(n))         for p in 0:(n - 1)]   # e^{−iπ p /N}
    post  = Complex{T}[cispi(-T(2k + 1) / T(4n))   for k in 0:(n - 1)]   # e^{−iπ(2k+1)/4N}
    cbuf  = Vector{Complex{T}}(undef, n)
    plan  = R2RPlan{RODFT11_T, T, typeof(inner)}(n, inner, pre, post, T[], cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply: reorder+negate x → pre-twiddled complex buffer, length-n complex FFT, y_k = 2·Im(post_k·C_k).
function _apply!(p::R2RPlan{RODFT11_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; c = p.cbuf
    @inbounds for m in 0:((n + 1) ÷ 2 - 1)            # even samples NEGATED into the front
        c[m + 1] = p.pre[m + 1] * (-T(x[2m + 1]))
    end
    @inbounds for m in 0:(n ÷ 2 - 1)                  # odd samples negated, reversed into the tail
        c[n - m] = p.pre[n - m] * (-T(x[2m + 2]))
    end
    apply_unnormalized!(p.inner, c)                   # size-N FFT
    @inbounds for k in 0:(n - 1)
        y[k + 1] = T(2) * imag(p.post[k + 1] * c[k + 1])
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

# ── User-facing throwing layer (FFTW drop-in) ────────────────────────────────
# ErrorTypes `@unwrap_or expr exec` runs `exec` (a plain expression, NOT a lambda) on Err; we
# pull the error out with `unwrap_error(r)` to build the ArgumentError message (FFTW-style).
function plan_r2r(x::AbstractVector{<:Real}, kind::R2RKind)
    r = tryplan_r2r(x, kind)
    return @unwrap_or r throw(ArgumentError(string(unwrap_error(r))))
end
function r2r(x::AbstractVector{<:Real}, kind::R2RKind)
    r = tryr2r(x, kind)
    return @unwrap_or r throw(ArgumentError(string(unwrap_error(r))))
end
r2r!(x::AbstractVector{<:Real}, kind::R2RKind) = copyto!(x, r2r(x, kind))

# plan application: p*x (fresh output) and mul!(y, p, x) (preallocated)
Base.:*(p::R2RPlan{K, T}, x::AbstractVector) where {K, T} = _apply!(p, Vector{T}(undef, p.n), x)
LinearAlgebra.mul!(y::AbstractVector, p::R2RPlan, x::AbstractVector) = _apply!(p, y, x)

# ── Orthonormal DCT-II / DCT-III (scipy norm="ortho" / FFTW.jl `dct`/`idct`) ──
# Built by scaling the UNNORMALIZED r2r. FFTW REDFT10 gives y_k = 2·Σ x_j cos(…); the ortho
# DCT-II is f_k·Σ x_j cos(…) with f_0=√(1/N), f_k=√(2/N) ⇒ scale = f_k/2: s0=√(1/4N), s=√(1/2N).
# idct is the inverse (transpose) of the orthogonal dct: idct = (1/2N)·R01·D⁻¹ since R01·R10=2N·I.
_dct_ortho_scales(::Type{T}, n) where {T} = (sqrt(T(1) / (4n)), sqrt(T(1) / (2n)))

function dct(x::AbstractVector{<:Real})
    T = float(eltype(x)); n = length(x)
    y = r2r(x, REDFT10)
    s0, s = _dct_ortho_scales(T, n)
    @inbounds y[1] *= s0
    @inbounds for k in 2:n; y[k] *= s; end
    return y
end
function idct(x::AbstractVector{<:Real})
    T = float(eltype(x)); n = length(x)
    s0, s = _dct_ortho_scales(T, n)
    x2 = Vector{T}(undef, n)
    @inbounds x2[1] = T(x[1]) / s0
    @inbounds for k in 2:n; x2[k] = T(x[k]) / s; end
    y = r2r(x2, REDFT01)
    inv2n = T(1) / (2n)
    @inbounds for k in 1:n; y[k] *= inv2n; end
    return y
end
dct!(x::AbstractVector{<:Real})  = copyto!(x, dct(x))
idct!(x::AbstractVector{<:Real}) = copyto!(x, idct(x))
plan_dct(x::AbstractVector{<:Real})  = plan_r2r(x, REDFT10)   # plan only; dct() applies the ortho scale
plan_idct(x::AbstractVector{<:Real}) = plan_r2r(x, REDFT01)

# ── inv / \ : unnormalized inverse of a REDFT10 plan (REDFT01 with the 1/2N scale) ───────────
# REDFT01·REDFT10 = 2N·I, so inv(REDFT10) = (1/2N)·REDFT01. Wrapped in a tiny scaled-plan so `*`
# and `\` compose cleanly.
struct ScaledR2RPlan{P, T}
    plan::P
    scale::T
end
Base.:*(sp::ScaledR2RPlan, x::AbstractVector) = (y = sp.plan * x; y .*= sp.scale; y)
function Base.inv(p::R2RPlan{REDFT10_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), REDFT01)
    return ScaledR2RPlan(ip, T(1) / (2 * p.n))
end
Base.:\(p::R2RPlan{REDFT10_T}, x::AbstractVector) = inv(p) * x
# REDFT11 is self-inverse up to 2N: REDFT11·REDFT11 = 2N·I ⇒ inv(REDFT11) = (1/2N)·REDFT11.
function Base.inv(p::R2RPlan{REDFT11_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), REDFT11)
    return ScaledR2RPlan(ip, T(1) / (2 * p.n))
end
Base.:\(p::R2RPlan{REDFT11_T}, x::AbstractVector) = inv(p) * x
# RODFT11 is self-inverse up to 2N: RODFT11·RODFT11 = 2N·I ⇒ inv(RODFT11) = (1/2N)·RODFT11.
function Base.inv(p::R2RPlan{RODFT11_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), RODFT11)
    return ScaledR2RPlan(ip, T(1) / (2 * p.n))
end
Base.:\(p::R2RPlan{RODFT11_T}, x::AbstractVector) = inv(p) * x
