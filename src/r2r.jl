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

# ---- plan structs ----
# Common supertype so `*` / `mul!` / `inv` / `\` and `tryr2r` work over BOTH the FFT-wrap plan
# (R2RPlan, wins for n≥128) and the small-N straight-line codelet plan (R2RCodeletPlan, below).
# K = kind singleton type; T = Float64/Float32.
abstract type AbstractR2RPlan{K, T} end
_plan_T(::AbstractR2RPlan{K, T}) where {K, T} = T

# K = kind singleton type; T = Float64/Float32; P = inner plan type.
# Preallocated buffers ⇒ zero-alloc apply.
struct R2RPlan{K, T, P} <: AbstractR2RPlan{K, T}
    n::Int
    inner::P
    pre::Vector{Complex{T}}     # pre-twiddles (kind-specific; may be empty)
    post::Vector{Complex{T}}    # post-twiddles
    rbuf::Vector{T}             # real work buffer
    cbuf::Vector{Complex{T}}    # half-spectrum / complex work buffer
end
_plan_n(p::R2RPlan) = p.n

# Small-N straight-line codelet plan: N is a TYPE parameter so the `@generated` body (`_r2r_codelet!`)
# specializes and the hot path is loop-/dispatch-free, zero scratch (everything lives in registers).
# All twiddles are baked into the generated code → the struct carries NO fields. Built by
# `tryplan_r2r` for the slow small-N kinds (DCT/DST II/III/I) when the inner DFT stays smooth.
struct R2RCodeletPlan{K, T, N} <: AbstractR2RPlan{K, T} end
_plan_n(::R2RCodeletPlan{K, T, N}) where {K, T, N} = N::Int

# Guard: _apply! dispatch on P <: RealFFTPlan / RealIFFTPlan vs P <: AbstractFFTPlan requires the
# real-FFT plans NOT to be <: AbstractFFTPlan (keeps the two method bodies disjoint at compile time).
@assert !(RealFFTPlan <: AbstractFFTPlan) && !(RealIFFTPlan <: AbstractFFTPlan) "r2r route dispatch assumes the real-FFT plans are not <: AbstractFFTPlan"

# Phase-1 support set. Returns Ok(plan) or Err(R2RError). Per-kind builders arrive in Tasks 3–5;
# this skeleton dispatches and returns Err for any unsupported kind.
function tryplan_r2r(x::AbstractVector{<:Real}, kind::R2RKind)
    T = float(eltype(x))
    n = length(x)
    if _use_r2r_codelet(kind, n)                       # small-N straight-line codelet route
        return Result{AbstractR2RPlan, R2RError}(Ok(R2RCodeletPlan{typeof(kind), T, n}()))
    end
    return _build_r2r(kind, T, n)
end

# Inner DFT length each kind reduces to: N for II/III, 2(N∓1) for the I extensions. Used both to
# size the codelet's straight-line DFT and to gate routing on its smoothness.
_r2r_inner_size(::Type{REDFT10_T}, n) = n
_r2r_inner_size(::Type{REDFT01_T}, n) = n
_r2r_inner_size(::Type{RODFT10_T}, n) = n
_r2r_inner_size(::Type{RODFT01_T}, n) = n
_r2r_inner_size(::Type{REDFT00_T}, n) = 2 * (n - 1)
_r2r_inner_size(::Type{RODFT00_T}, n) = 2 * (n + 1)
_r2r_inner_size(::Type, n) = 0                          # IV / unknown: never codelet

# Route to the @generated codelet only for the slow small-N kinds (DCT/DST II/III/I — DCT/DST-IV's
# complex route is already competitive small), and only when the unrolled inner DFT stays smooth
# (largest prime factor ≤ 7 → no O(p²) prime-leaf blowup) so compile time + code size stay sane.
#  · Forward kinds (II/DST-II, I/DST-I) use a HALF-SIZE real-packed DFT (efficient) → need an even
#    inner size; cutoff N ≤ 64 (the half-size DFT stays competitive even there).
#  · Inverse kinds (III/DST-III) use a full size-N complex DFT (≈2× the work of the real route) so
#    they only win for N ≤ 32; above that the FFT-wrap route is already at/near parity — keep it.
# Invalid sizes fall through to `_build_r2r` which returns the proper size Err.
_r2r_codelet_fwd(K) = K in (REDFT10_T, RODFT10_T, REDFT00_T, RODFT00_T)
function _use_r2r_codelet(kind::R2RKind, n::Int)
    K = typeof(kind)
    m = _r2r_inner_size(K, n)
    m >= 1 || return false
    (K === REDFT00_T ? n >= 2 : n >= 1) || return false
    _max_prime_factor(m) <= 7 || return false
    if _r2r_codelet_fwd(K)
        return iseven(m) && (m ÷ 2) <= 32         # half-size real pack: even inner size, DFT ≤ 32
    else
        return n <= 32                            # full-complex inverse codelet
    end
end

# fallthrough: any kind without a concrete _build_r2r method is unsupported (Phase 1)
_build_r2r(kind::R2RKind, ::Type{T}, n::Int) where {T} =
    Result{R2RPlan, R2RError}(Err(R2RError(ERR_UNSUPPORTED_KIND, "kind $(kind) not implemented yet")))

# ── DCT-I (REDFT00) — even-extension real-FFT ────────────────────────────────
# FFTW REDFT00 (unnormalized): y_k = x_0 + (−1)^k x_{N−1} + 2·Σ_{j=1}^{N−2} x_j cos(πjk/(N−1)).
# Reduction: build symmetric extension e = [x_0, x_1, …, x_{N−1}, x_{N−2}, …, x_1] of length
# M = 2(N−1) (always even for N≥2), run length-M real FFT, take y_k = Re(Ê_k) for k=0..N−1.
# The half-spectrum has M/2+1 = N entries — exactly the N outputs needed. Self-inverse: 2(N−1)·I.
function _build_r2r(::REDFT00_T, ::Type{T}, n::Int) where {T}
    n >= 2 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "REDFT00 needs n≥2")))
    M = 2 * (n - 1)
    inner = plan_prfft(T, M)
    rbuf  = Vector{T}(undef, M)                # extension buffer length M
    cbuf  = Vector{Complex{T}}(undef, n)        # half-spectrum (M/2+1 = n)
    plan  = R2RPlan{REDFT00_T, T, typeof(inner)}(n, inner, Complex{T}[], Complex{T}[], rbuf, cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply: fill symmetric extension e from x, real FFT → cbuf, y_k = Re(E_k).
function _apply!(p::R2RPlan{REDFT00_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealFFTPlan}
    n = p.n; e = p.rbuf; E = p.cbuf
    @inbounds for j in 1:n
        e[j] = T(x[j])                          # x_0 … x_{N−1}
    end
    @inbounds for j in 1:(n - 2)
        e[n + j] = T(x[n - j])                  # x_{N−2} … x_1 (symmetric tail)
    end
    apply_rfft!(p.inner, e, E)                  # E[1..n] = half-spectrum
    @inbounds for k in 1:n
        y[k] = real(E[k])
    end
    return y
end

# ── DST-I (RODFT00) — odd-extension real-FFT ─────────────────────────────────
# FFTW RODFT00 (unnormalized): y_k = 2·Σ_{j=0}^{N−1} x_j sin(π(j+1)(k+1)/(N+1)), k=0..N-1.
# Reduction: build the odd (antisymmetric) extension
#   o = [0, x_0, x_1, …, x_{N−1}, 0, −x_{N−1}, …, −x_0]  of length M = 2(N+1),
# run length-M real FFT, take y_k = −Im(Ô_{k+1}) for k=0..N−1 (0-indexed; Julia: −imag(O[k+2])).
# The half-spectrum has M/2+1 = N+2 entries; only indices 2..N+1 are used. Self-inverse: 2(N+1)·I.
function _build_r2r(::RODFT00_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "RODFT00 needs n≥1")))
    M = 2 * (n + 1)
    inner = plan_prfft(T, M)
    rbuf  = Vector{T}(undef, M)                  # extension buffer length M
    cbuf  = Vector{Complex{T}}(undef, n + 2)     # half-spectrum (M/2+1 = n+2)
    plan  = R2RPlan{RODFT00_T, T, typeof(inner)}(n, inner, Complex{T}[], Complex{T}[], rbuf, cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply: fill odd extension o from x, real FFT → cbuf, y[k+1] = −imag(O[k+2]) for k=0..N−1.
function _apply!(p::R2RPlan{RODFT00_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealFFTPlan}
    n = p.n; o = p.rbuf; O = p.cbuf
    o[1] = zero(T)                               # leading zero
    @inbounds for j in 1:n
        o[j + 1] = T(x[j])                      # x_0 … x_{N−1}
    end
    o[n + 2] = zero(T)                           # middle zero
    @inbounds for j in 1:n
        o[n + 2 + j] = -T(x[n + 1 - j])        # −x_{N−1} … −x_0 (antisymmetric tail)
    end
    apply_rfft!(p.inner, o, O)                   # O[1..n+2] = half-spectrum
    @inbounds for k in 0:(n - 1)
        y[k + 1] = -imag(O[k + 2])
    end
    return y
end

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

# ── DST-II (RODFT10) — sine sibling of DCT-II, via (−1)ʲ pre-sign + reversed output ──
# FFTW RODFT10 (unnormalized): y_k = 2·Σ_j x_j·sin(π(2j+1)(k+1)/(2N)), k=0..N-1.
# Reduction to DCT-II: sin(π(2j+1)(k+1)/2N) = (−1)ʲ·cos(π(2j+1)(N−1−k)/2N) (use k'=N−1−k;
# sin(π(2j+1)/2)=(−1)ʲ, cos(π(2j+1)/2)=0). So  DST-II_k(x) = DCT-II_{N−1−k}((−1)ʲ x_j):
# run REDFT10's exact machinery (same Makhoul reorder, same post-twiddle _dct_post_tw) with the
# odd-index samples NEGATED (the tail), then REVERSE the output (y'_k → y_{N−1−k}). Both routes
# (even-N real FFT, odd-N complex fallback) mirror REDFT10 with those two changes. No inv yet
# (DST-III / RODFT01 is Task 4).
function _build_r2r(::RODFT10_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "RODFT10 needs n≥1")))
    if iseven(n)
        inner = plan_prfft(T, n)                       # length-n real FFT
        post  = _dct_post_tw(T, n)                     # W_k = e^{−iπk/2N} (shared with DCT-II)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan  = R2RPlan{RODFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, rbuf, cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    else
        return _build_r2r_dst2_odd(T, n)
    end
end

# odd-N complex-FFT fallback (mirrors _build_r2r_dct2_odd): full complex spectrum, ~2× slower.
function _build_r2r_dst2_odd(::Type{T}, n::Int) where {T}
    inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = false)
    post  = _dct_post_tw(T, n)
    cbuf  = Vector{Complex{T}}(undef, n)               # full complex spectrum
    plan  = R2RPlan{RODFT10_T, T, typeof(inner)}(n, inner, Complex{T}[], post, T[], cbuf)
    return Result{R2RPlan, R2RError}(Ok(plan))
end

# Apply (even N): Makhoul reorder with odd samples NEGATED → real FFT → reversed DCT-II output.
function _apply!(p::R2RPlan{RODFT10_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealFFTPlan}
    n = p.n; m = n ÷ 2; v = p.rbuf; V = p.cbuf; W = p.post
    @inbounds for j in 0:(m - 1)
        v[j + 1] = T(x[2j + 1])       # even-index samples up front
        v[n - j] = -T(x[2j + 2])      # odd-index samples NEGATED, reversed into the tail
    end
    apply_rfft!(p.inner, v, V)                          # V[1..m+1] = half-spectrum
    @inbounds for k in 0:(n - 1)
        Vk = k <= m ? V[k + 1] : conj(V[n - k + 1])
        y[n - k] = T(2) * real(W[k + 1] * Vk)           # reversed output: DCT-II_k → DST-II_{N−1−k}
    end
    return y
end

# Apply (odd N): inner is a complex plan (P<:AbstractFFTPlan). Same reorder/negate, full-spectrum FFT.
function _apply!(p::R2RPlan{RODFT10_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; V = p.cbuf; W = p.post
    @inbounds for j in 0:((n - 1) ÷ 2)
        V[j + 1] = Complex{T}(T(x[2j + 1]), zero(T))    # even-index samples ascending into the front
    end
    @inbounds for j in 0:(n ÷ 2 - 1)
        V[n - j] = Complex{T}(-T(x[2j + 2]), zero(T))   # odd-index samples NEGATED, reversed into the tail
    end
    apply_unnormalized!(p.inner, V)
    @inbounds for k in 0:(n - 1)
        y[n - k] = T(2) * real(W[k + 1] * V[k + 1])     # reversed output
    end
    return y
end

# ── DST-III (RODFT01) — structural inverse of DST-II (mirrors REDFT01↔REDFT10) ──
# FFTW RODFT01 (unnormalized): y_k = (−1)^k x_{N−1} + 2·Σ_{j=0}^{N−2} x_j sin(π(j+1)(2k+1)/(2N)).
# Identity (DST-II = R∘DCT-II∘S with R = output reversal, S = input sign (−1)ʲ ⇒ DST-III = S∘DCT-III∘R):
#   RODFT01(x)_k = (−1)^k · REDFT01(reverse(x))_k.
# So run REDFT01's EXACT machinery on the reversed input (z_j = x_{N−1−j}), then negate the odd-index
# outputs. In the V build this just relabels x indices: z_k = x[N−k], z_{N−k} = x[k] (1-based). Buffers /
# inner plans (plan_pirfft even, inverse complex FFT odd) and the pre-twiddle (conj(W_k)) are identical
# to REDFT01. The 1/2N inv-pair with RODFT10 is wired below.
function _build_r2r(::RODFT01_T, ::Type{T}, n::Int) where {T}
    n >= 1 || return Result{R2RPlan, R2RError}(Err(R2RError(ERR_SIZE_TOO_SMALL, "RODFT01 needs n≥1")))
    if iseven(n)
        inner = plan_pirfft(T, n)
        pre   = _dct_post_tw(T, n)                     # W_k; III uses conj(W_k)
        rbuf  = Vector{T}(undef, n)
        cbuf  = Vector{Complex{T}}(undef, n ÷ 2 + 1)
        plan  = R2RPlan{RODFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], rbuf, cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    else
        inner = plan_pfft(Complex{T}, n; variant = :fast, inverse = true)
        pre   = _dct_post_tw(T, n)
        cbuf  = Vector{Complex{T}}(undef, n)           # full complex spectrum
        plan  = R2RPlan{RODFT01_T, T, typeof(inner)}(n, inner, pre, Complex{T}[], T[], cbuf)
        return Result{R2RPlan, R2RError}(Ok(plan))
    end
end

# Apply (even N): build the scaled half-spectrum V from REVERSED x, apply_irfft! → reordered real v,
# inverse-reorder into y, NEGATING the odd-index outputs (the (−1)^k sign).
function _apply!(p::R2RPlan{RODFT01_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: RealIFFTPlan}
    n = p.n; m = n ÷ 2; W = p.pre; V = p.cbuf; v = p.rbuf
    @inbounds V[1] = Complex{T}(T(n) * T(x[n]), zero(T))                # N·z_0 = N·x_{N−1}
    @inbounds for k in 1:m
        V[k + 1] = T(n) * conj(W[k + 1]) * Complex{T}(T(x[n - k]), -T(x[k]))   # z_k − i·z_{N−k}
    end
    apply_irfft!(p.inner, V, v)                                        # v = reordered time domain
    @inbounds for j in 0:(m - 1)
        y[2j + 1] = v[j + 1]                                           # even output (+)
        y[2j + 2] = -v[n - j]                                          # odd output negated (−1)^k
    end
    return y
end

# Apply (odd N): full Hermitian spectrum from REVERSED x, unnormalized inverse FFT, inverse-reorder
# into y with the odd-index outputs negated.
function _apply!(p::R2RPlan{RODFT01_T, T, P}, y::AbstractVector{T}, x::AbstractVector{<:Real}) where {T, P <: AbstractFFTPlan}
    n = p.n; W = p.pre; V = p.cbuf
    @inbounds V[1] = Complex{T}(T(x[n]), zero(T))                      # z_0 = x_{N−1}
    @inbounds for k in 1:(n - 1)
        V[k + 1] = conj(W[k + 1]) * Complex{T}(T(x[n - k]), -T(x[k]))  # z_k − i·z_{N−k}
    end
    apply_unnormalized!(p.inner, V)
    @inbounds for j in 0:((n - 1) ÷ 2)
        y[2j + 1] = real(V[j + 1])                                     # even output (+)
    end
    @inbounds for j in 0:(n ÷ 2 - 1)
        y[2j + 2] = -real(V[n - j])                                    # odd output negated
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

# ── @generated small-N direct codelets (the analogue of src/codelets.jl for r2r) ─────────────
# For a small size N and kind K, emit the FULLY-UNROLLED reduction as one branch-free, loop-free,
# dispatch-free routine: read all inputs into locals, build the kind's reorder into split (re/im)
# symbols, run a straight-line size-(N or 2(N∓1)) DFT via the existing `_gen_dft_soa_mixed!`
# emission (codelets.jl) with COMPILE-TIME twiddle literals, then the kind's pre/post twiddles.
# This kills the per-call + reorder-loop + plan-dispatch overhead that made small N lose to FFTW's
# hand-unrolled codelets. Uniformly uses the unnormalized complex-FFT route (valid for any N), so
# real inputs feed imag=0 literals which LLVM constant-folds away. Mirrors each `_apply!` exactly.

# Straight-line size-M REAL FFT (M even) → half-spectrum X[0..M/2] as (re,im) symbol pairs. Packs the
# M reals into M/2 complex, runs ONE size-(M/2) complex DFT via `_gen_dft_soa_mixed!` (the efficient
# "N/2 DFT"), then the standard split/unpack with baked twiddle literals. Half the arithmetic of a full
# size-M complex DFT — this is what makes the small-N forward codelets beat FFTW. `rin` = M real syms.
function _gen_rfft!(stmts, rin::Vector, M::Int, Tt, ctr, pfx)
    H = M ÷ 2
    zr = Any[rin[2j + 1] for j in 0:(H - 1)]       # z[j] = rin[2j] + i·rin[2j+1]   (1-based rin)
    zi = Any[rin[2j + 2] for j in 0:(H - 1)]
    Zr, Zi = _gen_dft_soa_mixed!(stmts, zr, zi, H, -1, Tt, ctr, pfx)
    ns() = (r = Symbol(pfx, "R", ctr[]); i = Symbol(pfx, "I", ctr[]); ctr[] += 1; (r, i))
    Xr = Vector{Any}(undef, H + 1); Xi = Vector{Any}(undef, H + 1)
    for k in 0:H
        ak = k % H; mk = (H - k) % H                # periodic: Z[H] ≡ Z[0]
        ar, ai = Zr[ak + 1], Zi[ak + 1]
        br = Zr[mk + 1]; bi = :(-$(Zi[mk + 1]))     # conj(Z[(H-k) mod H])
        c = cispi(-2 * k / M) * (-im)               # e^{-2πik/M}·(−i); X = ½(sum + c·dif)
        cr = Tt(real(c)); ci = Tt(imag(c))
        sr, si = ns(); push!(stmts, :($sr = $ar + $br)); push!(stmts, :($si = $ai + $bi))
        dr, di = ns(); push!(stmts, :($dr = $ar - $br)); push!(stmts, :($di = $ai - $bi))
        tr, ti = ns()
        push!(stmts, :($tr = muladd($cr, $dr, -$ci * $di)))
        push!(stmts, :($ti = muladd($cr, $di, $ci * $dr)))
        xr, xi = ns()
        push!(stmts, :($xr = $(Tt(0.5)) * ($sr + $tr)))
        push!(stmts, :($xi = $(Tt(0.5)) * ($si + $ti)))
        Xr[k + 1] = xr; Xi[k + 1] = xi
    end
    return Xr, Xi
end

# build the straight-line body for (K, N) at @generated time. `Tt` = element type (from y).
function _r2r_codelet_body(K, N::Int, Tt)
    stmts = Any[]
    z = :(zero($Tt))
    g = Vector{Any}(undef, N)                       # inputs xr[1..N] → locals g1..gN (read up front)
    for i in 1:N
        s = Symbol("g", i); g[i] = s
        push!(stmts, :(@inbounds $s = $Tt(x[$i])))
    end
    ctr = Ref(0)
    # post-twiddle W_k = e^{-iπk/2N} (II/III); helpers to bake real literals
    Wr(k) = Tt(real(cispi(-k / (2 * N)))); Wi(k) = Tt(imag(cispi(-k / (2 * N))))

    if K === REDFT10_T || K === RODFT10_T            # DCT-II / DST-II: Makhoul reorder → real FFT → post
        neg = K === RODFT10_T; H = N ÷ 2
        rin = Vector{Any}(undef, N)
        for j in 0:((N - 1) ÷ 2)
            rin[j + 1] = g[2j + 1]                   # even samples ascending into the front
        end
        for j in 0:(N ÷ 2 - 1)
            rin[N - j] = neg ? :(-$(g[2j + 2])) : g[2j + 2]   # odd samples (DST: negated) into tail
        end
        Xr, Xi = _gen_rfft!(stmts, rin, N, Tt, ctr, "w")       # half-spectrum X[0..N/2]
        for k in 0:(N - 1)
            vr = k <= H ? Xr[k + 1] : Xr[N - k + 1]            # Hermitian extension V_k = conj(V_{N-k})
            vi = k <= H ? Xi[k + 1] : :(-$(Xi[N - k + 1]))
            c1 = Tt(2) * Wr(k); c2 = Tt(-2) * Wi(k)            # y = 2·Re(W·V) = 2Wr·Vr − 2Wi·Vi
            out = :(muladd($c1, $vr, $c2 * $vi))
            idx = neg ? (N - k) : (k + 1)                      # DST-II reverses the output
            push!(stmts, :(@inbounds y[$idx] = $out))
        end

    elseif K === REDFT01_T || K === RODFT01_T        # DCT-III / DST-III: pre-twiddle → inv DFT → reorder
        rev = K === RODFT01_T                         # DST-III runs III on reversed input, negates odd out
        insr = Vector{Any}(undef, N); insi = Vector{Any}(undef, N)
        # V_0 = x_0 (DCT-III) or x_{N-1} (DST-III); imag 0
        insr[1] = rev ? g[N] : g[1]; insi[1] = z
        for k in 1:(N - 1)
            # V_k = conj(W_k)·(c − i·d):  re = Wr·c − Wi·d,  im = −Wr·d − Wi·c
            c = rev ? g[N - k] : g[k + 1]
            d = rev ? g[k]     : g[N - k + 1]
            rs = Symbol("p", k); is = Symbol("q", k)
            push!(stmts, :($rs = muladd($(Wr(k)), $c, $(Tt(-1) * Wi(k)) * $d)))
            push!(stmts, :($is = muladd($(Tt(-1) * Wr(k)), $d, $(Tt(-1) * Wi(k)) * $c)))
            insr[k + 1] = rs; insi[k + 1] = is
        end
        Vr, _ = _gen_dft_soa_mixed!(stmts, insr, insi, N, +1, Tt, ctr, "w")
        for j in 0:((N - 1) ÷ 2)
            push!(stmts, :(@inbounds y[$(2j + 1)] = $(Vr[j + 1])))            # even outputs (+)
        end
        for j in 0:(N ÷ 2 - 1)
            v = rev ? :(-$(Vr[N - j])) : Vr[N - j]                            # DST-III negates odd outputs
            push!(stmts, :(@inbounds y[$(2j + 2)] = $v))
        end

    elseif K === REDFT00_T                            # DCT-I: even extension → real FFT → Re
        M = 2 * (N - 1)
        er = Vector{Any}(undef, M)
        for i in 1:N; er[i] = g[i]; end                                      # x_0 … x_{N-1}
        for j in 1:(N - 2); er[N + j] = g[N - j]; end                        # symmetric tail x_{N-2}…x_1
        Xr, _ = _gen_rfft!(stmts, er, M, Tt, ctr, "w")                        # X[0..M/2] = X[0..N-1]
        for k in 1:N
            push!(stmts, :(@inbounds y[$k] = $(Xr[k])))                       # y_k = Re(E_k)
        end

    elseif K === RODFT00_T                            # DST-I: odd extension → real FFT → −Im
        M = 2 * (N + 1)
        o = Vector{Any}(undef, M)
        o[1] = z
        for j in 1:N; o[1 + j] = g[j]; end                                   # x_0 … x_{N-1}
        o[N + 2] = z
        for j in 1:N; o[N + 2 + j] = :(-$(g[N + 1 - j])); end                # −x_{N-1} … −x_0
        _, Xi = _gen_rfft!(stmts, o, M, Tt, ctr, "w")                         # X[0..M/2] = X[0..N+1]
        for k in 0:(N - 1)
            push!(stmts, :(@inbounds y[$(k + 1)] = -$(Xi[k + 2])))            # y_k = −Im(O_{k+1})
        end
    else
        error("no r2r codelet for kind $K")
    end
    push!(stmts, :(return nothing))
    return Expr(:block, stmts...)
end

@generated function _r2r_codelet!(::Type{K}, ::Val{N}, y, x) where {K, N}
    return _r2r_codelet_body(K, N, eltype(y))
end

function _apply!(p::R2RCodeletPlan{K, T, N}, y::AbstractVector, x::AbstractVector) where {K, T, N}
    _r2r_codelet!(K, Val(N), y, x)
    return y
end

# inv / \ for the codelet plan (the FFT-wrap plan keeps its per-kind methods below). Inverse kind +
# unnormalized scale by kind; rebuilds via plan_r2r (→ codelet again for small N) wrapped scaled.
_r2r_inv(::Type{REDFT10_T}, n, ::Type{T}) where {T} = (REDFT01, T(1) / (2n))
_r2r_inv(::Type{REDFT01_T}, n, ::Type{T}) where {T} = (REDFT10, T(1) / (2n))
_r2r_inv(::Type{RODFT10_T}, n, ::Type{T}) where {T} = (RODFT01, T(1) / (2n))
_r2r_inv(::Type{RODFT01_T}, n, ::Type{T}) where {T} = (RODFT10, T(1) / (2n))
_r2r_inv(::Type{REDFT00_T}, n, ::Type{T}) where {T} = (REDFT00, T(1) / (2 * (n - 1)))
_r2r_inv(::Type{RODFT00_T}, n, ::Type{T}) where {T} = (RODFT00, T(1) / (2 * (n + 1)))
function Base.inv(p::R2RCodeletPlan{K, T, N}) where {K, T, N}
    ik, sc = _r2r_inv(K, N, T)
    return ScaledR2RPlan(plan_r2r(Vector{T}(undef, N), ik), sc)
end
Base.:\(p::R2RCodeletPlan, x::AbstractVector) = inv(p) * x

# Generic apply entry + tryr2r (per-kind _apply! dispatched on the plan's K).
function tryr2r(x::AbstractVector{<:Real}, kind::R2RKind)
    r = tryplan_r2r(x, kind)
    ErrorTypes.is_error(r) && return Result{Vector, R2RError}(Err(ErrorTypes.unwrap_error(r)))
    p = ErrorTypes.unwrap(r)
    T = _plan_T(p)
    y = Vector{T}(undef, _plan_n(p))
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
Base.:*(p::AbstractR2RPlan{K, T}, x::AbstractVector) where {K, T} = _apply!(p, Vector{T}(undef, _plan_n(p)), x)
LinearAlgebra.mul!(y::AbstractVector, p::AbstractR2RPlan, x::AbstractVector) = _apply!(p, y, x)

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
# RODFT10·RODFT01 = 2N·I (the DST-II↔III pair) ⇒ each is (1/2N)× the other. Wire BOTH directions.
function Base.inv(p::R2RPlan{RODFT10_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), RODFT01)
    return ScaledR2RPlan(ip, T(1) / (2 * p.n))
end
Base.:\(p::R2RPlan{RODFT10_T}, x::AbstractVector) = inv(p) * x
function Base.inv(p::R2RPlan{RODFT01_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), RODFT10)
    return ScaledR2RPlan(ip, T(1) / (2 * p.n))
end
Base.:\(p::R2RPlan{RODFT01_T}, x::AbstractVector) = inv(p) * x
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
# REDFT00 is self-inverse up to 2(N−1): REDFT00·REDFT00 = 2(N−1)·I ⇒ inv(REDFT00) = (1/(2(N−1)))·REDFT00.
function Base.inv(p::R2RPlan{REDFT00_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), REDFT00)
    return ScaledR2RPlan(ip, T(1) / (2 * (p.n - 1)))
end
Base.:\(p::R2RPlan{REDFT00_T}, x::AbstractVector) = inv(p) * x
# RODFT00 is self-inverse up to 2(N+1): RODFT00·RODFT00 = 2(N+1)·I ⇒ inv(RODFT00) = (1/(2(N+1)))·RODFT00.
function Base.inv(p::R2RPlan{RODFT00_T, T}) where {T}
    ip = plan_r2r(Vector{T}(undef, p.n), RODFT00)
    return ScaledR2RPlan(ip, T(1) / (2 * (p.n + 1)))
end
Base.:\(p::R2RPlan{RODFT00_T}, x::AbstractVector) = inv(p) * x
