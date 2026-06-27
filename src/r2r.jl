# Real-to-real transforms (DCT / DST) — the 8 FFTW r2r kinds. FFTW's reodft reduction math
# (same-size real FFT + pre/post twiddle for II/III/IV; 2(N∓1) extension for I), implemented with
# Julia specialization (kind as a type parameter ⇒ concrete/dispatch-free plans).
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
