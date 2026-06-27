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
