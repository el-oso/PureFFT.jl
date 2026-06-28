# Real N-dimensional FFT (rfft / irfft / brfft), built on the proven 1-D real plans (rfft.jl)
# + the complex N-D engine (ndim.jl). Separable, FFTW/AbstractFFTs convention:
#
#   rfft(x, region):  r2c along FIRST(region) (that dim n → n÷2+1, real→complex) THEN c2c along the
#                     remaining region dims (on the now-complex half-spectrum). first(region) is NOT
#                     sorted — order matters (matches AbstractFFTs.rfft_output_size / FFTW).
#   irfft/brfft:      reverse — c2c⁻¹ along the rest, THEN c2r along first(region) (needs original len d).
#
# Correctness-first cut: the r2c/c2r along the chosen dim uses a reshape-to-(inner,len,outer) +
# per-column strided view of the 1-D real plan (apply_rfft!/apply_irfft!). Bit-exact, not the
# dispatch-free hot path the complex engine has. The c2c-on-the-rest reuses the complex NDPlan directly.

# Own plan (NOT <: NDPlan — the apply is r2c-then-c2c, a different shape). IS <: AbstractFFTs.Plan so
# rfft/brfft/irfft/plan_irfft all derive from the AbstractFFTs generics (ScaledPlan does the irfft
# normalization). T = input eltype (real Tr forward, Complex{Tr} inverse); Tr = real float type.
struct RealNDPlan{T, Tr, D, RP, CP, N, BR} <: AbstractFFTs.Plan{T}
    d::Int                       # r2c/c2r dim = first(region)
    n::Int                       # real length on dim d (even)
    rplan::RP                    # RealFFTPlan (forward) or RealIFFTPlan (inverse)
    cplan::CP                    # NDPlan over the remaining dims, or `nothing` (single-dim region)
    dims::NTuple{D, Int}         # full transformed region: (d, rest...)
    realsz::NTuple{N, Int}       # real-array shape
    cplxsz::NTuple{N, Int}       # half-spectrum shape (dim d → n÷2+1)
    inverse::Bool
    scale::Tr                    # output scale (brfft: n_d; forward: 1)
    bd1::BR                      # BatchedRDim1 (fast batched r2c on dim 1) or `nothing` (per-column path)
end

@inline function _prod_before(sz, d)
    p = 1; @inbounds for i in 1:(d - 1); p *= sz[i]; end; p
end
@inline function _prod_after(sz, d)
    p = 1; @inbounds for i in (d + 1):length(sz); p *= sz[i]; end; p
end

# Split a region into the r2c dim (literal first, unsorted) and the c2c rest. Accepts Int / tuple /
# range / Colon. Validates 1 ≤ d ≤ N (rest is validated by the complex engine's _canon_region).
function _split_region(region, N::Int)
    if region isa Colon
        d = 1; rest = ntuple(i -> i + 1, N - 1)
    elseif region isa Integer
        d = Int(region); rest = ()
    else
        r = Int.(collect(region))
        isempty(r) && throw(ArgumentError("empty region"))
        d = first(r)
        rest = Tuple(filter(!=(d), r))
    end
    1 <= d <= N || throw(ArgumentError("region $region: r2c dim $d out of bounds for a $N-d array"))
    return d, rest
end

# ── Plan constructors (force the PureFFT path even when FFTW.jl is loaded) ─────
function _pure_plan_rfft_nd(x::AbstractArray{<:Real, N}, region) where {N}
    Tr = float(eltype(x))
    realsz = size(x)
    d, rest = _split_region(region, N)
    n = realsz[d]
    rplan = plan_prfft(Tr, n)                                   # throws ArgumentError if n is odd
    cplxsz = ntuple(i -> i == d ? n ÷ 2 + 1 : realsz[i], N)
    cplan = isempty(rest) ? nothing :
        _pure_plan_fft_nd(Array{Complex{Tr}}(undef, cplxsz), rest; inverse = false)
    dims = (d, rest...)
    outer = _prod_after(realsz, d)
    bd1 = !_use_batched_rdim1(Tr, d, n ÷ 2, outer) ? nothing :
        _use_percol_rdim1(Tr, n ÷ 2) ? _build_percol_rdim1(Tr, n) : _build_batched_rdim1(Tr, n, outer)
    return RealNDPlan{Tr, Tr, length(dims), typeof(rplan), typeof(cplan), N, typeof(bd1)}(
        d, n, rplan, cplan, dims, realsz, cplxsz, false, one(Tr), bd1)
end

function _pure_plan_brfft_nd(X::AbstractArray{<:Complex, N}, n::Integer, region) where {N}
    Tr = real(float(eltype(X)))
    cplxsz = size(X)
    d, rest = _split_region(region, N)
    cplxsz[d] == n ÷ 2 + 1 ||
        throw(ArgumentError("brfft: size(X, $d)=$(cplxsz[d]) ≠ n÷2+1=$(n ÷ 2 + 1) for real length n=$n"))
    rplan = plan_pirfft(Tr, n)                                  # throws ArgumentError if n is odd
    realsz = ntuple(i -> i == d ? Int(n) : cplxsz[i], N)
    cplan = isempty(rest) ? nothing :
        _pure_plan_fft_nd(Array{Complex{Tr}}(undef, cplxsz), rest; inverse = true)
    dims = (d, rest...)
    outer = _prod_after(realsz, d)
    bd1 = _use_batched_rdim1(Tr, d, Int(n) ÷ 2, outer) ? _build_batched_ridim1(Tr, Int(n), outer) : nothing
    return RealNDPlan{Complex{Tr}, Tr, length(dims), typeof(rplan), typeof(cplan), N, typeof(bd1)}(
        d, Int(n), rplan, cplan, dims, realsz, cplxsz, true, Tr(n), bd1)
end

# ── Cores ─────────────────────────────────────────────────────────────────────
# Forward: r2c along dim d (per strided column), then c2c on the rest (in place on Y).
function _rfft_core!(p::RealNDPlan{T, Tr}, Y::AbstractArray, x::AbstractArray) where {T, Tr}
    d = p.d; n = p.n; rsz = p.realsz; h = n ÷ 2 + 1
    inner = _prod_before(rsz, d); outer = _prod_after(rsz, d)
    xf = eltype(x) === Tr ? x : Tr.(x)
    if p.bd1 isa PercolRDim1 && xf isa Array && Y isa Array   # fast per-column dim-1 path (F64 pow2)
        _rfft_dim1_percol!(p.bd1, p.rplan, Y, xf, outer)
    elseif !isnothing(p.bd1) && xf isa Array && Y isa Array   # fast batched dim-1 path (d==1, contiguous)
        _rfft_dim1_batched!(p.bd1, Y, xf, outer)
    else
        xr = reshape(xf, inner, n, outer)
        Yr = reshape(Y, inner, h, outer)
        @inbounds for o in 1:outer, i in 1:inner
            apply_rfft!(p.rplan, view(xr, i, :, o), view(Yr, i, :, o))
        end
    end
    isnothing(p.cplan) || apply_unnormalized!(p.cplan, Y)
    return Y
end

# Inverse (UNNORMALIZED brfft): c2c⁻¹ on the rest (in place on the owned X), then c2r along dim d.
# apply_irfft! already normalizes dim d; the bfft on the rest is unnormalized (factor ∏rest); the
# stored scale = n_d makes the result the brfft (= irfft · ∏region). ScaledPlan→irfft divides it back.
function _brfft_core!(p::RealNDPlan{T, Tr}, y::AbstractArray, X::AbstractArray) where {T, Tr}
    isnothing(p.cplan) || apply_unnormalized!(p.cplan, X)
    d = p.d; n = p.n; rsz = p.realsz; h = n ÷ 2 + 1
    inner = _prod_before(rsz, d); outer = _prod_after(rsz, d)
    if !isnothing(p.bd1) && X isa Array && y isa Array      # fast batched dim-1 c2r path
        _brfft_dim1_batched!(p.bd1, y, X, outer)
    else
        Xr = reshape(X, inner, h, outer)
        yr = reshape(y, inner, n, outer)
        @inbounds for o in 1:outer, i in 1:inner
            apply_irfft!(p.rplan, view(Xr, i, :, o), view(yr, i, :, o))
        end
    end
    s = p.scale
    s == one(Tr) || (y .*= s)
    return y
end

# ── Batched dim-1 r2c (the optimized r2c path) ────────────────────────────────
# The per-column path above runs ONE length-n real FFT at a time (apply_rfft!): its inner length-m
# complex FFT is single-transform (underfills the SIMD register) and its pack/recombine are scalar.
# FFTW instead batches the real transforms ACROSS the trailing columns to fill the SIMD width — the
# per-column path measures 0.2–0.7× FFTW (bench/run_compare_rndim.jl). We close most of that gap the
# same way the complex engine's BatchedDim1 does: for the contiguous r2c dim (d==1), process a chunk of
# `M` columns at once —
#   (1) transpose-pack the (m × M) packed-complex block → (M × m) so the M columns are the contiguous
#       (SIMD-inner) axis (packing pairs into complex is FREE: a real column of length n == m complex);
#   (2) run ONE batched length-m complex FFT over all M columns (reuses batched_fft8!/batched_fft_mr! —
#       the same bit-exact kernels the complex N-D dims use), inner=M;
#   (3) a VECTORIZED half-complex recombine across the M columns (W complex per Vec, contiguous loads);
#   (4) transpose the (M × h) half-spectrum back into Y (h × M block).
# Reuses _transpose_block!, the batched complex kernels, and AvxRadix.avx_mul_complex; the only new SIMD
# is the recombine, which is the scalar apply_rfft! step-4 algebra applied W columns at a time. Zero-alloc
# (pointer-based under GC.@preserve, plan-owned staging). Only d==1, outer≥W, m=n÷2 pow2-or-smooth (so a
# BatchPlan exists) and m≥8; everything else keeps the per-column path.
struct BatchedRDim1{T, L, BP}
    bp::BP                       # BatchPlan8 (pow2 m) or BatchPlanMR (smooth m), forward, size m
    m::Int                       # n÷2
    h::Int                       # m+1 (half-spectrum length on dim d)
    M::Int                       # columns per chunk (multiple of W=L÷2, ≤ outer)
    stageB::Vector{Complex{T}}   # m × M: transpose-packed columns, FFT'd in place
    stageY::Vector{Complex{T}}   # h × M: recombined half-spectrum (before transpose-back)
    cf::Vector{Vec{L, T}}        # cf[k+1] = broadcast(W_n^k · (0,-0.5)) for k=0..m (k=0 unused)
    sgn::Vec{L, T}               # conjugation mask [1,-1,1,-1,…] (negates the imaginary lanes)
    tw::Vector{Complex{T}}       # W_n^k, k=0..m (scalar twiddles for the c-tail)
end

@inline function _use_batched_rdim1(::Type{Tr}, d::Int, m::Int, outer::Int) where {Tr}
    d == 1 && m >= 8 && outer >= (_batch_lanes(Tr) >> 1) && (ispow2(m) || _is_smooth_2a3(m))
end

function _build_batched_rdim1(::Type{Tr}, n::Int, outer::Int) where {Tr}
    m = n ÷ 2; h = m + 1
    L = _batch_lanes(Tr); W = L >> 1
    bp = ispow2(m) ? BatchPlan8(Tr, m; forward = true) : BatchPlanMR(Tr, m; forward = true)
    M = clamp((8192 ÷ m) & ~(W - 1), W, outer)
    tw = _rfft_twiddles(Tr, n)                       # W_n^k, k=0..m
    cf = Vector{Vec{L, Tr}}(undef, m + 1)
    @inbounds for k in 0:m
        c = tw[k + 1] * Complex{Tr}(zero(Tr), Tr(-0.5))   # W_n^k · (0,-0.5): folds xo's 1/(2i) + twiddle
        cf[k + 1] = _bcast_c(Vec{L, Tr}, real(c), imag(c))
    end
    sgn = Vec{L, Tr}(ntuple(k -> iseven(k) ? -one(Tr) : one(Tr), Val(L)))  # ntuple is 1-based: imag lanes → -1
    return BatchedRDim1{Tr, L, typeof(bp)}(bp, m, h, M, Vector{Complex{Tr}}(undef, m * M),
                                           Vector{Complex{Tr}}(undef, h * M), cf, sgn, tw)
end

# ── Per-column dim-1 r2c (the optimized F64 path) ─────────────────────────────
# The batched-transpose path above (BatchedRDim1) pays TWO full matrix transposes (pack + unpack) and
# runs the *batched* complex FFT(m). For F64 those transposes are a large relative cost because the FFT is
# only half-length (m = n/2) — measured 0.50–0.64× FFTW (the c2c-rest already BEATS FFTW; the whole F64
# rfft deficit is here). The complex N-D engine learned the same lesson and routes F64 dim-1 to a
# per-column (NO transpose) 1-D FFT — the radix-8 codelet already fills the F64 register for a single
# transform, so per-column FFT(m) runs at ~0.88–1.2× FFTW vs the transpose path's ~0.3–0.46×. So for
# F64 pow2 m we mirror that: per column, (1) copy-pack the real column (== m complex, free reinterpret)
# into the rplan's scratch, (2) run ONE per-column complex FFT(m) reusing the tuned 1-D `rplan.inner`
# (NO transpose), (3) a within-column VECTORIZED half-complex recombine (reverse-shuffle pairs bin k with
# conj(bin m-k), W complex/Vec) → Y. Bit-exact with the batched path (≤1.4e-15). F32 and smooth-m keep
# the batched path (F32 pow2 clears the gate there; smooth m=120 measured slower per-column).
struct PercolRDim1{Tr, L}
    m::Int                       # n÷2
    h::Int                       # m+1
    cf::Vector{Complex{Tr}}      # cf[k+1] = W_n^k · (0,-0.5): folds xo's 1/(2i) + twiddle (contiguous, vload'd)
    tw::Vector{Complex{Tr}}      # W_n^k, k=0..m (scalar c-tail)
    sgn::Vec{L, Tr}              # conjugation mask [1,-1,1,-1,…]
end

@inline _use_percol_rdim1(::Type{Tr}, m::Int) where {Tr} = Tr === Float64 && ispow2(m)

function _build_percol_rdim1(::Type{Tr}, n::Int) where {Tr}
    m = n ÷ 2; h = m + 1; L = _batch_lanes(Tr)
    tw = _rfft_twiddles(Tr, n)
    cf = Vector{Complex{Tr}}(undef, m + 1)
    @inbounds for k in 0:m
        cf[k + 1] = tw[k + 1] * Complex{Tr}(zero(Tr), Tr(-0.5))
    end
    sgn = Vec{L, Tr}(ntuple(k -> isodd(k) ? one(Tr) : -one(Tr), Val(L)))  # imag lanes (even, 1-based) → -1
    return PercolRDim1{Tr, L}(m, h, cf, tw, sgn)
end

# Within-column vectorized half-complex recombine: zc (m complex, FFT'd, contiguous) → outc (h complex).
# Reproduces apply_rfft! step-3/4 W complex at a time WITHIN one column: pairs bin k with conj(bin m-k)
# via a reverse-shuffle (no transpose). Boundary k=0,m scalar; k=1..m-1 vectorized + scalar c-tail.
@inline function _recombine_col!(::Type{Vec{L, T}}, pZ::Ptr{T}, pO::Ptr{T}, m::Int,
                                 cf::Vector{Complex{T}}, tw::Vector{Complex{T}}, sgn::Vec{L, T}) where {L, T}
    W = L >> 1; half = T(0.5)
    pZc = reinterpret(Ptr{Complex{T}}, pZ); pOc = reinterpret(Ptr{Complex{T}}, pO)
    pCf = reinterpret(Ptr{T}, pointer(cf))
    rev = Val(ntuple(j -> (W - 1 - (j - 1) >> 1) * 2 + (isodd(j) ? 0 : 1), Val(L)))  # reverse W complex lanes
    @inbounds begin
        z0 = unsafe_load(pZc, 1)                                  # boundary k=0, k=m (real)
        unsafe_store!(pOc, Complex{T}(real(z0) + imag(z0), zero(T)), 1)
        unsafe_store!(pOc, Complex{T}(real(z0) - imag(z0), zero(T)), m + 1)
        k = 1
        while k + W - 1 <= m - 1
            fwd = _ldv(Vec{L, T}, pZ, k)
            cjm = shufflevector(_ldv(Vec{L, T}, pZ, m - k - W + 1), rev) * sgn  # conj(z[m-k..m-k-W+1])
            xe = (fwd + cjm) * half
            _stv!(pO, k, xe + AvxRadix.avx_mul_complex(fwd - cjm, _ldv(Vec{L, T}, pCf, k)))
            k += W
        end
        while k <= m - 1                                          # scalar c-tail
            zk = unsafe_load(pZc, k + 1); zmk = conj(unsafe_load(pZc, m - k + 1))
            xe = (zk + zmk) * half
            xo = (zk - zmk) * Complex{T}(zero(T), T(-0.5))
            unsafe_store!(pOc, xe + tw[k + 1] * xo, k + 1)
            k += 1
        end
    end
    return nothing
end

# Per-column r2c on dim 1: copy-pack column → per-column complex FFT(m) (no transpose) → recombine → Y.
function _rfft_dim1_percol!(pc::PercolRDim1{T, L}, rplan, Y::AbstractArray, x::AbstractArray, outer::Int) where {T, L}
    m = pc.m; h = pc.h; z = rplan.zbuf
    GC.@preserve x Y z pc begin
        pxc = reinterpret(Ptr{Complex{T}}, pointer(x))           # x (n×outer real) viewed as (m×outer) complex
        pYc = reinterpret(Ptr{Complex{T}}, pointer(Y))           # Y (h×outer) complex
        pz = reinterpret(Ptr{T}, pointer(z))
        es = sizeof(Complex{T})
        @inbounds for o in 0:(outer - 1)
            unsafe_copyto!(pointer(z), pxc + o * m * es, m)       # pack: copy real column (= m complex, free)
            apply_unnormalized!(rplan.inner, z)                  # per-column complex FFT(m), in place
            _recombine_col!(Vec{L, T}, pz, reinterpret(Ptr{T}, pYc + o * h * es), m, pc.cf, pc.tw, pc.sgn)
        end
    end
    return Y
end

# Vectorized half-complex recombine of a transpose-packed, FFT'd block B → half-spectrum Yb. Both are
# laid out (S × …) with the column index c the CONTIGUOUS axis (stride S = current chunk width): B[c+S·k]
# is FFT bin k of column c; Yb[c+S·k] is X[k] of column c. Reproduces apply_rfft! step-3/4 W columns at a
# time. Boundary bins k=0,m are real (scalar over c); k=1..m-1 vectorized (+ scalar c-tail for c%W).
@inline function _recombine_fwd!(::Type{Vec{L, T}}, pB::Ptr{T}, pY::Ptr{T}, S::Int, mc::Int, m::Int,
                                 cf::Vector{Vec{L, T}}, sgn::Vec{L, T}, tw::Vector{Complex{T}}) where {L, T}
    W = L >> 1; nv = mc ÷ W; half = T(0.5)
    pBc = reinterpret(Ptr{Complex{T}}, pB); pYc = reinterpret(Ptr{Complex{T}}, pY)
    @inbounds begin
        for k in 1:(m - 1)
            cfk = cf[k + 1]; offk = S * k; offmk = S * (m - k)
            g = 0
            while g < nv
                c = g * W
                Bk = _ldv(Vec{L, T}, pB, offk + c)
                cjm = _ldv(Vec{L, T}, pB, offmk + c) * sgn       # conj(B[m-k])
                xe = (Bk + cjm) * half
                _stv!(pY, offk + c, xe + AvxRadix.avx_mul_complex(Bk - cjm, cfk))
                g += 1
            end
            twk = tw[k + 1]
            c = nv * W
            while c < mc
                bk = unsafe_load(pBc, offk + c + 1); bmk = conj(unsafe_load(pBc, offmk + c + 1))
                xe = (bk + bmk) * half
                xo = (bk - bmk) * Complex{T}(zero(T), T(-0.5))
                unsafe_store!(pYc, xe + twk * xo, offk + c + 1)
                c += 1
            end
        end
        offm = S * m
        for c in 0:(mc - 1)                                       # boundary k=0, k=m (real)
            b0 = unsafe_load(pBc, c + 1); r = real(b0); i = imag(b0)
            unsafe_store!(pYc, Complex{T}(r + i, zero(T)), c + 1)
            unsafe_store!(pYc, Complex{T}(r - i, zero(T)), offm + c + 1)
        end
    end
    return nothing
end

# Batched r2c on dim 1: chunked transpose-pack → batched length-m FFT → recombine → transpose-back into Y.
function _rfft_dim1_batched!(rb::BatchedRDim1{T, L}, Y::AbstractArray, x::AbstractArray, outer::Int) where {T, L}
    m = rb.m; h = rb.h; M = rb.M
    GC.@preserve x Y rb begin
        pxc = reinterpret(Ptr{Complex{T}}, pointer(x))           # x (n×outer real) viewed as (m×outer) complex
        pY = reinterpret(Ptr{Complex{T}}, pointer(Y))            # Y (h×outer) complex
        pBc = pointer(rb.stageB); pYc = pointer(rb.stageY)
        pBt = reinterpret(Ptr{T}, pBc); pYt = reinterpret(Ptr{T}, pYc)
        es = sizeof(Complex{T})                                   # Julia Ptr arithmetic is BYTE-wise
        t0 = 0
        @inbounds while t0 < outer
            mc = min(M, outer - t0)
            _transpose_block!(pBc, pxc + t0 * m * es, m, mc)      # (m×mc) → (mc×m): columns now contiguous
            _batched_apply!(rb.bp, pBc, 0, mc, 1)                 # batched length-m complex FFT, inner=mc
            _recombine_fwd!(Vec{L, T}, pBt, pYt, mc, mc, m, rb.cf, rb.sgn, rb.tw)
            _transpose_block!(pY + t0 * h * es, pYc, mc, h)       # (mc×h) → (h×mc): scatter back into Y
            t0 += mc
        end
    end
    return Y
end

# ── Batched dim-1 c2r (the optimized inverse path) ────────────────────────────
# Mirror of the forward path for brfft/irfft: transpose-pack the half-spectrum columns, batched
# inverse-recombine (the apply_irfft! step reconstruction, W columns at a time), ONE batched length-m
# *inverse* complex FFT (bp built forward=false ⇒ a genuine unnormalized IFFT — no conj-FFT-conj dance),
# transpose-back into the real output. The 1/m normalization (apply_irfft!'s `invm`) is folded into the
# recombine constants (IFFT is linear ⇒ scaling Z scales the result), so the output equals the per-column
# apply_irfft! exactly; the downstream brfft `scale` (= n_d) is applied by _brfft_core! as before.
struct BatchedRIDim1{T, L, BP}
    bp::BP                       # BatchPlan8/MR of size m, INVERSE (forward=false)
    m::Int; h::Int; M::Int
    stageX::Vector{Complex{T}}   # h × M: transpose-packed half-spectrum columns
    stageZ::Vector{Complex{T}}   # m × M: inverse-recombined Z, IFFT'd in place
    cf::Vector{Vec{L, T}}        # cf[k+1] = conj(W_n^k)·(0,0.5)·(1/m), k=0..m (k=0 unused)
    sgn::Vec{L, T}               # conjugation mask
    tw::Vector{Complex{T}}       # W_n^k, k=0..m (scalar twiddles for the c-tail)
    invm::T                      # 1/m (folds apply_irfft!'s normalization)
end

function _build_batched_ridim1(::Type{Tr}, n::Int, outer::Int) where {Tr}
    m = n ÷ 2; h = m + 1
    L = _batch_lanes(Tr); W = L >> 1
    bp = ispow2(m) ? BatchPlan8(Tr, m; forward = false) : BatchPlanMR(Tr, m; forward = false)
    M = clamp((8192 ÷ m) & ~(W - 1), W, outer)
    tw = _rfft_twiddles(Tr, n)
    invm = inv(Tr(m))
    cf = Vector{Vec{L, Tr}}(undef, m + 1)
    @inbounds for k in 0:m
        c = conj(tw[k + 1]) * Complex{Tr}(zero(Tr), Tr(0.5)) * invm  # conj(W_n^k)·0.5i·(1/m)
        cf[k + 1] = _bcast_c(Vec{L, Tr}, real(c), imag(c))
    end
    sgn = Vec{L, Tr}(ntuple(k -> iseven(k) ? -one(Tr) : one(Tr), Val(L)))
    return BatchedRIDim1{Tr, L, typeof(bp)}(bp, m, h, M, Vector{Complex{Tr}}(undef, h * M),
                                            Vector{Complex{Tr}}(undef, m * M), cf, sgn, tw, invm)
end

# Vectorized inverse half-complex recombine: half-spectrum block Xb (S × h, c contiguous) → Z (S × m),
# already scaled by 1/m. Reproduces apply_irfft!'s Z reconstruction W columns at a time; boundary bin
# k=0 packs the (k=0,k=m) real pair into Z[0] (scalar over c), k=1..m-1 vectorized (+ scalar c-tail).
@inline function _recombine_inv!(::Type{Vec{L, T}}, pX::Ptr{T}, pZ::Ptr{T}, S::Int, mc::Int, m::Int,
                                 cf::Vector{Vec{L, T}}, sgn::Vec{L, T}, tw::Vector{Complex{T}}, invm::T) where {L, T}
    W = L >> 1; nv = mc ÷ W; halfm = T(0.5) * invm
    pXc = reinterpret(Ptr{Complex{T}}, pX); pZc = reinterpret(Ptr{Complex{T}}, pZ)
    @inbounds begin
        offm = S * m
        for c in 0:(mc - 1)                                      # boundary: z[0] from X[0],X[m] (real)
            x0 = real(unsafe_load(pXc, c + 1)); xm = real(unsafe_load(pXc, offm + c + 1))
            unsafe_store!(pZc, Complex{T}((x0 + xm) * halfm, (x0 - xm) * halfm), c + 1)
        end
        for k in 1:(m - 1)
            cfk = cf[k + 1]; offk = S * k; offmk = S * (m - k)
            g = 0
            while g < nv
                c = g * W
                Xk = _ldv(Vec{L, T}, pX, offk + c)
                cjm = _ldv(Vec{L, T}, pX, offmk + c) * sgn        # conj(X[m-k])
                sh = (Xk + cjm) * halfm
                _stv!(pZ, offk + c, sh + AvxRadix.avx_mul_complex(Xk - cjm, cfk))
                g += 1
            end
            wcj = conj(tw[k + 1]); c = nv * W
            while c < mc
                xk = unsafe_load(pXc, offk + c + 1); xmk = conj(unsafe_load(pXc, offmk + c + 1))
                sh = (xk + xmk) * halfm
                iwd = (xk - xmk) * wcj
                unsafe_store!(pZc, sh + Complex{T}(-imag(iwd) * halfm, real(iwd) * halfm), offk + c + 1)
                c += 1
            end
        end
    end
    return nothing
end

function _brfft_dim1_batched!(rb::BatchedRIDim1{T, L}, y::AbstractArray, X::AbstractArray, outer::Int) where {T, L}
    m = rb.m; h = rb.h; M = rb.M
    GC.@preserve y X rb begin
        pXc = reinterpret(Ptr{Complex{T}}, pointer(X))           # X (h×outer) complex half-spectrum
        pyc = reinterpret(Ptr{Complex{T}}, pointer(y))           # y (n×outer real) viewed as (m×outer)
        pXs = pointer(rb.stageX); pZc = pointer(rb.stageZ)
        pXt = reinterpret(Ptr{T}, pXs); pZt = reinterpret(Ptr{T}, pZc)
        es = sizeof(Complex{T})
        t0 = 0
        @inbounds while t0 < outer
            mc = min(M, outer - t0)
            _transpose_block!(pXs, pXc + t0 * h * es, h, mc)      # (h×mc) → (mc×h): columns contiguous
            _recombine_inv!(Vec{L, T}, pXt, pZt, mc, mc, m, rb.cf, rb.sgn, rb.tw, rb.invm)
            _batched_apply!(rb.bp, pZc, 0, mc, 1)                 # batched length-m INVERSE FFT
            _transpose_block!(pyc + t0 * m * es, pZc, mc, m)      # (mc×m) → (m×mc): real output block
            t0 += mc
        end
    end
    return y
end

# ── Apply surface ─────────────────────────────────────────────────────────────
function Base.:*(p::RealNDPlan{T, Tr}, x::AbstractArray) where {T, Tr}
    if p.inverse
        Xc = Array{Complex{Tr}}(undef, p.cplxsz); copyto!(Xc, x)   # owned copy (c2c is in place)
        y = Array{Tr}(undef, p.realsz)
        return _brfft_core!(p, y, Xc)
    else
        Y = Array{Complex{Tr}}(undef, p.cplxsz)
        return _rfft_core!(p, Y, x)
    end
end

function LinearAlgebra.mul!(y::AbstractArray, p::RealNDPlan{T, Tr}, x::AbstractArray) where {T, Tr}
    out = p.inverse ? p.realsz : p.cplxsz
    in_ = p.inverse ? p.cplxsz : p.realsz
    size(y) == out && size(x) == in_ ||
        throw(DimensionMismatch("RealNDPlan expects in $in_ → out $out; got $(size(x)) → $(size(y))"))
    if p.inverse
        Xc = Array{Complex{Tr}}(undef, p.cplxsz); copyto!(Xc, x)
        _brfft_core!(p, y, Xc)
    else
        _rfft_core!(p, y, x)
    end
    return y
end

Base.size(p::RealNDPlan) = p.inverse ? p.cplxsz : p.realsz
AbstractFFTs.fftdims(p::RealNDPlan) = p.dims

# ── AbstractFFTs drop-in. plan_rfft + plan_brfft are enough: rfft/brfft/irfft/plan_irfft all derive
# from the AbstractFFTs generics (definitions.jl). NOTE FFTW.jl's StridedArray methods are more
# specific and win when FFTW is loaded — use _pure_plan_*_nd to force PureFFT (as the tests do).
AbstractFFTs.plan_rfft(x::AbstractArray{<:Real}, region; kws...) = _pure_plan_rfft_nd(x, region)
AbstractFFTs.plan_brfft(x::AbstractArray{<:Complex}, d::Integer, region; kws...) = _pure_plan_brfft_nd(x, d, region)

# ── Prefixed convenience (matches pfft(::AbstractArray, dims)); thin wrappers over the plan path.
prfft(x::AbstractArray{<:Real}, region) = _pure_plan_rfft_nd(x, region) * x
function pirfft(X::AbstractArray{<:Complex}, d::Integer, region)
    p = _pure_plan_brfft_nd(X, d, region)
    y = p * X                                                   # unnormalized brfft
    y .*= AbstractFFTs.normalization(real(float(eltype(X))), p.realsz, p.dims)
    return y
end
