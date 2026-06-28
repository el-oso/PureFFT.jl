# Prototype: F64 r2c dim-1 per-column (no transpose) + within-column vectorized recombine.
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
using SIMD: Vec, shufflevector, Val
FFTW.set_num_threads(1)
const AR = P.AvxRadix

# within-column vectorized recombine: zc (m complex, FFT'd, contiguous) -> outc (h=m+1 complex)
# cf_lin[k+1] = tw[k+1]*(0,-0.5). pZ points at zc (T), pO at outc (T). sgn = [1,-1,...].
@inline function recombine_col!(::Type{Vec{L,T}}, pZ::Ptr{T}, pO::Ptr{T}, m::Int,
                                cf::Vector{Complex{T}}, tw::Vector{Complex{T}}, sgn::Vec{L,T}) where {L,T}
    W = L >> 1; half = T(0.5)
    pZc = reinterpret(Ptr{Complex{T}}, pZ); pOc = reinterpret(Ptr{Complex{T}}, pO)
    pCf = reinterpret(Ptr{T}, pointer(cf))
    rev = Val(ntuple(j -> (W-1 - (j-1)>>1)*2 + (isodd(j) ? 0 : 1), Val(L)))  # reverse W complex lanes
    @inbounds begin
        z0 = unsafe_load(pZc, 1)
        unsafe_store!(pOc, Complex{T}(real(z0)+imag(z0), zero(T)), 1)
        unsafe_store!(pOc, Complex{T}(real(z0)-imag(z0), zero(T)), m+1)
        k = 1
        while k + W - 1 <= m - 1
            fwd  = P._ldv(Vec{L,T}, pZ, k)
            revr = P._ldv(Vec{L,T}, pZ, m - k - W + 1)
            conjrev = shufflevector(revr, rev) * sgn
            xe = (fwd + conjrev) * half
            diff = fwd - conjrev
            cfk = P._ldv(Vec{L,T}, pCf, k)          # cf_lin[k+1 .. k+W] (0-based offset k)
            P._stv!(pO, k, xe + AR.avx_mul_complex(diff, cfk))
            k += W
        end
        while k <= m - 1
            zk = unsafe_load(pZc, k+1); zmk = conj(unsafe_load(pZc, m-k+1))
            xe = (zk + zmk)*half
            xo = (zk - zmk)*Complex{T}(zero(T), T(-0.5))
            unsafe_store!(pOc, xe + tw[k+1]*xo, k+1)
            k += 1
        end
    end
    nothing
end

function r2c_percol!(Y, x, rplan, cf, sgn, outer, m)
    L = 8
    h = m + 1
    z = rplan.zbuf
    GC.@preserve x Y z cf begin
        pxc = reinterpret(Ptr{ComplexF64}, pointer(x))
        pYc = reinterpret(Ptr{ComplexF64}, pointer(Y))
        pz  = reinterpret(Ptr{Float64}, pointer(z))
        es = sizeof(ComplexF64)
        @inbounds for o in 0:(outer-1)
            unsafe_copyto!(pointer(z), pxc + o*m*es, m)   # pack: copy column (m complex)
            P.apply_unnormalized!(rplan.inner, z)
            pO = reinterpret(Ptr{Float64}, pYc + o*h*es)
            recombine_col!(Vec{L,Float64}, pz, pO, m, cf, rplan.twiddles, sgn)
        end
    end
    Y
end

# validate + measure on 256x256
function med(f; secs=1.2, maxn=4000)
    f(); ts=Float64[]; el=0.0
    while el<secs && length(ts)<maxn
        s=time_ns(); f(); d=time_ns()-s; push!(ts,d); el+=d/1e9; end
    median(ts)
end

for (lbl,sz) in [("256x256",(256,256)),("512x512",(512,512)),("128x128",(128,128)),("64x64x64",(64,64,64)),("240x240",(240,240)),("96x96x96",(96,96,96))]
    x = randn(Float64, sz...); region = ntuple(identity, length(sz))
    pp = P._pure_plan_rfft_nd(x, region)
    m = pp.n ÷ 2; outer = P._prod_after(pp.realsz, pp.d)
    tw = pp.rplan.twiddles
    cf = [tw[k+1]*Complex{Float64}(0.0,-0.5) for k in 0:m]
    sgn = Vec{8,Float64}(ntuple(j->isodd(j) ? 1.0 : -1.0, Val(8)))
    Yref = Array{ComplexF64}(undef, pp.cplxsz...)
    Ynew = Array{ComplexF64}(undef, pp.cplxsz...)
    # ref via batched path (current)
    P._rfft_dim1_batched!(pp.bd1, Yref, x, outer)
    r2c_percol!(Ynew, x, pp.rplan, cf, sgn, outer, m)
    err = maximum(abs.(Ynew .- Yref)) / maximum(abs.(Yref))
    t_new = med(()->r2c_percol!(Ynew, x, pp.rplan, cf, sgn, outer, m))
    t_old = med(()->P._rfft_dim1_batched!(pp.bd1, Yref, x, outer))
    @printf("%-10s m=%d outer=%d relerr=%.2e | percol=%.0f batched=%.0f speedup=%.2f\n",
        lbl, m, outer, err, t_new, t_old, t_old/t_new)
end
