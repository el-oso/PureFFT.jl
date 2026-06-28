# Split the PercolRDim1 path: pack | per-col FFT(m) | recombine, vs FFTW whole-rfft-dim.
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
using SIMD: Vec
FFTW.set_num_threads(1)
function med(f; secs=1.5, maxn=5000)
    f(); ts=Float64[]; el=0.0
    while el<secs && length(ts)<maxn
        s=time_ns(); f(); d=time_ns()-s; push!(ts,d); el+=d/1e9
    end
    median(ts)
end
const SHAPES=[("256x256",(256,256)),("128x128",(128,128)),("64x64x64",(64,64,64)),("512x512",(512,512))]
for (lbl,sz) in SHAPES
    x=randn(Float64,sz...); region=ntuple(identity,length(sz))
    pp=P._pure_plan_rfft_nd(x,region); pc=pp.bd1
    if !(pc isa P.PercolRDim1); println("$lbl: bd1=$(typeof(pc)) not percol"); continue; end
    Y=Array{ComplexF64}(undef,pp.cplxsz...)
    outer=P._prod_after(pp.realsz,pp.d)
    m=pc.m; h=pc.h; L=8; es=sizeof(ComplexF64)
    rplan=pp.rplan; z=rplan.zbuf
    pz=reinterpret(Ptr{Float64},pointer(z))
    pxc=reinterpret(Ptr{ComplexF64},pointer(x)); pYc=reinterpret(Ptr{ComplexF64},pointer(Y))

    t_pack=med(()->GC.@preserve x z begin
        for o in 0:(outer-1); unsafe_copyto!(pointer(z),pxc+o*m*es,m); end
    end)
    t_fft=med(()->GC.@preserve z begin
        for o in 0:(outer-1); unsafe_copyto!(pointer(z),pxc+o*m*es,m); P.apply_unnormalized!(rplan.inner,z); end
    end)
    t_recomb=med(()->GC.@preserve x Y z pc begin
        for o in 0:(outer-1)
            P._recombine_col!(Vec{L,Float64},pz,reinterpret(Ptr{Float64},pYc+o*h*es),m,pc.cf,pc.tw,pc.sgn)
        end
    end)
    t_full=med(()->P._rfft_dim1_percol!(pc,rplan,Y,x,outer))
    # FFTW r2c on dim 1 only (matching the r2c dim)
    n=sz[1]
    pf=FFTW.plan_rfft(copy(x),1;flags=FFTW.MEASURE); Yf=Array{ComplexF64}(undef,(n÷2+1,sz[2:end]...))
    t_fftw=med(()->mul!(Yf,pf,x))
    fftonly = t_fft - t_pack
    @printf("%-9s m=%d outer=%d | full=%.0f fftw_r2c=%.0f par=%.3f || pack=%.0f fft=%.0f recomb=%.0f\n",
        lbl,m,outer,t_full,t_fftw,t_fftw/t_full,t_pack,fftonly,t_recomb)
end
