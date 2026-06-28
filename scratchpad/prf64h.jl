# quick bit-exact spot check of the new recombine vs a reference rfft, + isolated recombine timing.
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
    Y=Array{ComplexF64}(undef,pp.cplxsz...)
    outer=P._prod_after(pp.realsz,pp.d)
    # correctness vs FFTW full rfft
    Yref=FFTW.rfft(x,region)
    P._rfft_dim1_percol!(pc,pp.rplan,Y,x,outer)  # only r2c dim; apply c2c rest for full compare
    isnothing(pp.cplan) || P.apply_unnormalized!(pp.cplan,Y)
    err = maximum(abs.(Y .- Yref))/maximum(abs.(Yref))
    t_full=med(()->P._rfft_dim1_percol!(pc,pp.rplan,Y,x,outer))
    n=sz[1]; pf=FFTW.plan_rfft(copy(x),1;flags=FFTW.MEASURE)
    Yf=Array{ComplexF64}(undef,(n÷2+1,sz[2:end]...)); t_fftw=med(()->mul!(Yf,pf,x))
    @printf("%-9s relerr=%.2e | percol r2c=%.0f fftw=%.0f par=%.3f\n",lbl,err,t_full,t_fftw,t_fftw/t_full)
end
