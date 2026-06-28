# Compare PercolRDim1 vs BatchedRDim1 for the SAME F64 shapes, r2c-dim isolated, vs FFTW.
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
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
    x=randn(Float64,sz...)
    n=sz[1]; outer=prod(sz[2:end])
    Y=Array{ComplexF64}(undef,(n÷2+1,sz[2:end]...))
    # percol
    pp=P._pure_plan_rfft_nd(x,ntuple(identity,length(sz)))
    pc=pp.bd1
    t_pc = pc isa P.PercolRDim1 ? med(()->P._rfft_dim1_percol!(pc,pp.rplan,Y,x,outer)) : NaN
    # batched (force-build)
    rb = P._build_batched_rdim1(Float64,n,outer)
    t_bt = med(()->P._rfft_dim1_batched!(rb,Y,x,outer))
    # fftw r2c dim1
    pf=FFTW.plan_rfft(copy(x),1;flags=FFTW.MEASURE)
    t_fftw=med(()->mul!(Y,pf,x))
    @printf("%-9s m=%d outer=%d | percol=%.0f(%.3f) batched=%.0f(%.3f) fftw=%.0f\n",
        lbl,n÷2,outer,t_pc,t_fftw/t_pc,t_bt,t_fftw/t_bt,t_fftw)
end
