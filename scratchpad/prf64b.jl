# Drill into the r2c batched dim-1: transpose-pack | batched FFT(m) | recombine | transpose-back.
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
using SIMD: Vec
FFTW.set_num_threads(1)

function med(f; secs=1.2, maxn=3000)
    f(); ts=Float64[]; el=0.0
    while el<secs && length(ts)<maxn
        s=time_ns(); f(); d=time_ns()-s; push!(ts,d); el+=d/1e9
    end
    median(ts)
end

const SHAPES=[("256x256",(256,256)),("512x512",(512,512)),("128x128",(128,128)),("64x64x64",(64,64,64))]

for (lbl,sz) in SHAPES
    x=randn(Float64,sz...); region=ntuple(identity,length(sz))
    pp=P._pure_plan_rfft_nd(x,region); rb=pp.bd1
    isnothing(rb) && (println("$lbl: no bd1"); continue)
    Y=Array{ComplexF64}(undef,pp.cplxsz...)
    outer=P._prod_after(pp.realsz,pp.d)
    m=rb.m; h=rb.h; M=rb.M; L=8
    es=sizeof(ComplexF64)

    # one-chunk timings (mc=M) to estimate per-step; full loop covers outer/M chunks
    nchunks = cld(outer, M)
    pxc=reinterpret(Ptr{ComplexF64},pointer(x)); pY=reinterpret(Ptr{ComplexF64},pointer(Y))
    pBc=pointer(rb.stageB); pYc=pointer(rb.stageY)
    pBt=reinterpret(Ptr{Float64},pBc); pYt=reinterpret(Ptr{Float64},pYc)

    t_pack = med(()->GC.@preserve x rb begin
        t0=0; while t0<outer; mc=min(M,outer-t0); P._transpose_block!(pBc,pxc+t0*m*es,m,mc); t0+=mc; end
    end)
    t_fft = med(()->GC.@preserve rb begin
        t0=0; while t0<outer; mc=min(M,outer-t0); P._batched_apply!(rb.bp,pBc,0,mc,1); t0+=mc; end
    end)
    t_recomb = med(()->GC.@preserve rb begin
        t0=0; while t0<outer; mc=min(M,outer-t0); P._recombine_fwd!(Vec{L,Float64},pBt,pYt,mc,mc,m,rb.cf,rb.sgn,rb.tw); t0+=mc; end
    end)
    t_unpack = med(()->GC.@preserve Y rb begin
        t0=0; while t0<outer; mc=min(M,outer-t0); P._transpose_block!(pY+t0*h*es,pYc,mc,h); t0+=mc; end
    end)
    t_full=med(()->P._rfft_dim1_batched!(rb,Y,x,outer))
    tot=t_pack+t_fft+t_recomb+t_unpack
    @printf("%-10s m=%d M=%d nchunks=%d full=%.0f sum=%.0f | pack=%.0f(%.0f%%) fft=%.0f(%.0f%%) recomb=%.0f(%.0f%%) unpack=%.0f(%.0f%%)\n",
        lbl,m,M,nchunks,t_full,tot,t_pack,100t_pack/tot,t_fft,100t_fft/tot,t_recomb,100t_recomb/tot,t_unpack,100t_unpack/tot)
end
