# Compare each PureFFT piece vs its FFTW equivalent, to find which piece is below parity.
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
    m=rb.m; outer=P._prod_after(pp.realsz,pp.d)
    Y=Array{ComplexF64}(undef,pp.cplxsz...)

    # --- piece A: PureFFT batched FFT(m) over `outer` cols (incl pack+unpack transposes, fair) ---
    t_r2c = med(()->P._rfft_dim1_batched!(rb,Y,x,outer))
    # FFTW: a length-n REAL rfft of the same (n x outer) batched along dim 2  == r2c on dim 1
    xr = reshape(copy(x), sz[1], outer)
    pf_r2c = FFTW.plan_rfft(xr, 1; flags=FFTW.MEASURE)
    Yr = Array{ComplexF64}(undef, sz[1]÷2+1, outer)
    t_fftw_r2c = med(()->mul!(Yr, pf_r2c, xr))

    # --- piece B: c2c-rest, PureFFT vs FFTW ---
    P._rfft_dim1_batched!(rb,Y,x,outer); Yf=copy(Y)
    if isnothing(pp.cplan)
        t_c2c=0.0; t_fftw_c2c=0.0
    else
        t_c2c = med(()->begin copyto!(Y,Yf); P.apply_unnormalized!(pp.cplan,Y) end) - med(()->copyto!(Y,Yf))
        restdims = Tuple(2:length(sz))
        Yc = copy(Yf)
        pf_c2c = FFTW.plan_fft(Yc, restdims; flags=FFTW.MEASURE)
        t_fftw_c2c = med(()->begin copyto!(Yc,Yf); pf_c2c*Yc end) - med(()->copyto!(Yc,Yf))
    end
    @printf("%-10s | R2C pf=%.0f fftw=%.0f par=%.3f | C2C pf=%.0f fftw=%.0f par=%.3f\n",
        lbl, t_r2c, t_fftw_r2c, t_fftw_r2c/t_r2c, t_c2c, t_fftw_c2c, t_fftw_c2c/max(t_c2c,1))
end
