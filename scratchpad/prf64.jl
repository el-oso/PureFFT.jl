# Profile F64 rfft: split r2c-dim (batched dim1) vs c2c-rest (apply_unnormalized! on half-spectrum).
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
FFTW.set_num_threads(1)

function med(f; secs=1.5, maxn=2000)
    f(); ts = Float64[]; el=0.0
    while el < secs && length(ts) < maxn
        s=time_ns(); f(); d=time_ns()-s; push!(ts,d); el+=d/1e9
    end
    median(ts)
end

const SHAPES = [("256x256",(256,256)),("512x512",(512,512)),("128x128",(128,128)),
                ("64x64x64",(64,64,64)),("240x240",(240,240)),("96x96x96",(96,96,96))]

for (lbl,sz) in SHAPES
    x = randn(Float64, sz...)
    region = ntuple(identity, length(sz))
    pp = P._pure_plan_rfft_nd(x, region)
    pf = FFTW.plan_rfft(copy(x), region; flags=FFTW.MEASURE)
    Y  = Array{ComplexF64}(undef, pp.cplxsz...)
    yf = Array{ComplexF64}(undef, pp.cplxsz...)

    outer = P._prod_after(pp.realsz, pp.d)
    # phase 1: r2c dim only
    t_r2c = med(() -> begin
        if !isnothing(pp.bd1)
            P._rfft_dim1_batched!(pp.bd1, Y, x, outer)
        end
    end)
    # phase 2: c2c rest only (on an already-r2c'd Y; need filled Y so reuse)
    P._rfft_dim1_batched!(pp.bd1, Y, x, outer)
    Yfilled = copy(Y)
    t_c2c = med(() -> begin
        copyto!(Y, Yfilled)
        isnothing(pp.cplan) || P.apply_unnormalized!(pp.cplan, Y)
    end)
    t_copy = med(() -> copyto!(Y, Yfilled))   # subtract the copy overhead
    t_c2c_net = t_c2c - t_copy

    t_full = med(() -> mul!(Y, pp, x))
    t_fftw = med(() -> mul!(yf, pf, x))

    @printf("%-10s  full=%7.1fns fftw=%7.1fns ratio=%.3f | r2c=%7.1f (%4.1f%%) c2c=%7.1f (%4.1f%%) copyovh=%.0f\n",
        lbl, t_full, t_fftw, t_fftw/t_full,
        t_r2c, 100*t_r2c/t_full, t_c2c_net, 100*t_c2c_net/t_full, t_copy)
end
