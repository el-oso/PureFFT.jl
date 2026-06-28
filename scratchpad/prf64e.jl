# Per-column complex FFT(m) in place (NO transpose, the complex-engine F64 path) vs FFTW & vs the
# batched-transpose path. If per-column FFT(m) is at parity, an rfft = pack + per-col FFT + recombine
# (no transpose) should beat the batched-transpose rfft for F64.
using LinearAlgebra, Statistics, Printf
import FFTW, PureFFT
const P = PureFFT
FFTW.set_num_threads(1)
function med(f; secs=1.2, maxn=4000)
    f(); ts=Float64[]; el=0.0
    while el<secs && length(ts)<maxn
        s=time_ns(); f(); d=time_ns()-s; push!(ts,d); el+=d/1e9
    end
    median(ts)
end
for (m, outer) in [(128,256),(256,512),(64,128),(32,4096)]
    A = randn(ComplexF64, m, outer)
    # per-column 1-D plan (the tuned :fast plan the complex Dim1Plan uses)
    pl = P.plan_pfft(ComplexF64, m; variant=:fast, inverse=false)
    t_percol = med(()->begin
        @inbounds for c in 1:outer
            P.apply_unnormalized!(pl, view(A,:,c))
        end
    end)
    pf = FFTW.plan_fft(copy(A), 1; flags=FFTW.MEASURE); B=similar(A)
    t_fftw = med(()->mul!(B,pf,A))
    @printf("m=%-4d outer=%-5d | per-col(no transp)=%.0f  fftw=%.0f  par=%.3f\n", m, outer, t_percol, t_fftw, t_fftw/t_percol)
end
