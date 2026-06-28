# Is PureFFT's batched complex FFT(m) itself at parity vs FFTW batched complex FFT(m)?
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
# batched complex FFT of `outer` columns of length m, dim-1 (contiguous) — mirror what r2c does.
for (m, outer) in [(128,256),(256,512),(64,128),(32,4096)]
    radices = ispow2(m) ? "pow2" : "mr"
    bp = P.BatchPlan8(Float64, m; forward=true)
    Mc = clamp((8192 ÷ m) & ~3, 4, outer)
    stage = Vector{ComplexF64}(undef, m*Mc)
    A = randn(ComplexF64, m, outer)
    # PureFFT: transpose-pack chunk, batched fft, transpose back (matches r2c minus recombine)
    es = sizeof(ComplexF64)
    pA = pointer(A); pS = pointer(stage)
    t_pf = med(()->GC.@preserve A stage bp begin
        t0=0; while t0<outer; mc=min(Mc,outer-t0)
            P._transpose_block!(pS, pA+t0*m*es, m, mc)
            P._batched_apply!(bp, pS, 0, mc, 1)
            P._transpose_block!(pA+t0*m*es, pS, mc, m)
            t0+=mc; end
    end)
    # FFTW: batched complex fft along dim 1
    pf = FFTW.plan_fft(copy(A), 1; flags=FFTW.MEASURE)
    B = similar(A)
    t_fw = med(()->mul!(B, pf, A))
    @printf("m=%-4d outer=%-5d %-4s | pf(transp+fft+transp)=%.0f fftw=%.0f par=%.3f\n", m, outer, radices, t_pf, t_fw, t_fw/t_pf)
end
