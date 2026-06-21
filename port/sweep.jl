include("/home/el_oso/Documents/claude/PureFFT.jl/port/planner.jl")
import FFTW; using Printf, Statistics
const LIB=joinpath(pwd(),"bench","rustfft_compare","rust","target","release","librustfft_bench.so")
rplan(n)=ccall((:rfft_plan,LIB),Ptr{Cvoid},(Csize_t,),n); rproc(h,d,n)=ccall((:rfft_process,LIB),Cvoid,(Ptr{Cvoid},Ptr{ComplexF64},Csize_t),h,d,n); rfree(h)=ccall((:rfft_free,LIB),Cvoid,(Ptr{Cvoid},),h)
function measure(n, jp)
    y=copy(seeded(n)); applyplan!(jp,y); rel=maximum(abs.(y.-FFTW.fft(seeded(n))))/maximum(abs.(FFTW.fft(seeded(n))))
    h=rplan(n); jb=copy(seeded(n)); rb=copy(seeded(n)); kit=max(200,round(Int,2.5e8/(n*log2(n))))
    for _ in 1:15; for _ in 1:kit; applyplan!(jp,jb);end; for _ in 1:kit; rproc(h,rb,n);end; end
    rt=Float64[];jt=Float64[]; GC.@preserve rb for _ in 1:121
        t=time_ns();for _ in 1:kit;applyplan!(jp,jb);end;push!(jt,(time_ns()-t)/kit)
        t=time_ns();for _ in 1:kit;rproc(h,rb,n);end;push!(rt,(time_ns()-t)/kit) end
    rfree(h); (rel, median(rt)/median(jt), 100std(jt)/median(jt))
end
sizes = [720,1080,1440,1620,2160,2880,3240,4320,5760,6480,8640,9720,11520,12960,17280,25920,34560]
for n in sizes
    p2,p3,p5,_=factor235(n); jp=plan_tree(n,true)
    if isnothing(jp); @printf("n=%-6d (2^%d·3^%d·5^%d)  unsupported (fallback)\n",n,p2,p3,p5); continue; end
    rel,ratio,sj = measure(n,jp)
    @printf("n=%-6d (2^%d·3^%d·5^%d) ratio=%.3f rel=%.0e (σ%.1f) %s\n",n,p2,p3,p5,ratio,rel,sj, rel<1e-9 ? (ratio≥0.96 ? "✓" : "✗") : "WRONG")
end
