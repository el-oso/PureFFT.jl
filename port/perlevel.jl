include(joinpath("/home/el_oso/Documents/claude/PureFFT.jl","port","recursive.jl"))
using Printf, Statistics
const LIB = joinpath(pwd(),"bench","rustfft_compare","rust","target","release","librustfft_bench.so")
rplan(n)=ccall((:rfft_plan,LIB),Ptr{Cvoid},(Csize_t,),n)
rproc(h,d,n)=ccall((:rfft_process,LIB),Cvoid,(Ptr{Cvoid},Ptr{ComplexF64},Csize_t),h,d,n)
rfree(h)=ccall((:rfft_free,LIB),Cvoid,(Ptr{Cvoid},),h)
function compare(n,jp)
    h=rplan(n); jb=copy(seeded(n)); rb=copy(seeded(n)); kit=max(300,round(Int,3e8/(n*log2(n))))
    for _ in 1:20; for _ in 1:kit; applyplan!(jp,jb);end; for _ in 1:kit; rproc(h,rb,n);end; end
    rt=Float64[];jt=Float64[]
    GC.@preserve rb begin
      for _ in 1:151
        t=time_ns();for _ in 1:kit;applyplan!(jp,jb);end;push!(jt,(time_ns()-t)/kit)
        t=time_ns();for _ in 1:kit;rproc(h,rb,n);end;push!(rt,(time_ns()-t)/kit)
      end
    end
    rfree(h); jm=median(jt);rm=median(rt)
    @printf("n=%-5d %-10s julia %8.1f(σ%.1f%%) rust %8.1f(σ%.1f%%) ratio=%.3f %s\n",n,"",jm,100std(jt)/jm,rm,100std(rt)/rm,rm/jm, rm/jm≥0.96 ? "✓" : "✗")
end
compare(36, RPlan(B36(true)))                       # leaf
compare(144, RPlan(MR4(B36(true),true)))            # +1 R4 level
compare(180, RPlan(MR5(B36(true),true)))            # +1 R5 level
compare(576, RPlan(MR4(MR4(B36(true),true),true)))  # +2 R4 levels
compare(900, RPlan(MR5(MR5(B36(true),true),true)))  # +2 R5 levels
