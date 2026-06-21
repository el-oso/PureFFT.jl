include("/home/el_oso/Documents/claude/PureFFT.jl/port/recursive.jl")
using InteractiveUtils
buf=zeros(ComplexF64,576); tw=[V4f((1.0,0,0,0)) for _ in 1:108]
io=IOBuffer(); code_native(io, _trans4!, Tuple{Vector{ComplexF64},Int,Vector{ComplexF64},Int,Val{36}}; syntax=:intel, debuginfo=:none)
a=String(take!(io))
println("_trans4! M=36: lines=",count(==('\n'),a))
println("  vperm2f128/vinsert/vextract (cross-lane): ", length(collect(eachmatch(r"vperm2f128|vinsertf128|vextractf128",a))))
println("  spills (mov to/from [rsp/rbp]): ", length(collect(eachmatch(r"(mov|vmov)\w*\s+[^,]*,\s*\[r[sb]p|(mov|vmov)\w*\s+\[r[sb]p",a))))
println("  vmovup/vmovap total: ", length(collect(eachmatch(r"vmovup|vmovap",a))))
# colbf4 too
io=IOBuffer(); code_native(io, _colbf4!, Tuple{Vector{ComplexF64},Int,Val{36},Vector{V4f},V4f}; syntax=:intel, debuginfo=:none)
b=String(take!(io))
println("_colbf4! M=36: lines=",count(==('\n'),b), "  spills: ", length(collect(eachmatch(r"\[rsp",b))), "  vfmaddsub: ", length(collect(eachmatch(r"vfmaddsub",b))))
