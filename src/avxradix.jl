# Isolated AVX2 mixed-radix submodule. Kept in its own module so its kernel names (proc_ip!, RPlan,
# _colbf*!, MR*, B36/B64, plan_tree …) don't clash with PureFFT's own recursive/mixedradix code.
# Hot path (applyplan!) is concrete/dispatch-free; plan_tree (construction) is precompile/setup only.
# Float64-only (Vec{4,Float64} = one 256-bit AVX register = 2 complex).
module AvxRadix
include(joinpath(@__DIR__, "avxradix", "planner.jl"))   # chains recursive.jl → kernels.jl → avxport.jl
include(joinpath(@__DIR__, "avxradix", "width8.jl"))    # AVX-512 (Vec{8}) kernels + plan_tree_w8
include(joinpath(@__DIR__, "gen", "transpose.jl"))      # @generated packed transpose (gen_transpose_packed)
include(joinpath(@__DIR__, "gen", "colgen.jl"))         # @generated column-packed P² codelet (gen_pp_codelet!)
include(joinpath(@__DIR__, "gen", "composite.jl"))      # @generated composite-radix column butterfly (avx_colbf_composite)
end
