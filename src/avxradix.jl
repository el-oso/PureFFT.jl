# Isolated AVX2 mixed-radix submodule. Kept in its own module so its kernel names (proc_ip!, RPlan,
# _colbf*!, MR*, B36/B64, plan_tree …) don't clash with PureFFT's own recursive/mixedradix code.
# Hot path (applyplan!) is concrete/dispatch-free; plan_tree (construction) is precompile/setup only.
# Float64-only (Vec{4,Float64} = one 256-bit AVX register = 2 complex).
module AvxRadix
include(joinpath(@__DIR__, "avxradix", "planner.jl"))   # chains recursive.jl → kernels.jl → avxport.jl
end
