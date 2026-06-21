# Isolated faithful-port submodule (RustFFT AVX2 mechanical port). Kept in its own module so its
# kernel names (proc_ip!, RPlan, _colbf*!, MR*, B36/B64, plan_tree …) don't clash with PureFFT's own
# recursive/mixedradix code. Hot path (applyplan!) is concrete/dispatch-free; plan_tree (construction)
# is precompile/setup only. f64-only (Vec{4,Float64} ≅ __m256d).
module RustPort
include(joinpath(@__DIR__, "rustport", "planner.jl"))   # chains recursive.jl → kernels.jl → avxport.jl
end
