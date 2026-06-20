# Compile-time interface contract for FFT plans (TypeContracts.jl).
#
# Every plan type (the current PureFFTPlan, and the BlockedPlan / autotuned plan added later)
# must provide the same small interface, so callers like `pfft!` depend on the interface, not
# the concrete struct. `@verify` checks method existence + inferred return types at
# PRECOMPILE time and is eliminated by the trimmer — zero runtime cost (confirmed by
# bench/alloccheck.jl staying at 0 bytes). See [[feedback-typecontracts]]: implementing
# methods carry explicit concrete return-type annotations so inference matches the contract.

using TypeContracts
using TypeContracts: interface_trait, Implemented, NotImplemented

"""
    AbstractFFTPlan{T}

Supertype of all PureFFT plans. Concrete plans must satisfy the [`@contract`](@ref) below:
`plan_length`, `plan_inverse`, and `apply_unnormalized!`.
"""
abstract type AbstractFFTPlan{T} end

function plan_length end        # transform length N
function plan_inverse end       # direction flag (inverse?)
function apply_unnormalized! end # run the raw (un-normalized) transform in place

@contract AbstractFFTPlan begin
    plan_length(::Self)::Int
    plan_inverse(::Self)::Bool
    apply_unnormalized!(::Self, ::AbstractVector)
end
