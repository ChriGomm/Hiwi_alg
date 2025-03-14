# __precompile__(false)
# Code runs on Julia 1.9.3
# ------------------------------------------------------------------------------
## Package Calls
# ------------------------------------------------------------------------------

module SteadyState
# Sibling modules
using ..Tools
using ..Parsing
using ..IncomesETC

# 3rd Party modules
using   LinearAlgebra,
        SparseArrays,
        Distributions,
        Roots,
        ForwardDiff,
        Flatten,
        PCHIPInterpolation,
        TimerOutputs
    
using Parameters: @unpack
using KrylovKit: eigsolve
using FFTW: dct, ifft

export  updateV!,
        updateV,
        EGM_policyupdate!,
        EGM_policyupdate,
        DirectTransition_Splines!,
        DirectTransition!,
        DirectTransition,
        DirectTransitionSplines,
        Ksupply,
        find_steadystate,
        find_steadystate_splines,
        first_stage_reduction,
        expected_value
        

# ------------------------------------------------------------------------------
## Define Functions
# ------------------------------------------------------------------------------
include("../Model/input_aggregate_names.jl") # this dependency is not ideal
include("SteadyState/EGM/EGM_policyupdate.jl")
include("SteadyState/EGM/updateV.jl")
include("SteadyState/find_steadystate.jl")
include("SteadyState/first_stage_reduction.jl")
# include("SteadyState/IM_fcns/fcn_directtransition.jl")
# include("SteadyState/IM_fcns/fcn_directtransition_total_wealth_illiquid.jl")
include("SteadyState/IM_fcns/fcn_directtransition_conditionals.jl")
include("SteadyState/IM_fcns/fcn_kdiff.jl")
include("SteadyState/IM_fcns/fcn_ksupply.jl")
include("SteadyState/IM_fcns/fcn_makeweights.jl")
include("SteadyState/IM_fcns/fcn_maketransition.jl")
include("SteadyState/IM_fcns/fcn_MultipleDirectTransitions.jl")
end # module BASEforHANK.SteadyState
