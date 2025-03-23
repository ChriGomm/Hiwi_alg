#------------------------------------------------------------------------------
# Header: load module
#------------------------------------------------------------------------------
# ATTENTION: make sure that your present working directory pwd() is set to the folder
# containing script.jl and BASEforHANK.jl. Otherwise adjust the load path.
cd("./src")
# push!(LOAD_PATH, pwd())
# pre-process user inputs for model setup
include("Preprocessor/PreprocessInputs.jl")
include("BASEforHANK.jl")
using .BASEforHANK
using BenchmarkTools, Revise, LinearAlgebra
# set BLAS threads to the number of Julia threads.
# prevents BLAS from grabbing all threads on a machine
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

#------------------------------------------------------------------------------
# initialize parameters to priors to select coefficients of DCTs of Vm, Vk]
# that are retained 
#------------------------------------------------------------------------------
m_par = ModelParameters()
e_set = BASEforHANK.e_set;

BASEforHANK.Random.seed!(e_set.seed)

# Calculate Steady State at prior mode 
println("Calculating the steady state")
# ss_full_young = call_find_steadystate(m_par)
# jldsave("Output/Saves/steadystate_young.jld2", true; ss_full_young) # true enables compression
@load "Output/Saves/steadystate_young.jld2" ss_full_young

Vm_guess = ss_full_young.VmSS;
Vk_guess = ss_full_young.VkSS;
CDF_guess = ss_full_young.CDFSS;
K_guess = ss_full_young.KSS;

# Vm_guess = ss_full_splines.VmSS;
# Vk_guess = ss_full_splines.VkSS;
# CDF_guess = ss_full_splines.CDFSS;
# K_guess = ss_full_splines.KSS;


# -----------------------------------------------



# find_steadystate_splines -----------------------------

# n_par = ss_full_splines.n_par
n_par = NumericalParameters(
        m_par = m_par,
        naggrstates = length(BASEforHANK.state_names),
        naggrcontrols = length(BASEforHANK.control_names),
        aggr_names = BASEforHANK.aggr_names,
        distr_names = BASEforHANK.distr_names,
        method_for_ss_distr="splines" # method for finding the stationary distribution
    )

# Kdiff ---------------------
Paux = n_par.Π^1000          # Calculate ergodic ince distribution from transitions
    distr_y = Paux[1, :]            # stationary income distribution
    N = BASEforHANK.IncomesETC.employment(K_guess, 1.0 ./ (m_par.μ * m_par.μw), m_par)
    r = BASEforHANK.IncomesETC.interest(K_guess, 1.0 ./ m_par.μ, N, m_par) + 1.0
    w = BASEforHANK.IncomesETC.wage(K_guess, 1.0 ./ m_par.μ, N, m_par)
    Y = BASEforHANK.IncomesETC.output(K_guess, 1.0, N, m_par)
    profits = BASEforHANK.IncomesETC.profitsSS_fnc(Y, m_par.RB, m_par)
    unionprofits = (1.0 .- 1.0 / m_par.μw) .* w .* N

    LC = 1.0 ./ m_par.μw * w .* N
    taxrev =
        ((n_par.grid_y / n_par.H) .* LC) -
        m_par.τlev .* ((n_par.grid_y / n_par.H) .* LC) .^ (1.0 - m_par.τprog )
    taxrev[end] =
        n_par.grid_y[end] .* profits -
        m_par.τlev .* (n_par.grid_y[end] .* profits) .^ (1.0 - m_par.τprog )
    incgrossaux = ((n_par.grid_y / n_par.H) .* LC)
    incgrossaux[end] = n_par.grid_y[end] .* profits
    av_tax_rate = dot(distr_y, taxrev) ./ (dot(distr_y, incgrossaux))

    incgross, inc, eff_int = BASEforHANK.IncomesETC.incomes(
        n_par,
        m_par,
        1.0 ./ m_par.μw,
        1.0,
        1.0,
        m_par.RB,
        m_par.τprog ,
        m_par.τlev,
        n_par.H,
        1.0,
        1.0,
        r,
        w,
        N,
        profits,
        unionprofits,
        av_tax_rate,
    )
    #----------------------------------------------------------------------------
    # Initialize policy function (guess/stored values)
    #----------------------------------------------------------------------------

    # # initial guess consumption and marginal values (if not set)
    # if initial
    #     c_guess =
    #         inc[1] .+ inc[2] .* (n_par.mesh_k .* r .> 0) .+ inc[3] .* (n_par.mesh_m .> 0)
    #     if any(any(c_guess .< 0.0))
    #         @warn "negative consumption guess"
    #     end
    #     Vm = eff_int .* mutil(c_guess, m_par)
    #     Vk = (r + m_par.λ) .* mutil(c_guess, m_par)
    #     CDF = n_par.CDF_guess
    # else
        Vm = Vm_guess
        Vk = Vk_guess
        CDF = CDF_guess
    # end
    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    # KS = Ksupply(m_par.RB, r, n_par, m_par, Vm, Vk, CDF, inc, eff_int)


# Ksupply ------------------------

RB_guess = m_par.RB
R_guess = r

c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk = BASEforHANK.SteadyState.find_ss_policies(
        Vm,
        Vk,
        inc,
        eff_int,
        n_par,
        m_par,
        RB_guess,
        R_guess,
    )

# find_ss_distribution_splines --------------------------
# DirectTransitionSplines(

cdf_prime = zeros(eltype(CDF_guess), size(CDF_guess))
cdf_initial = copy(CDF_guess)


w_mesh = 