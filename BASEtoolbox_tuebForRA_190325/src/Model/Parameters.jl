@metadata prior nothing
@metadata label ""
@metadata latex_label L""
@doc raw"""
ModelParameters()

Collect all model parameters with calibrated values / priors for estimation in a `struct`.

Uses packages `Parameters`, `FieldMetadata`, `Flatten`. Boolean value denotes
whether parameter is estimated.

# Example
```jldoctest
julia> m_par = ModelParameters();
julia> # Obtain vector of prior distributions of parameters that are estimated.
julia> priors = collect(metaflatten(m_par, prior))
```
"""
@label @latex_label @prior @flattenable @with_kw struct ModelParameters{T}
    # variable = value  | ascii name 	| LaTex name 	| Prior distribution | estimated? # description

    # Household preference parameters
    ξ::T = 4.0 | "xi" | L"\xi" | _ | false # risk aversion
    γ::T = 2.0 | "gamma" | L"\gamma" | _ | false # inverse Frisch elasticity
    β::T = 0.98255 | "beta" | L"\beta" | _ | false # discount factor
    λ::T = 0.065 | "lambda" | L"\lambda" | _ | false # adjustment probability
    γ_scale::T = 0.2 | "gamma_scale" | L"\gamma_{scale}" | _ | false # disutiltiy of labor

    # Individual income process
    ρ_h::T = 0.98 | "rho" | L"\rho" | _ | false # autocorrelation income shock
    σ_h::T = 0.12 | "sigma" | L"\sigma" | _ | false # std of income shocks (steady state)
    ι::T = 1 / 16 | "iota" | L"\iota" | _ | false # probability to return worker
    ζ::T = ι/6 | "zeta" | L"\zeta" | _ | false # probability to become entrepreneur # was 1/4500

    # Technological parameters
    α::T = 0.318 | "alpha" | L"\alpha" | _ | false # capital share
    δ_0::T = (0.07 + 0.016) / 4 | "delta" | L"\delta" | _ | false # depreciation rate
    δ_s::T = 0.1 | "delta_s" | L"\delta_s" | Gamma(gamma_pars(5.0, 2.0^2)...) | true  # depreciation rate increase (flex utilization)
    ϕ::T = 0.5 | "phi" | L"\phi" | Gamma(gamma_pars(4.0, 2.0^2)...) | true  # Capital adjustment costs
    μ::T = 1.1 | "mu" | L"\mu" | _ | false # Price markup
    κ::T = 1 / 11 | "kappa" | L"\kappa" | Gamma(gamma_pars(0.1, 0.03^2)...) | true  # Price adjustment costs (in terms of Calvo probs.)
    μw::T = 1.1 | "mu_w" | L"\mu_w" | _ | false # wage markup
    κw::T = 1 / 11 | "kappa_w" | L"\kappa_w" | Gamma(gamma_pars(0.1, 0.03^2)...) | true  # Wage  adjustment costs (in terms of Calvo probs.)

    # Further steady-state parameters
    ψ::T = 0.1 | "psi" | L"\psi" | _ | false # steady-state bond to capital ratio
    τlev::T = 0.8225 | "tau_lev" | L"\tau^L" | _ | false # steady-state income tax rate level
    τprog ::T = 0.1022 | "tau_pro" | L"\tau^P" | _ | false # steady-state income tax rate progressivity

    R::T = 1.01 | "R" | L"R" | _ | false # steady state rate of return capital (unused)
    K::T = 40.0 | "K" | L"K" | _ | false # steady state quantity of capital (unused)
    π::T = 1.0 .^ 0.25 | "Pi" | L"\pi" | _ | false # Steady State Inflation
    RB::T = π * (1.0 .^ 0.25) | "RB" | L"RB" | _ | false # Nominal Interest Rate
    Rbar::T = (π * (1.09 .^ 0.25) .- 1.0) | "Rbar" | L"\bar R" | _ | false # borrowing wedge in interest rate
    ASHIFT::T = π * (1.0 .^ 0.25) | "ASHIFT" | L"ASHIFT" | _ | false # borrowing wedge in interest rate

    # Tradable shares
    ωΠ::T = 0.2 | "omegaPi" | L"\omega^{\Pi}" | Beta(beta_pars(0.2, 0.075^2)...) | false # fraction of tradable firm-profits
    ωΠbar::T = 0.2 | "omegaPiBar" | L"\omega^{\Pi}Bar" | _ | false # fraction of tradable firm-profits
    ιΠ::T = 0.016 | "iotaPi" | L"\iota^{\Pi}" | _ | false # fraction of shares that retire / depreciate
    shiftΠ::T = 0.5 | "shiftPi" | L"\shift^{\Pi}" | Beta(beta_pars(0.5, 0.25^2)...) | true # fraction of tradable firm-profits

    # exogeneous aggregate "shocks"
    ρ_A::T = 0.9 | "rho_A" | L"\rho_A" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of bond-spread
    σ_A::T = 0.0 | "sigma_A" | L"\sigma_A" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std of bond-spread shock

    ρ_Z::T = 0.9 | "rho_Z" | L"\rho_Z" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of TFP
    σ_Z::T = 0.0 | "sigma_Z" | L"\sigma_Z" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std of TFP

    ρ_ZI::T = 0.9 | "rho_Psi" | L"\rho_\Psi" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of TFP
    σ_ZI::T =
        0.0 | "sigma_Psi" | L"\sigma_\Psi" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std of TFP

    ρ_μ::T = 0.9 | "rho_mu" | L"\rho_\mu" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of price markup
    σ_μ::T =
        0.0 | "sigma_mu" | L"\sigma_\mu" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std of cost push shock

    ρ_μw::T = 0.9 | "rho_muw" | L"\rho_{\mu w}" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. of wage markup
    σ_μw::T =
        0.0 |
        "sigma_muw" |
        L"\sigma_{\mu w}" |
        InverseGamma(ig_pars(0.001, 0.02^2)...) |
        true  # Std of cost push shock

    # income risk
    ρ_s::T = 0.84 | "rho_sigma" | L"\rho_s" | Beta(beta_pars(0.7, 0.2^2)...) | true  # Persistence of idiosyncratic income risk
    σ_Sshock::T =
        0.0 | "sigma_Sshock" | L"\sigma_s" | Gamma(gamma_pars(0.65, 0.3^2)...) | true  # std of idiosyncratic income risk
    Σ_n::T = 0.0 | "Sigma_n" | L"\Sigma_N" | Normal(0.0, 100.0) | true  # reaction of risk to employment

    # monetary policy
    ρ_R::T = 0.7 | "rho_R" | L"\rho_R" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. in Taylor rule
    σ_Rshock::T =
        0.0 | "sigma_Rshock" | L"\sigma_R" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std R
    θ_π::T = 2.0 | "theta_pi" | L"\theta_\pi" | Normal(1.7, 0.3) | true  # Reaction to inflation
    θ_Y::T = 0.125 | "theta_Y" | L"\theta_y" | Normal(0.125, 0.05) | true  # Reaction to inflation

    # fiscal policy
    γ_B::T = 0.1 | "gamma_B" | L"\gamma_B" | Gamma(gamma_pars(0.1, 0.075^2)...) | true  # reaction of deficit to debt
    γ_π::T = 0.0 | "gamma_pi" | L"\gamma_{\pi}" | Normal(0.0, 1.0) | true  # reaction of deficit to inflation
    γ_Y::T = 0.0 | "gamma_Y" | L"\gamma_Y" | Normal(0.0, 1.0) | true  # reaction of deficit to output
    ρ_Gshock::T = 0.98 | "rho_Gshock" | L"\rho_D" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. in structural deficit
    σ_Gshock::T =
        0.00 | "sigma_G" | L"\sigma_D" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std G

    ρ_τ::T = 0.5 | "rho_tau" | L"\rho_\tau" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. in tax level
    γ_Bτ::T = 0.0 | "gamma_Btau" | L"\gamma_B^\tau" | Normal(0.0, 1.0) | true  # reaction of tax level to debt
    γ_Yτ::T = 0.0 | "gamma_Ytau" | L"\gamma_Y_\tau" | Normal(0.0, 1.0) | true  # reaction of tax level to output
    γ_Wτ::T = 0.0 | "gamma_Wtau" | L"\gamma_W_\tau" | Normal(0.0, 1.0) | false  # reaction of tax level to output

    ρ_P::T = 0.5 | "rho_P" | L"\rho_P" | Beta(beta_pars(0.5, 0.2^2)...) | true  # Pers. in tax progr. rule
    σ_Tprogshock::T =
        0.0 | "sigma_Pshock" | L"\sigma_P" | InverseGamma(ig_pars(0.001, 0.02^2)...) | true  # Std tax progr.
    γ_BP::T = 0.0 | "gamma_BP" | L"\gamma_B^P" | Normal(0.0, 1.0) | false # reaction of tax progr. to debt
    γ_YP::T = 0.0 | "gamma_YP" | L"\gamma_Y^P" | Normal(0.0, 1.0) | false # reaction of tax progr. to output
    γ_WP::T = 0.0 | "gamma_WP" | L"\gamma_W^P" | Normal(0.0, 1.0) | false # reaction of tax progr. to output

    # auxiliary shock parameters
    ρ_Rshock::T =
        1e-8 | "rho_Rshock" | L"\rho_{Rshock}" | Beta(beta_pars(0.5, 0.2^2)...) | false # Shock persistence (MA)
    ρ_Pshock::T =
        1e-8 | "rho_Pshock" | L"\rho_{Pshock}" | Beta(beta_pars(0.5, 0.2^2)...) | false # Shock persistence (MA)
    ρ_Sshock::T =
        1e-8 | "rho_Sshock" | L"\rho_{Sshock}" | Beta(beta_pars(0.5, 0.2^2)...) | false # Shock persistence (MA)
end

@doc raw"""
NumericalParameters()

Collect parameters for the numerical solution of the model in a `struct`.

Use package `Parameters` to provide initial values.

# Example
```jldoctest
julia> n_par = NumericalParameters(mmin = -6.6, mmax = 1000)
```
"""
@with_kw struct NumericalParameters
    # Numerical Parameters to be set in advance
    m_par::ModelParameters = ModelParameters()
    ny::Int = 4     # ngrid income (4 is the coarse grid used initially in finding the StE)
    nk::Int = 100      # ngrid illiquid assets (capital) (10 is the coarse grid used initially in finding the StE)
    nm::Int = 100      # ngrid liquid assets (bonds) (10 is the coarse grid used initially in finding the StE)
    ny_copula::Int = 4  # ngrid for copula in income (rule of thumb: divide ny, w/o entrepreneur, by two)
    nk_copula::Int = 20   # ngrid for copula in illiquid assets (capital, rule of thumb: divide nk by twelve)
    nm_copula::Int = 20  # ngrid for copula in liquid assets (bonds, rule of thumb: divide nm by twelve)
    w_sel_k::Vector{Int} = collect(1:3:nk) # select every *?* gridpoint
    w_sel_m::Vector{Int} = collect(1:3:nm) # select every *?* gridpoint
    kmin::Float64 = 0.0       # gridmin capital
    kmax::Float64 = 80.0    # gridmax capital
    mmin::Float64 = 0.0      # gridmin bonds
    mmax::Float64 = 50.0    # gridmax bonds
    ϵ::Float64 = 1e-13 # precision of solution 

    method_for_ss_distr::String="splines" # method for finding the stationary distribution
    capital_aggregation_method::String = "integration" # one of ["discrete", "integration"]

    sol_algo::Symbol = :schur # options: :schur (Klein's method), :lit (linear time iteration), :litx (linear time iteration with Howard improvement)
    verbose::Bool = true   # verbose model
    reduc_value::Float64 = 1e-3   # Lost fraction of "energy" in the DCT compression for value functions
    reduc_marginal_value::Float64 = 1e-3   # Lost fraction of "energy" in the DCT compression for value functions

    further_compress::Bool = true   # run model-reduction step based on MA(∞) representation
    further_compress_critC = eps()  # critical value for eigenvalues for Value functions
    further_compress_critS = ϵ      # critical value for eigenvalues for copula

    # Parameters that will be overwritten in the code
    aggr_names::Array{String,1} = ["Something"] # Placeholder for names of aggregates
    distr_names::Array{String,1} = ["Something"] # Placeholder for names of distributions

    naggrstates::Int = 16 # (placeholder for the) number of aggregate states
    naggrcontrols::Int = 16 # (placeholder for the) number of aggregate controls
    nstates::Int = ny + nk + nm + naggrstates - 3 # (placeholder for the) number of states + controls in total
    ncontrols::Int = 16 # (placeholder for the) number of controls in total
    ntotal::Int = nstates + ncontrols     # (placeholder for the) number of states+ controls in total
    n_agg_eqn::Int = nstates + ncontrols     # (placeholder for the) number of aggregate equations
    naggr::Int = length(aggr_names)     # (placeholder for the) number of aggregate states + controls
    ntotal_r::Int = nstates + ncontrols# (placeholder for the) number of states + controls in total after reduction
    nstates_r::Int = nstates# (placeholder for the) number of states in total after reduction
    ncontrols_r::Int = ncontrols# (placeholder for the) number of controls in total after reduction

    PRightStates::AbstractMatrix = Diagonal(ones(nstates)) # (placeholder for the) Matrix used for second stage reduction (states only)
    PRightAll::AbstractMatrix = Diagonal(ones(ntotal))  # (placeholder for the) Matrix used for second stage reduction 

    # income grid
    grid_y::Array{Float64,1} = [
        exp.(Tauchen(m_par.ρ_h, ny - 1)[1] .* m_par.σ_h ./ sqrt(1.0 .- m_par.ρ_h .^ 2))
        (m_par.ζ .+ m_par.ι) / m_par.ζ
    ]
    # Transition matrix for income in steady state
    Π::Matrix{Float64} = [
        Tauchen(m_par.ρ_h, ny - 1)[2].*(1.0 .- m_par.ζ) m_par.ζ.*ones(ny - 1)
        m_par.ι./(ny-1)*ones(1, ny - 1) 1.0 .- m_par.ι
    ]
    # bounds of income bins (except entrepreneur) 
    bounds_y::Array{Float64,1} = Tauchen(m_par.ρ_h, ny - 1)[3]

    H::Float64 = ((Π^1000)[1, 1:end-1]' * grid_y[1:end-1]) # stationary equilibrium average human capital
    HW::Float64 = (1.0 / (1.0 - (Π^1000)[1, end]))     # stationary equilibrium fraction workers

    # initial gues for stationary distribution (needed if iterative procedure is used)
    # CDF_guess::Array{Float64,3} = cumsum(cumsum(ones(nm, nk, ny) / (nm * nk * ny), dims=1), dims=2)
    distr_guess::Array{Float64,3} = vcat(cumsum(ones(nm,  nk, ny) / sum(nm * ny), dims=1),reshape(cumsum(ones(nk,ny)/sum(nk * ny),dims = 1),1,nk,ny))

    # dist_guess::Array{Float64,3} = ones(nm, nk, ny) / (nm * nk * ny)

    # grid illiquid assets:
    grid_k::Array{Float64,1} = #ones(nk)#range(kmin,stop=kmax,length=nk)# grid_k[1:nk] =# 
    # vcat(exp.(range(log(kmin + 1.0), stop = log(kmax + 1.0), length = nk-1)) .- 1.0,ones(1)*150)
    # grid_k[nk+1]= 500
        # vcat((range(0, stop = sqrt(kmax - kmin ), length = nk-2)) .^ 2 .+ kmin,vcat(ones(1)*110,ones(1)*150))
        vcat((range(0, stop = sqrt(kmax - kmin ), length = nk-1)) .^ 2 .+ kmin,ones(1)*150)
    # grid liquid assets:
    grid_m::Array{Float64,1} = 
    # vcat(exp.(range(0, stop = log(mmax - mmin + 1.0), length = nm-1)) .+ mmin .- 1.0,ones(1)*150)
    #ones(nm)#range(mmin,stop=mmax,length=nk)
    # grid_m[1:nm] =  grid_m[nm+1] = 500
        vcat((range(0, stop = sqrt(mmax - mmin ), length = nm-1)) .^ 2 .+ mmin,ones(1)*150)
    aux::Float64 = 2*grid_k[end-2]-(grid_k[end-3]+grid_k[end-2])/2
    grid_k_cdf::Array{Float64,1} = vcat(vcat(zeros(1),(grid_k[2:end-3]+grid_k[3:end-2])/2),vcat(ones(1)*aux,ones(1)*150))

    # meshes for income, bonds, capital
    mesh_y::Array{Float64,3} = repeat(reshape(grid_y, (1, 1, ny)), outer = [nm, nk, 1])
    mesh_m::Array{Float64,3} = repeat(reshape(grid_m, (nm, 1, 1)), outer = [1, nk, ny])
    mesh_k::Array{Float64,3} = repeat(reshape(grid_k, (1, nk, 1)), outer = [nm, 1, ny])

    # grid for copula marginal distributions
    copula_marginal_m::Array{Float64,1} =
        collect(range(0.0, stop = 1.0, length = nm_copula))
    copula_marginal_k::Array{Float64,1} =
        collect(range(0.0, stop = 1.0, length = nk_copula))
    copula_marginal_y::Array{Float64,1} =
        collect(range(0.0, stop = 1.0, length = ny_copula))

    # Storage for linearization results
    LOMstate_save::Array{Float64,2} = zeros(nstates, nstates)
    State2Control_save::Array{Float64,2} = zeros(ncontrols, nstates)
end



@doc raw"""
EstimationSettings()

Collect settings for the estimation of the model parameters in a `struct`.

Use package `Parameters` to provide initial values. Input and output file names are
stored in the fields `mode_start_file`, `data_file`, `save_mode_file` and `save_posterior_file`.
"""
@with_kw struct EstimationSettings
    shock_names::Array{Symbol,1} = shock_names # set in Model/input_aggregate_names.jl
    observed_vars_input::Array{Symbol,1} = [
        :Ygrowth,
        :Igrowth,
        :Cgrowth,
        :N,
        :wgrowth,
        :RB,
        :π,
        :TOP10Wshare,
        :TOP10Ishare,
        :τprog,
        :σ,
    ]

    nobservables = length(observed_vars_input)

    data_rename::Dict{Symbol,Symbol} = Dict(
        :pi => :π,
        :sigma2 => :σ,
        :tauprog => :τprog,
        :w90share => :TOP10Wshare,
        :I90share => :TOP10Ishare,
    )

    me_treatment::Symbol = :unbounded
    me_std_cutoff::Float64 = 0.2

    meas_error_input::Array{Symbol,1} = [:TOP10Wshare, :TOP10Ishare]
    meas_error_distr::Array{InverseGamma{Float64},1} =
        [InverseGamma(ig_pars(0.01, 0.01^2)...), InverseGamma(ig_pars(0.01, 0.01^2)...)]

    # Leave empty to start with prior mode
    mode_start_file::String = "Output/Saves/parameter_example.jld2"

    data_file::String = "Data/bbl_data_inequality.csv"
    save_mode_file::String = "Output/Saves/HANK_mode.jld2"
    save_posterior_file::String = "Output/Saves/HANK_chain.jld2"

    estimate_model::Bool = true

    max_iter_mode::Int = 3
    optimizer::Optim.AbstractOptimizer = NelderMead()
    compute_hessian::Bool = false    # true: computes Hessian at posterior mode; false: sets Hessian to identity matrix
    f_tol::Float64 = 1.0e-4
    x_tol::Float64 = 1.0e-4

    multi_chain_init::Bool = false
    ndraws::Int = 400
    burnin::Int = 100
    mhscale::Float64 = 0.00015
    debug_print::Bool = true
    seed::Int = 778187

end
