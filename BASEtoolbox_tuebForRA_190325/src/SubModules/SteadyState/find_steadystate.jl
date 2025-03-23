@doc raw"""
    find_steadystate(m_par)

Find the stationary equilibrium capital stock.

# Returns
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_steadystate(m_par)

    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------
    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------
    # Read out numerical parameters for starting guess solution with reduced income grid.

    # Numerical parameters 
    #m_par = reconstruct(ModelParameters(), flatten(m_par)); # Scope confusion below otherwise
    n_par = NumericalParameters(m_par = m_par, ny = 4, nm = 20, nk = 20, ϵ = 1e-13);# NumericalParameters(m_par = m_par, ny = 4, nm = 10, nk = 10, ϵ = 1e-6)
    if n_par.verbose
        println("Finding equilibrium capital stock for coarse income grid")
    end

    # Capital stock guesses
    # rmin = 0.0001
    # rmax = (1.0 .- m_par.β) ./ m_par.β - 0.0025
    rmin = 0.25*0.0050201198890283 
    rmax = 1.75*0.0050201198890283 

    Kmax = CompMarketsCapital(rmin, m_par)
    Kmin = CompMarketsCapital(rmax, m_par)
  
    println("Kmin: ", Kmin)
    println("Kmax: ", Kmax)

    # a.) Define excess demand function
    d(
        K,
        initial::Bool = true,
        Vm_guess = zeros(1, 1, 1),
        Vk_guess = zeros(1, 1, 1),
        CDF_guess = n_par.CDF_guess,
    ) = Kdiff(K, n_par, m_par, initial, Vm_guess, Vk_guess, CDF_guess)

    # b.) Find equilibrium capital stock (multigrid on y,m,k)
    KSS = CustomBrent(d, Kmin, Kmax)[1]
    if n_par.verbose
        println("Capital stock is")
        println(KSS)
    end
    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------
    if n_par.verbose
        println("Finding equilibrium capital stock for final income grid")
    end
    # Write changed parameter values to n_par
    n_par = NumericalParameters(
        m_par = m_par,
        naggrstates = length(state_names),
        naggrcontrols = length(control_names),
        aggr_names = aggr_names,
        distr_names = distr_names,
    )

    # Find stationary equilibrium for refined economy
    BrentOut = CustomBrent(d, KSS * 0.9, KSS * 1.1; tol = n_par.ϵ)
    KSS = BrentOut[1]
    VmSS = BrentOut[3][2]
    VkSS = BrentOut[3][3]
    CDFSS = BrentOut[3][4]
    if n_par.verbose
        println("Capital stock is")
        println(KSS)
    end
    return KSS, VmSS, VkSS, CDFSS, n_par, m_par

end

@doc raw"""
    find_steadystate(m_par)

Find the stationary equilibrium capital stock.

# Returns
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
"""
function find_steadystate_splines(m_par; K_guess=nothing, Vm_guess=nothing, Vk_guess=nothing, distr_guess=nothing)

    # -------------------------------------------------------------------------------
    ## STEP 1: Find the stationary equilibrium for coarse grid
    # -------------------------------------------------------------------------------

    #-------------------------------------------------------
    # Income Process and Income Grids
    #-------------------------------------------------------

    # Write changed parameter values to n_par

    n_par = NumericalParameters(
        m_par = m_par,
        naggrstates = length(state_names),
        naggrcontrols = length(control_names),
        aggr_names = aggr_names,
        distr_names = distr_names,
        method_for_ss_distr="splines" # method for finding the stationary distribution
    )

    # a.) Define excess demand function
    # set initial values of not provided
    if Vm_guess === nothing || Vk_guess === nothing || distr_guess === nothing
        Vm_guess = zeros(1, 1, 1)
        Vk_guess = zeros(1, 1, 1)
        distr_guess = n_par.distr_guess
        initial = true
    else
        initial = false
    end
    K_guess = K_guess === nothing ? 30.0 : K_guess

    d(
        K,
        initial::Bool = initial,
        Vm_guess = Vm_guess,
        Vk_guess = Vk_guess,
        distr_guess = distr_guess,
    ) = Kdiff(K, n_par, m_par, initial, Vm_guess, Vk_guess, distr_guess)


    # -------------------------------------------------------------------------------
    ## STEP 2: Find the stationary equilibrium for final grid
    # -------------------------------------------------------------------------------

    if n_par.verbose
        println("Finding equilibrium capital stock for spline method")
    end
    # Find stationary equilibrium for refined economy
    println("KSS: ", K_guess*0.9)
    BrentOut = CustomBrent(d, K_guess * 0.9, K_guess * 1.5; tol = n_par.ϵ)
    KSS = BrentOut[1]
    VmSS = BrentOut[3][2]
    VkSS = BrentOut[3][3]
    distrSS = BrentOut[3][4]
    if n_par.verbose
        println("Capital stock is")
        println(KSS)
    end
    return KSS, VmSS, VkSS, distrSS, n_par, m_par

end
