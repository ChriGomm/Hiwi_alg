@doc raw"""
    Kdiff(K_guess, 
    n_par, 
    m_par, 
    initial = true, 
    Vm_guess = zeros(1, 1, 1), 
    Vk_guess = zeros(1, 1, 1), 
    distr_guess = zeros(1, 1, 1))

Calculate the difference between the capital stock that is assumed and the capital
stock that prevails under that guessed capital stock's implied prices when
households face idiosyncratic income risk (Aiyagari model).

Requires global functions from the IncomesETC module `incomes()` and [`Ksupply()`](@ref).

# Arguments
- `K_guess::Float64`: capital stock guess
- `n_par::NumericalParameters`, 
- `m_par::ModelParameters`
- 5 optional arguments:
    - `initial::Bool = true`: whether to use initial guess for marginal values
    - `Vm_guess::AbstractArray = zeros(1, 1, 1)`: initial guess for marginal value of liquid assets
    - `Vk_guess::AbstractArray = zeros(1, 1, 1)`: initial guess for marginal value of illiquid assets
    - `CDF_guess::AbstractArray = zeros(1, 1, 1)`: initial guess for stationary distribution
"""
function Kdiff(
    K_guess::Float64,
    n_par,
    m_par,
    initial::Bool = true,
    Vm_guess::AbstractArray = zeros(1, 1, 1),
    Vk_guess::AbstractArray = zeros(1, 1, 1),
    distr_guess::AbstractArray = zeros(1, 1, 1),
)
    #----------------------------------------------------------------------------
    # Calculate other prices from capital stock
    #----------------------------------------------------------------------------
    # #----------------------------------------------------------------------------
    # # Array (inc) to store incomes
    # # inc[1] = labor income , inc[2] = rental income,
    # # inc[3]= liquid assets income, inc[4] = capital liquidation income
    # #----------------------------------------------------------------------------
    Paux = n_par.Π^1000          # Calculate ergodic ince distribution from transitions
    distr_y = Paux[1, :]            # stationary income distribution
    N = employment(K_guess, 1.0 ./ (m_par.μ * m_par.μw), m_par)
    r = interest(K_guess, 1.0 ./ m_par.μ, N, m_par) + 1.0
    w = wage(K_guess, 1.0 ./ m_par.μ, N, m_par)
    Y = output(K_guess, 1.0, N, m_par)
    profits = profitsSS_fnc(Y, m_par.RB, m_par)
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


    incgross, inc, eff_int = incomes(
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

    # initial guess consumption and marginal values (if not set)
    if initial
        c_guess =
            inc[1] .+ inc[2] .* (n_par.mesh_k .* r .> 0) .+ inc[3] .* (n_par.mesh_m .> 0)
        if any(any(c_guess .< 0.0))
            @warn "negative consumption guess"
        end
        Vm = eff_int .* mutil(c_guess, m_par)
        Vk = (r + m_par.λ) .* mutil(c_guess, m_par)
        distr = n_par.distr_guess
    else
        Vm = Vm_guess
        Vk = Vk_guess
        distr = distr_guess
    end

    w_eval_grid = [    (m_par.RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ r .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)) /(m_par.RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar) for i_b in 1:length(n_par.w_sel_m), i_k in 1:length(n_par.w_sel_k), j_k in 1:n_par.nk    ]
    w = NaN*ones(length(n_par.w_sel_m)*length(n_par.w_sel_k))
    for (i_w_b,i_b) in enumerate(n_par.w_sel_m)
        for (i_w_k,i_k) in enumerate(n_par.w_sel_k)
            w[i_w_b + length(n_par.w_sel_m)*(i_w_k-1)] = m_par.RB .* n_par.grid_m[i_b] .+ r .* n_par.grid_k[i_k] .+ m_par.Rbar .* n_par.grid_m[i_b] .* (n_par.grid_m[i_b] .< 0)
        end
    end
    sortingw = sortperm(w)
    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    KS = Ksupply(m_par.RB, r, n_par, m_par, Vm, Vk, distr, distr_y, inc, eff_int, w_eval_grid,sortingw,w[sortingw])
    K = KS[1]                                                     # capital
    Vm = KS[end-2]                                                 # marginal value of liquid assets
    Vk = KS[end-1]                                                 # marginal value of illiquid assets
    distr = KS[end]                                                   # stationary distribution  
    diff = K - K_guess                                               # excess supply of funds
    return diff, Vm, Vk, distr
end
