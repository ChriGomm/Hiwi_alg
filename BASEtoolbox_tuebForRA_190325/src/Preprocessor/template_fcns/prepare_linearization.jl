
@doc raw"""
    prepare_linearization(KSS, VmSS, VkSS, distrSS, n_par, m_par)

Compute a number of equilibrium objects needed for linearization.

# Arguments
- `KSS`: steady-state capital stock
- `VmSS`, `VkSS`: marginal value functions
- `distrSS::Array{Float64,3}`: steady-state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`

# Returns
- `XSS::Array{Float64,1}`, `XSSaggr::Array{Float64,1}`: steady state vectors produced by [`@writeXSS()`](@ref)
- `indexes`, `indexes_aggr`: `struct`s for accessing `XSS`,`XSSaggr` by variable names, produced by [`@make_fn()`](@ref),
        [`@make_fnaggr()`](@ref)
- `compressionIndexes::Array{Array{Int,1},1}`: indexes for compressed marginal value functions (``V_m`` and ``V_k``)
- `Copula(x,y,z)`: function that maps marginals `x`,`y`,`z` to approximated joint distribution, produced by
        [`myinterpolate3()`](@ref)
- `n_par::NumericalParameters`,`m_par::ModelParameters`
- `CDFSS`, `CDF_m`, `CDF_k`, `CDF_y`: cumulative distribution functions (joint and marginals)
- `distrSS::Array{Float64,3}`: steady state distribution of idiosyncratic states, computed by [`Ksupply()`](@ref)
"""
function prepare_linearization(KSS, VmSS, VkSS, distrSS, n_par, m_par)
    if n_par.verbose
        println("Running reduction step on steady-state value functions to prepare linearization")
    end
    # ------------------------------------------------------------------------------
    # STEP 1: Evaluate StE to calculate steady state variable values
    # ------------------------------------------------------------------------------
    # Calculate other equilibrium quantities
    Paux      = n_par.Π^1000                                              # Calculate ergodic ince distribution from transitions
    distr_y   = Paux[1, :] 
    NSS       = employment(KSS, 1.0./(m_par.μ*m_par.μw), m_par)
    rSS       = interest(KSS, 1.0./m_par.μ, NSS, m_par) + 1.0 
    wSS       = wage(KSS, 1.0./m_par.μ, NSS, m_par)
    YSS       = output(KSS, 1.0, NSS, m_par)                              # stationary income distribution

    profitsSS = profitsSS_fnc(YSS,m_par.RB,m_par)
    unionprofitsSS  = (1.0 .- 1.0/m_par.μw) .* wSS .* NSS
    LC              = 1.0./m_par.μw *wSS.*NSS  
    taxrev          = ((n_par.grid_y/n_par.H).*LC)-m_par.τlev.*((n_par.grid_y/n_par.H).*LC).^(1.0-m_par.τprog )
    taxrev[end]     =  n_par.grid_y[end].*profitsSS - m_par.τlev.*( n_par.grid_y[end].*profitsSS).^(1.0-m_par.τprog )
    incgrossaux     = ((n_par.grid_y/n_par.H).*LC)
    incgrossaux[end]=  n_par.grid_y[end].*profitsSS
    av_tax_rateSS   = dot(distr_y, taxrev)./(dot(distr_y,incgrossaux))
    incgross, incnet, eff_int = 
            incomes(n_par, m_par, 1.0 ./ m_par.μw, 1.0, 1.0, 
                    m_par.RB, m_par.τprog , m_par.τlev, n_par.H, 1.0, 1.0,rSS,wSS,NSS,profitsSS,unionprofitsSS, av_tax_rateSS)
    w_eval_grid = [    (m_par.RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ rSS .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)) /(m_par.RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar) for i_b in 1:length(n_par.w_sel_m), i_k in 1:length(n_par.w_sel_k), j_k in 1:n_par.nk    ]
    w = NaN*ones(length(n_par.w_sel_m)*length(n_par.w_sel_k))
    for (i_w_b,i_b) in enumerate(n_par.w_sel_m)
        for (i_w_k,i_k) in enumerate(n_par.w_sel_k)
            w[i_w_b + length(n_par.w_sel_m)*(i_w_k-1)] = m_par.RB .* n_par.grid_m[i_b] .+ rSS .* n_par.grid_k[i_k] .+ m_par.Rbar .* n_par.grid_m[i_b] .* (n_par.grid_m[i_b] .< 0)
        end
    end
    sortingw = sortperm(w)
    #----------------------------------------------------------------------------
    # Calculate supply of funds for given prices
    #----------------------------------------------------------------------------
    # obtain other steady state variables
    # KSS, BSS, CDF_mSS, CDF_wSS,
    # c_a_starSS, m_a_starSS, k_a_starSS, c_n_starSS, m_n_starSS, VmSS, VkSS, distrSS =
    return Ksupply(m_par.RB, rSS, n_par, m_par, VmSS, VkSS, distrSS, distr_y, incnet, eff_int, w_eval_grid,sortingw,w[sortingw])

    VmSS      .*= eff_int

    # generate CDFSS (conditional on y, times pdf(y))
    CDFSS = NaN .* ones(n_par.nm, n_par.nk, n_par.ny)

    for i_y in 1:n_par.ny
        cdf_k_intp = SteadyState.Interpolator(n_par.grid_k,distrSS[end,:,i_y]/distrSS[end,end,i_y])
        pdf_k = [SteadyState.ForwardDiff.derivative(cdf_k_intp,k) for k in n_par.grid_k]
        for i_b in 1:n_par.nm
            CDFSS[i_b,1,i_y] = distrSS[i_b,1,i_y]*distrSS[end,1,i_y]/distrSS[end,end,i_y]
            to_int = SteadyState.Interpolator(
                            n_par.grid_k[1:end],
                            [pdf_k[i_k]*distrSS[i_b,i_k,i_y] for i_k = 1:n_par.nk]
                    )
            for i_k in 2:n_par.nk
                CDFSS[i_b,i_k,i_y] = CDFSS[i_b,1,i_y] + SteadyState.integrate(to_int,n_par.grid_k[1],n_par.grid_k[i_k])
            end
        end
        for i_k in 1:n_par.nk
            CDFSS[:,i_k,i_y] = CDFSS[:,i_k,i_y] / CDFSS[end,i_k,i_y] * distrSS[end,end,i_y]
        end
    end
   
    # for i_y in 1:n_par.ny
    #     cdf_k_intp = SteadyState.Interpolator(
    #         n_par.grid_k,
    #         distrSS[end,:,i_y]./distrSS[end,end,i_y]
    #     )
    #     pdf_k(i_k) = SteadyState.ForwardDiff.derivative(cdf_k_intp,n_par.grid_k[i_k])
    #     for i_b in 1:n_par.nm-1
    #         CDFSS[i_b,1,i_y] = distrSS[i_b,1,i_y] .* distrSS[end,1,i_y] ./ (distrSS[end,end,i_y]^2)
    #         to_int = SteadyState.Interpolator(
    #                                 n_par.grid_k,
    #                                 [pdf_k(j_k)*distrSS[i_b,j_k,i_y]/distrSS[end,end,i_y] for j_k in 1:n_par.nk]
    #                                 )
    #         for i_k in 2:n_par.nk-1
    #             CDFSS[i_b,i_k,i_y] = CDFSS[i_b,1,i_y] + SteadyState.integrate(to_int,n_par.grid_k[1],n_par.grid_k[i_k])
    #         end
    #         CDFSS[i_b,end,i_y] = CDF_mSS[i_b,i_y] / distr_y[i_y]
    #     end
    #     CDFSS[end,:,i_y] = distrSS[end,:,i_y] ./ distr_y[i_y]
    #     CDFSS[:,:,i_y] = CDFSS[:,:,i_y] .* distr_y[i_y]
    # end


    # Produce distributional summary statistics
    distr_mSS, distr_kSS, distr_ySS, TOP10WshareSS, TOP10IshareSS,TOP10InetshareSS, GiniWSS, GiniCSS, sdlogySS = 
        distrSummaries(CDFSS, 1.0, c_a_starSS, c_n_starSS, n_par, incnet,incgross, m_par)
    CDF_mSS = sum(CDF_mSS,dims=2)[:]
    println("CDF_m: ",CDF_mSS)
    # ------------------------------------------------------------------------------
    # STEP 2: Dimensionality reduction
    # ------------------------------------------------------------------------------
    # 2 a.) Selection of DCT coefficients for reduction of the marginal value functions
    # ------------------------------------------------------------------------------
    # Vector of all prices in the household problem (in SS)
    # price_vectorSS = [1.0/m_par.μw, 1.0, m_par.RB, m_par.τprog , m_par.τlev, 
    #                  rSS, wSS, profitsSS, unionprofitsSS, av_tax_rateSS, 1.0][:]

    # indk, indm, _ = first_stage_reduction(VkSS, VmSS, TransitionMat_aSS, TransitionMat_nSS, 
    #                                     NSS, m_a_starSS, k_a_starSS, m_n_starSS,
    #                                     price_vectorSS, n_par, m_par)
    indk = select_ind(
        dct(reshape(log.(invmutil(VkSS, m_par)), (n_par.nm, n_par.nk, n_par.ny))),
        n_par.reduc_value,
    )
    indm = select_ind(
        dct(reshape(log.(invmutil(VmSS, m_par)), (n_par.nm, n_par.nk, n_par.ny))),
        n_par.reduc_value,
    )

    VmSS          = log.(invmutil(VmSS,m_par))
    VkSS          = log.(invmutil(VkSS,m_par))
    compressionIndexesVm = sort(unique(vcat(indm...)))
    compressionIndexesVk = sort(unique(vcat(indk...)))

    # ------------------------------------------------------------------------------
    # 2b.) Select polynomials for copula perturbation
    # ------------------------------------------------------------------------------
    SELECT                = [ (!((i==1) & (j ==1)) & !((k==1) & (j ==1)) & !((k==1) & (i ==1))) 
                             for i = 1:n_par.nm_copula, j = 1:n_par.nk_copula, k = 1:n_par.ny_copula]

    compressionIndexesCOP = findall(SELECT[:])     # store indices of selected coeffs 

    # ------------------------------------------------------------------------------
    # 2c.) Store Compression Indexes
    # ------------------------------------------------------------------------------
    compressionIndexes    = Array{Array{Int,1},1}(undef ,3)       # Container to store all retained coefficients in one array
    compressionIndexes[1] = compressionIndexesVm
    compressionIndexes[2] = compressionIndexesVk
    compressionIndexes[3] = compressionIndexesCOP


    # ------------------------------------------------------------------------------
    # 2d.) Produce marginals
    # ------------------------------------------------------------------------------
   
    CDF_y                = cumsum(distr_ySS[:])          # Marginal distribution (cdf) of income
    CDF_kSS = sum(distrSS[end,:,:],dims=2)[:]

    # Calculate interpolation nodes for the copula as those elements of the marginal distribution 
    # that yield close to equal aggregate shares in liquid wealth, illiquid wealth and income.
    # Entrepreneur state treated separately. 
    @set! n_par.copula_marginal_m = copula_marg_equi(distr_mSS, (n_par.grid_m), n_par.nm_copula)
    @set! n_par.copula_marginal_k = copula_marg_equi(distr_kSS,  (n_par.grid_k), n_par.nk_copula)
    @set! n_par.copula_marginal_y = CDF_y #copula_marg_equi_y(distr_ySS, n_par.grid_y, n_par.ny_copula)

    # ------------------------------------------------------------------------------
    # DO NOT DELETE OR EDIT NEXT LINE! This is needed for parser.
    # aggregate steady state marker
    # @include "../3_Model/input_aggregate_steady_state.jl"

    # write to XSS vector
    @writeXSS

    # produce indexes to access XSS etc.
    indexes               = produce_indexes(n_par, compressionIndexesVm, compressionIndexesVk, compressionIndexesCOP)
    indexes_aggr          = produce_indexes_aggr(n_par)

    @set! n_par.ntotal    = length(vcat(compressionIndexes...)) + (n_par.ny + n_par.nm + n_par.nk - 3 + n_par.naggr) 
    @set! n_par.nstates   = n_par.ny + n_par.nk + n_par.nm - 3 + n_par.naggrstates + length(compressionIndexes[3]) # add to no. of states the coefficients that perturb the copula
    @set! n_par.ncontrols = length(vcat(compressionIndexes[1:2]...)) + n_par.naggrcontrols
    @set! n_par.LOMstate_save = zeros(n_par.nstates, n_par.nstates)
    @set! n_par.State2Control_save = zeros(n_par.ncontrols, n_par.nstates)
    @set! n_par.nstates_r   = copy(n_par.nstates)
    @set! n_par.ncontrols_r = copy(n_par.ncontrols)
    @set! n_par.ntotal_r    = copy(n_par.ntotal)
    @set! n_par.PRightStates= Diagonal(ones(n_par.nstates))
    @set! n_par.PRightAll   = Diagonal(ones(n_par.ntotal))

    if n_par.n_agg_eqn != n_par.naggr - length(n_par.distr_names)
        @warn("Inconsistency in number of aggregate variables and equations")
    end

    return XSS, XSSaggr, indexes, indexes, indexes_aggr, compressionIndexes, n_par, #=
            =# m_par, CDFSS, CDF_mSS, CDF_kSS, distr_ySS, distrSS, m_a_starSS, k_a_starSS, m_n_starSS       
end

function copula_marg_equi_y(distr_i, grid_i, nx)

    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 2:nx-1
        equi(x1)            = equishares(x1, x2, grid_i[1:end-1], distr_i[1:end-1], nx-1) 
        x2                  = find_zero(equi, (1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end]   = CDF_i[end]
    aux_marginal[1]     = CDF_i[1]
    aux_marginal[end-1] = CDF_i[end-1]
    copula_marginal     = copy(aux_marginal)
    jlast               = nx-1
    for i = nx-2:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j 
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function copula_marg_equi(distr_i, grid_i, nx)

    CDF_i        = cumsum(distr_i[:])          # Marginal distribution (cdf) of liquid assets
    aux_marginal = collect(range(CDF_i[1], stop = CDF_i[end], length = nx))

    x2 = 1.0
    for i = 1:nx-1
        equi(x1)            = equishares(x1, x2, grid_i, distr_i, nx) 
        x2                  = find_zero(equi ,(1e-9, x2))
        aux_marginal[end-i] = x2
    end

    aux_marginal[end] = CDF_i[end]
    aux_marginal[1]   = CDF_i[1]
    copula_marginal   = copy(aux_marginal)
    jlast             = nx
    for i = nx-1:-1:1
        j = locate(aux_marginal[i], CDF_i) + 1
        if jlast == j 
            j -=1
        end
        jlast = j
        copula_marginal[i] = CDF_i[j]
    end
    return copula_marginal
end

function equishares(x1, x2, grid_i, distr_i, nx) 

    FN_Wshares = cumsum(grid_i .* distr_i) ./ sum(grid_i .* distr_i)
    Wshares    = diff(mylinearinterpolate(cumsum(distr_i), FN_Wshares, [x1; x2]))
    dev_equi   = Wshares .- 1.0 ./ nx

    return dev_equi
end
