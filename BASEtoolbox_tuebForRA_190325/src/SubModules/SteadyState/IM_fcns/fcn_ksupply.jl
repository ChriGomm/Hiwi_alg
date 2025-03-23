@doc raw"""
    Ksupply(RB_guess,R_guess,w_guess,profit_guess,n_par,m_par)

Calculate the aggregate savings when households face idiosyncratic income risk.

Idiosyncratic state is tuple ``(m,k,y)``, where
``m``: liquid assets, ``k``: illiquid assets, ``y``: labor income

# Arguments
- `R_guess`: real interest rate illiquid assets
- `RB_guess`: nominal rate on liquid assets
- `w_guess`: wages
- `profit_guess`: profits
- `n_par::NumericalParameters`
- `m_par::ModelParameters`

# Returns
- `K`,`B`: aggregate saving in illiquid (`K`) and liquid (`B`) assets
-  `TransitionMat`,`TransitionMat_a`,`TransitionMat_n`: `sparse` transition matrices
    (average, with [`a`] or without [`n`] adjustment of illiquid asset)
- `distr`: ergodic steady state of `TransitionMat`
- `c_a_star`,`m_a_star`,`k_a_star`,`c_n_star`,`m_n_star`: optimal policies for
    consumption [`c`], liquid [`m`] and illiquid [`k`] asset, with [`a`] or
    without [`n`] adjustment of illiquid asset
- `V_m`,`V_k`: marginal value functions
"""
function Ksupply(
    RB_guess::Float64,
    R_guess::Float64,
    n_par,
    m_par,
    Vm::AbstractArray,
    Vk::AbstractArray,
    distr_guess::AbstractArray,
    distr_y,
    inc::AbstractArray,
    eff_int::AbstractArray,
    w_eval_grid::AbstractArray,
    sortingw::AbstractArray,
    wgrid::AbstractArray
)


    #----------------------------------------------------------------------------
    # Iterate over consumption policies
    #----------------------------------------------------------------------------

    c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, m_a_aux, w_bar, aux_c = find_ss_policies(
        Vm,
        Vk,
        inc,
        eff_int,
        n_par,
        m_par,
        RB_guess,
        R_guess,
    )

    i_l, w_r = SteadyState.MakeWeightsLight(m_a_aux[1,:],n_par.grid_m) 
    w_k = ones(n_par.ny)
    for i_y in 1:n_par.ny
        if isempty(w_bar[i_y])
            w_k[i_y] = NaN
        else
            w_k[i_y] = w_bar[i_y][end] + w_r[i_y]*(n_par.grid_m[i_l[i_y]+1]+aux_c[i_l[i_y]+1,1]-(n_par.grid_m[i_l[i_y]]+aux_c[i_l[i_y],1]))
        end
    end
    w_m = [isempty(w_bar[i_y]) ? NaN : w_bar[i_y][1] for i_y in 1:n_par.ny]

    #------------------------------------------------------
    # Find stationary distribution 
    #------------------------------------------------------

        distr, K, B, cdf_b, cdf_w = if n_par.method_for_ss_distr == "splines"
            find_ss_distribution_splines(distr_guess, m_n_star, m_a_star, k_a_star, distr_y, RB_guess, R_guess,w_eval_grid,sortingw,wgrid,m_a_aux, w_k, w_m, n_par, m_par)
        else
            find_ss_distribution_young(distr_guess, m_n_star, m_a_star, k_a_star, n_par, m_par)
        end
    

    return K,
    B,
    cdf_b,
    cdf_w,
    c_a_star,
    m_a_star,
    k_a_star,
    c_n_star,
    m_n_star,
    Vm,
    Vk,
    distr
end

# REVIEW - needed?
# function next_dist(mu, Q)
#     @unpack m, n, colptr, rowval, nzval = Q
#     nextmu = similar(mu)
#     @inbounds for col = 1:m
#         nextmu[col] = 0.0
#         for n_row = colptr[col]:colptr[col+1]-1
#             nextmu[col] += mu[rowval[n_row]] * nzval[n_row]
#         end
#     end
#     return nextmu
# end


function find_ss_policies(
    Vm::AbstractArray,
    Vk::AbstractArray,
    inc::AbstractArray,
    eff_int::AbstractArray,
    n_par::NumericalParameters,
    m_par::ModelParameters,
    RB_guess::Float64,
    R_guess::Float64,
)
    #   initialize distance variables
    dist = 9999.0
    dist1 = dist
    dist2 = dist

    q = 1.0       # price of Capital
    
    count = 0
    n = size(Vm)
    # containers for policies, marginal value functions etc.
    m_n_star = similar(Vm)
    m_a_star = similar(Vm)
    k_a_star = similar(Vm)
    c_a_star = similar(Vm)
    c_n_star = similar(Vm)
    EVm = similar(Vm)
    EVk = similar(Vk)
    Vm_new = similar(Vm)
    Vk_new = similar(Vk)
    iVm = invmutil(Vm, m_par)
    iVk = invmutil(Vk, m_par)
    iVm_new = similar(iVm)
    iVk_new = similar(iVk)
    E_return_diff = similar(EVm)
    EMU = similar(EVm)
    c_star_n = similar(EVm)
    m_star_n = similar(EVm)
    mutil_c_a = similar(EVm)
    D1 = similar(EVm)
    D2 = similar(EVm)
    Resource_grid = reshape(inc[2] .+ inc[3] .+ inc[4], (n[1] .* n[2], n[3]))

    m_a_aux = NaN*ones(eltype(Vm),n_par.nk,n_par.ny)
    w_bar = Array{Array{eltype(Vm)}}(undef, n_par.ny, 1)
    aux_c = Array{eltype(Vm)}(undef,n_par.nm,n_par.ny)
    # iterate over consumption policies until convergence
    while dist > n_par.ϵ && count < 10000 # Iterate consumption policies until converegence
        count = count + 1
        # Take expectations for labor income change
        #EVk  .= reshape(reshape(Vk, (n[1] .* n[2], n[3])) * n_par.Π', (n[1], n[2], n[3]))
        BLAS.gemm!(
            'N',
            'T',
            1.0,
            reshape(Vk, (n[1] .* n[2], n[3])),
            n_par.Π,
            0.0,
            reshape(EVk, (n[1] .* n[2], n[3])),
        )
        EVk .= reshape(EVk, (n[1], n[2], n[3]))
        BLAS.gemm!(
            'N',
            'T',
            1.0,
            reshape(Vm, (n[1] .* n[2], n[3])),
            n_par.Π,
            0.0,
            reshape(EVm, (n[1] .* n[2], n[3])),
        )
        EVm .= reshape(EVm, (n[1], n[2], n[3]))
        EVm .*= eff_int

        # Policy update step
        m_a_aux, w_bar, aux_c = EGM_policyupdate!(
            c_a_star,
            m_a_star,
            k_a_star,
            c_n_star,
            m_n_star,
            E_return_diff,
            EMU,
            c_star_n,
            m_star_n,
            Resource_grid,
            EVm,
            EVk,
            q,
            m_par.π,
            RB_guess,
            1.0,
            inc,
            n_par,
            m_par,
            false,
        )

        # marginal value update step
        updateV!(
            Vk_new,
            Vm_new,
            mutil_c_a,
            EVk,
            c_a_star,
            c_n_star,
            m_n_star,
            R_guess - 1.0,
            q,
            m_par,
            n_par,
        )
        invmutil!(iVk_new, Vk_new, m_par)
        invmutil!(iVm_new, Vm_new, m_par)
        # Calculate distance in updates
        D1 .= iVk_new .- iVk
        D2 .= iVm_new .- iVm
        dist1 = maximum(abs, D1)
        dist2 = maximum(abs, D2)
        dist = max(dist1, dist2) # distance of old and new policy

        # update policy guess/marginal values of liquid/illiquid assets
        Vm .= Vm_new
        Vk .= Vk_new
        iVk .= iVk_new
        iVm .= iVm_new
    end
    println("EGM Iterations: ", count)
    println("EGM Dist: ", dist)

    return c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, m_a_aux, w_bar, aux_c
end


function expected_value(cdf::AbstractArray, grid)

    cdf_splines = Interpolator(grid, cdf)

    right_part = integrate(cdf_splines, grid[1], grid[end])
    left_part = grid[end] * cdf_splines(grid[end]) - grid[1] * cdf_splines(grid[1])

    EV = left_part - right_part

    return EV
end


function find_ss_distribution_splines(
    distr_guess::AbstractArray,
    m_n_star::AbstractArray,
    m_a_star::AbstractArray,
    k_a_star::AbstractArray,
    distr_y,
    RB_guess::Float64,
    R_guess::Float64,
    w_eval_grid::AbstractArray,
    sortingw::AbstractArray,
    wgrid::AbstractArray,
    m_a_aux::AbstractArray,
    w_k::AbstractArray,
    w_m::AbstractArray,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    # total_wealth_unsorted = Array{eltype(CDF_guess)}(undef, n_par.nk .* n_par.nm)
    # nm_map = Array{eltype(Int)}(undef, n_par.nk .* n_par.nm)
    # nk_map = Array{eltype(Int)}(undef, n_par.nk .* n_par.nm)
    # for k = 1:n_par.nk
    #     for m = 1:n_par.nm
    #         total_wealth_unsorted[m+(k-1)*n_par.nm] = RB_guess .* n_par.grid_m[m] .+ R_guess .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
    #         nm_map[m+(k-1)*n_par.nm] = m
    #         nk_map[m+(k-1)*n_par.nm] = k
    #     end
    # end
    # total_wealth_sorting = sortperm(total_wealth_unsorted)
    # total_wealth_sorted = total_wealth_unsorted[total_wealth_sorting]

    # Initialize distribution
    distr_initial = copy(distr_guess)
    cdf_w = NaN*ones(eltype(distr_guess),length(n_par.w_sel_k)*length(n_par.w_sel_m), n_par.ny)

    # Tolerance for change in cdf from period to period
    tol = n_par.ϵ
    # Maximum iterations to find steady state distribution
    max_iter = 2000
    # Init 
    distance = 9999.0
    count = 0
    # Iterate on distribution until convergence
    while distance > tol && count < max_iter
        count = count + 1
        distr_old = copy(distr_initial)

        cdf_w = DirectTransition_Splines!(
            distr_initial, 
            m_n_star, 
            m_a_star, 
            k_a_star, 
            copy(distr_initial), 
            n_par.Π, 
            distr_y,
            RB_guess,
            R_guess,
            w_eval_grid,
            sortingw,
            wgrid,
            m_a_aux,
            w_k,
            w_m,
            n_par, 
            m_par;
            speedup = true)

        difference = distr_old .- distr_initial
        distance = maximum(abs, difference)

    end

    println("Distribution Iterations: ", count)
    println("Distribution Dist: ", distance)

    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    cdf_b = NaN*ones(n_par.nm,n_par.ny)
    for i_y in 1:n_par.ny
        diffcdfk = diff(distr_initial[end,:,i_y],dims=1)/distr_initial[end,end,i_y]
        for i_b = 1:n_par.nm
            for i_k = 1:n_par.nk
                cdf_b[i_b,i_y] = distr_initial[end,1,i_y]/distr_initial[end,end,i_y]*distr_initial[i_b,1,i_y] + .5*sum((distr_initial[i_b,2:end,i_y] .+ distr_initial[i_b,1:end-1,i_y]).*diffcdfk)
            end
        end
    end


    K = expected_value(sum(distr_initial[end,:,:],dims=2)[:],n_par.grid_k)
    B = expected_value(sum(cdf_b,dims=2)[:],n_par.grid_m)

    println("K: ", K)
    println("B: ", B)

    return distr_initial, K, B, cdf_b, cdf_w
end


function find_ss_distribution_young(
    CDF_guess::AbstractArray,
    m_n_star::AbstractArray,
    m_a_star::AbstractArray,
    k_a_star::AbstractArray,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    # Get initial pdf guess
    distr_guess = cdf_to_pdf(cumsum(CDF_guess, dims=3))

    # Define transition matrix
    S_a, T_a, W_a, S_n, T_n, W_n =
    MakeTransition(m_a_star, m_n_star, k_a_star, n_par.Π, n_par)
    TransitionMat_a = sparse(
        S_a,
        T_a,
        W_a,
        n_par.nm * n_par.nk * n_par.ny,
        n_par.nm * n_par.nk * n_par.ny,
    )
    TransitionMat_n = sparse(
        S_n,
        T_n,
        W_n,
        n_par.nm * n_par.nk * n_par.ny,
        n_par.nm * n_par.nk * n_par.ny,
    )
    Γ = m_par.λ .* TransitionMat_a .+ (1.0 .- m_par.λ) .* TransitionMat_n

    # Calculate left-hand unit eigenvector

    aux = real.(eigsolve(Γ', distr_guess[:], 1)[2][1])

    ## Exploit that the Eigenvector of eigenvalue 1 is the nullspace of TransitionMat' -I
    #     Q_T = LinearMap((dmu, mu) -> dist_change!(dmu, mu, Γ), n_par.nm * n_par.nk * n_par.ny, ismutating = true)
    #     aux = fill(1/(n_par.nm * n_par.nk * n_par.ny), n_par.nm * n_par.nk * n_par.ny)#distr_guess[:] # can't use 0 as initial guess
    #     gmres!(aux, Q_T, zeros(n_par.nm * n_par.nk * n_par.ny))  # i.e., solve x'(Γ-I) = 0 iteratively
    ##qr algorithm for nullspace finding
    #     aux2 = qr(Γ - I)
    #     aux = Array{Float64}(undef, n_par.nm * n_par.nk * n_par.ny)
    #     aux[aux2.prow] = aux2.Q[:,end]
    #
    distr = (reshape((aux[:]) ./ sum((aux[:])), (n_par.nm, n_par.nk, n_par.ny)))

    # distr_splines = find_ss_distribution_splines(m_a_star, m_n_star, k_a_star, distr, n_par)

    #-----------------------------------------------------------------------------
    # Calculate capital stock
    #-----------------------------------------------------------------------------
    # TODO: implement integration
    K = dot(sum(distr, dims=(1, 3)), n_par.grid_k)
    B = dot(sum(distr, dims=(2, 3)), n_par.grid_m)

    println("K: ", K)
    println("B: ", B)

    CDF = cumsum(cumsum(distr, dims=1), dims=2)
    return CDF, K, B
end

