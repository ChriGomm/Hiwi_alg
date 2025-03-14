function DirectTransition(
    m_a_star::Array,
    m_n_star::Array,
    k_a_star::Array,
    distr::Array,
    λ,
    Π::Array,
    n_par::NumericalParameters,
)

    dPrime = zeros(eltype(distr), size(distr))
    DirectTransition!(dPrime, m_a_star, m_n_star, k_a_star, distr, λ, Π, n_par)
    return dPrime
end
function DirectTransition!(
    dPrime,
    m_a_star::Array,
    m_n_star::Array,
    k_a_star::Array,
    distr::Array,
    λ,
    Π::Array,
    n_par::NumericalParameters,
)

    idk_a, wR_k_a = MakeWeightsLight(k_a_star, n_par.grid_k)
    idm_a, wR_m_a = MakeWeightsLight(m_a_star, n_par.grid_m)
    idm_n, wR_m_n = MakeWeightsLight(m_n_star, n_par.grid_m)
    blockindex = (0:n_par.ny-1) * n_par.nk * n_par.nm
    @inbounds begin
        for zz = 1:n_par.ny # all current income states
            for kk = 1:n_par.nk # all current illiquid asset states
                #idk_n = kk
                for mm = 1:n_par.nm
                    dd = distr[mm, kk, zz]
                    IDD_a = (idm_a[mm, kk, zz] .+ (idk_a[mm, kk, zz] .- 1) .* n_par.nm)
                    IDD_n = (idm_n[mm, kk, zz] .+ (kk - 1) .* n_par.nm)
                    # liquid assets of non adjusters
                    w = wR_m_n[mm, kk, zz]
                    DL_n = (1.0 .- λ) .* (dd .* (1.0 .- w))
                    DR_n = (1.0 .- λ) .* (dd .* w)
                    # illiquid assets of adjusters
                    w = wR_k_a[mm, kk, zz]
                    dl = λ .* (dd .* (1.0 .- w))
                    dr = λ .* (dd .* w)
                    # liquid assets of adjusters
                    w = wR_m_a[mm, kk, zz]
                    DLL_a = (dl .* (1.0 .- w))
                    DLR_a = (dl .* w)
                    DRL_a = (dr .* (1.0 .- w))
                    DRR_a = (dr .* w)
                    for yy = 1:n_par.ny # add income transitions
                        pp = Π[zz, yy]
                        id_a = IDD_a .+ blockindex[yy]
                        id_n = IDD_n .+ blockindex[yy]
                        dPrime[id_a] += pp .* DLL_a
                        dPrime[id_a+1] += pp .* DLR_a
                        dPrime[id_a+n_par.nm] += pp .* DRL_a
                        dPrime[id_a+n_par.nm+1] += pp .* DRR_a
                        dPrime[id_n] += pp .* DL_n
                        dPrime[id_n+1] += pp .* DR_n
                    end
                end
            end
        end
    end

end

@doc raw"""
    DirectTransition_Splines(dPrime::Array, m_star::Array, distr::Array, Π::Array, n_par::NumericalParameters)

Iterates the distribution one period forward.

# Arguments
- `m_star::Array`: optimal savings function
- `Π::Array`: transition matrix
- `distr::Array`: distribution in period t

# Returns
- `dPrime::Array`: distribution in period t+1


"""
function DirectTransition_Splines(
    m_n_star::Array,
    m_a_star::Array,
    k_a_star::Array,
    distr::Array,
    Π::Array,
    RB::Real,
    RK::Real,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    dPrime = zeros(eltype(distr), size(distr))
    cdf_prime = zeros(eltype(distr), size(distr))


    cdf_initial = sr_full.distrSS
    cdf_initial_m = cumsum(cdf_initial, dims=1)
    cdf_initial = cumsum(cumsum(cdf_initial, dims=1), dims=2)

    cdf_prime = DirectTransition_Splines!(cdf_initial, m_n_star, m_a_star, k_a_star, copy(cdf_initial), cdf_initial_m, Π, RB, RK, n_par, m_par)

    # Back out discrete pdf
    # for i_y in 1:n_par.ny
    #     dPrime[:, i_y] .= cdf_to_pdf(cdf_prime[:, i_y])
    # end
    dPrime = cdf_to_pdf(cumsum(distr_splines, dims=3))

    return dPrime
end

@doc raw"""
    DirectTransition_Splines!(
        m_prime_grid::AbstractArray,
        cdf_initial::AbstractArray,
        n_par::NumericalParameters,
    )

    Direct transition of the savings cdf from one period to the next. 
        Transition is done using monotonic spline interpolation to bring the next periods cdf's 
        back to the reference grid.
        Logic: Given assets in t (on-grid) and an the income shock realization, the decision
        of next periods assets is deterministic and thus the probability mass move from the 
        on grid to the off-grid values. Using the spline interpolation the cdf is evaluated at
        the fix asset grid.

    # Arguments
    - `cdf_prime_on_grid::AbstractArray`: Next periods CDF on fixed asset grid.
    - `m_prime_grid::AbstractArray`: Savings function defined on the fixed asset and income grid.
    - `cdf_initial::AbstractArray`: CDF over fixed assets grid for each income realization.
    - `Π::Array`: Stochastic transition matrix.
    - `n_par::NumericalParameters`
    - `m_par::ModelParameters`
"""
function DirectTransition_Splines!(
    cdf_prime_on_grid::AbstractArray,
    m_n_prime_grid::AbstractArray,
    m_a_prime_grid::AbstractArray,
    k_a_prime_grid::AbstractArray,
    cdf_initial::AbstractArray,
    Π::AbstractArray,
    RB::Real,
    RK::Real,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)


    n = size(cdf_initial)

    # cdf_initial .= reshape(reshape(cdf_initial, (n[1] .* n[2], n[3])) * Π, (n[1], n[2], n[3]))

    t = collect(size(cdf_initial))
    t[2] = 1
    aux = zeros(eltype(cdf_initial), Tuple(t))
    cdf_initial_m = diff(cat([aux, cdf_initial]...; dims=2); dims=2)


    total_wealth = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
    for k = 1:n_par.nk
        for m = 1:n_par.nm
            total_wealth[m+(k-1)*n_par.nm] = RB .* n_par.grid_m[m] .+ RK .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
        end
    end

    IX = sortperm(total_wealth)
    total_wealth = total_wealth[IX]

    cdf_prime_on_grid_a = similar(cdf_prime_on_grid)

    for i_y = 1:n_par.ny

        cdf_prime_given_y = cdf_initial[:, :, i_y]

        pdf_prime_given_y = cdf_to_pdf(cdf_prime_given_y)[:]

        cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

        m_a_prime_grid_totalwealth_given_y = m_a_prime_grid[:, :, i_y][IX]
        k_a_prime_grid_totalwealth_given_y = k_a_prime_grid[:, :, i_y][IX]

        idx_last_at_constraint_k = findlast(k_a_prime_grid_totalwealth_given_y .== n_par.grid_k[1])
        # idx_last_on_grid_k = min.(findlast(k_a_prime_grid_totalwealth_given_y .< n_par.grid_k[end]) .+ 1, n_par.nk * n_par.nm)
        idx_last_on_grid_k =  n_par.nm * n_par.nk

        idx_last_at_constraint_m = findlast(m_a_prime_grid_totalwealth_given_y .== n_par.grid_m[1])
        # idx_last_on_grid_m = min.(findlast(m_a_prime_grid_totalwealth_given_y .< n_par.grid_m[end]) .+ 1, n_par.nm * n_par.nk)
        idx_last_on_grid_m =  n_par.nm * n_par.nk

        # Start interpolation from last unique value (= last value at the constraint)
        if isnothing(idx_last_at_constraint_k)
            k_to_cdf_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k], cdf_prime_totalwealth_given_y[1:idx_last_on_grid_k])
            k_to_m_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k], m_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k])

        else
            k_to_cdf_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k],
                cdf_prime_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k])
            k_to_m_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k],
                m_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k])
        end
        if isnothing(idx_last_at_constraint_m)
            m_to_cdf_spline = Interpolator(m_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_m], cdf_prime_totalwealth_given_y[1:idx_last_on_grid_m])
            m_to_k_spline = Interpolator(m_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_m], k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_m])
            # elseif idx_last_at_constraint_m == idx_last_on_grid_m
            #     m_to_k_spline(m) = k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_m]
            #     m_to_cdf_spline(m) = cdf_prime_totalwealth_given_y[idx_last_at_constraint_m]

        else
            idx_last = idx_last_at_constraint_m
            m_to_k_spline = Interpolator(m_a_prime_grid_totalwealth_given_y[idx_last:idx_last_on_grid_m],
                k_a_prime_grid_totalwealth_given_y[idx_last:idx_last_on_grid_m])
            m_to_cdf_spline = Interpolator(m_a_prime_grid_totalwealth_given_y[idx_last:idx_last_on_grid_m],
                cdf_prime_totalwealth_given_y[idx_last:idx_last_on_grid_m])

        end

        # Extrapolation for values below and above observed m_primes
        function m_to_k_spline_extr(m)
            if m < m_a_prime_grid_totalwealth_given_y[1]
                k_extr = n_par.grid_k[1]
            elseif m > m_a_prime_grid_totalwealth_given_y[end]
                k_extr = n_par.grid_k[end]
            else
                k_extr = m_to_k_spline(m)
            end
            return k_extr
        end

        # Extrapolation for values below and above observed m_primes
        function k_to_m_spline_extr(k)
            if k < k_a_prime_grid_totalwealth_given_y[1]
                m_extr = n_par.grid_m[1]
            elseif k > k_a_prime_grid_totalwealth_given_y[end]
                m_extr = n_par.grid_m[end]
            else
                m_extr = k_to_m_spline(k)
            end
            return m_extr
        end
        # Extrapolation for values below and above observed k_primes
        function k_to_cdf_spline_extr(k)
            if k < k_a_prime_grid_totalwealth_given_y[1]
                cdf_value = 0.0
            elseif k > k_a_prime_grid_totalwealth_given_y[end]
                cdf_value = 1.0 * cdf_prime_totalwealth_given_y[end]
            else
                cdf_value = k_to_cdf_spline(k)
            end
            return cdf_value
        end

        # Extrapolation for values below and above observed k_primes
        function m_to_cdf_spline_extr(m)
            if m < m_a_prime_grid_totalwealth_given_y[1]
                cdf_value = 0.0
            elseif m > m_a_prime_grid_totalwealth_given_y[end]
                cdf_value = 1.0 * cdf_prime_totalwealth_given_y[end]
            else
                cdf_value = m_to_cdf_spline(m)
            end
            return cdf_value
        end


        for i_k = 1:n_par.nk
            for i_m = 1:n_par.nm
                if n_par.grid_m[i_m] > m_a_prime_grid_totalwealth_given_y[end]
                    if n_par.grid_k[i_k] > k_a_prime_grid_totalwealth_given_y[end]
                        cdf_prime_on_grid_a[i_m, i_k, i_y] = 1.0 * cdf_prime_totalwealth_given_y[end]
                    else
                        cdf_prime_on_grid_a[i_m, i_k, i_y] = k_to_cdf_spline_extr.(n_par.grid_k[i_k])
                    end
                elseif n_par.grid_m[i_m] < m_a_prime_grid_totalwealth_given_y[1] || n_par.grid_k[i_k] < k_a_prime_grid_totalwealth_given_y[1]
                    cdf_prime_on_grid_a[i_m, i_k, i_y] = 0.0
                else
                    if n_par.grid_k[i_k] >= m_to_k_spline(n_par.grid_m[i_m])
                        cdf_prime_on_grid_a[i_m, i_k, i_y] = m_to_cdf_spline_extr.(n_par.grid_m[i_m])
                    else
                        cdf_prime_on_grid_a[i_m, i_k, i_y] = k_to_cdf_spline_extr.(n_par.grid_k[i_k])
                    end
                end
            end
        end

   
        cdf_prime_on_grid_a[end, end, i_y] = cdf_prime_given_y[end, end]
        

    end

    cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Adjusters

    cdf_prime_on_grid_n = similar(cdf_prime_on_grid)


    # 1. Map cdf back to fixed asset grid.
    for i_k = 1:n_par.nk
        for i_y = 1:n_par.ny

            # m_prime_given_y = min.(m_prime_grid[:, i_y], n_par.grid_m[end]) # Cap at maximum gridpoint
            m_prime_given_y = m_n_prime_grid[:, i_k, i_y] # Cap at maximum gridpoint
            cdf_prime_given_y = cdf_initial_m[:, i_k, i_y]

            # 2a. Specify mapping from assets to cdf with monotonic PCIHP interpolation

            # Find values where the constraint binds:
            # We can only interpolate using the last value at the constraint, because 
            # m_prime is not unique otherwise.
            idx_last_at_constraint = findlast(m_prime_given_y .== n_par.grid_m[1])
            idx_last_on_grid = min.(findlast(m_prime_given_y .< n_par.grid_m[end]) .+ 1, n_par.nm)

            # Start interpolation from last unique value (= last value at the constraint)
            if isnothing(idx_last_at_constraint)
                m_to_cdf_spline = Interpolator(m_prime_given_y[1:idx_last_on_grid], cdf_prime_given_y[1:idx_last_on_grid])
            else
                m_to_cdf_spline = Interpolator(m_prime_given_y[idx_last_at_constraint:idx_last_on_grid],
                    cdf_prime_given_y[idx_last_at_constraint:idx_last_on_grid])
            end

            # Extrapolation for values below and above observed m_primes
            function m_to_cdf_spline_extr(m)
                if m < m_prime_given_y[1]
                    cdf_value = 0.0
                elseif m > m_prime_given_y[end]
                    cdf_value = 1.0 * cdf_prime_given_y[end]
                else
                    cdf_value = m_to_cdf_spline(m)
                end
                return cdf_value
            end

            # 2b. Evaluate cdf at fixed grid
            cdf_prime_on_grid_n[:, i_k, i_y] = m_to_cdf_spline_extr.(n_par.grid_m)
            # Cap cdf at maximum gridpoint
            cdf_prime_on_grid_n[end, i_k, i_y] = cdf_prime_given_y[end]
        end
    end

    # 2. Build expectation of cdf over income states
    cdf_prime_on_grid_n .= cumsum(cdf_prime_on_grid_n, dims=2) * (1.0 - m_par.λ) # Non-adjusters

    # cdf_prime_on_grid .= cdf_prime_on_grid_a ./ m_par.λ # Adjusters

    cdf_prime_on_grid .= cdf_prime_on_grid_a .+ cdf_prime_on_grid_n

    n = size(cdf_prime_on_grid)

    cdf_prime_on_grid .= reshape(reshape(cdf_prime_on_grid, (n[1] .* n[2], n[3])) * Π, (n[1], n[2], n[3]))

end