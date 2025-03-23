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
function DirectTransitionSplines(
    distr::Array,
    m_n_star::Array,
    m_a_star::Array,
    k_a_star::Array,
    Π::Array,
    RB::Real,
    RK::Real,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    dPrime = zeros(eltype(distr), size(distr))
    cdf_prime = zeros(eltype(distr), size(distr))

    cdf_initial = distr
    cdf_initial = cumsum(cumsum(cdf_initial, dims=1), dims=2)

    cdf_prime = DirectTransition_Splines!(cdf_initial, m_n_star, m_a_star, k_a_star, copy(cdf_initial), Π, RB, RK, n_par, m_par)

    dPrime = cdf_to_pdf(cumsum(cdf_prime, dims=3))

    return dPrime
end


function DirectTransition_Splines!(
    cdf_prime_on_grid::AbstractArray,   # Defined as cdf over total wealth x illiquid wealth x income
    m_n_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    m_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    k_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    cdf_initial_on_grid::AbstractArray,     # Defined as cdf over total wealth x illiquid wealth x income
    Π::AbstractArray,
    grid_w::AbstractArray,
    grid_w_sorting::AbstractArray,
    nk_map::AbstractArray,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    cdf_prime_on_grid_a = similar(cdf_prime_on_grid)

    DirectTransition_Splines_adjusters!(
        cdf_prime_on_grid_a,
        m_a_prime, 
        k_a_prime,
        cdf_initial_on_grid,
        grid_w,
        grid_w_sorting,
        n_par,
    )

    cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Adjusters

    cdf_prime_on_grid_n = similar(cdf_prime_on_grid)

    DirectTransition_Splines_non_adjusters!(
        cdf_prime_on_grid_n,
        m_n_prime, 
        cdf_initial_on_grid,
        grid_w,
        grid_w_sorting,
        nk_map,
        1.0,
        n_par,
    )

    cdf_prime_on_grid_n .= cdf_prime_on_grid_n * (1.0 - m_par.λ) # Non-adjusters

    cdf_prime_on_grid .= cdf_prime_on_grid_a .+ cdf_prime_on_grid_n

    n = size(cdf_prime_on_grid)
    cdf_prime_on_grid .= reshape(reshape(cdf_prime_on_grid, (n[1] .* n[2], n[3])) * Π, (n[1], n[2], n[3]))

end

function DirectTransition_Splines_adjusters!(
    cdf_prime_on_grid::AbstractArray,   # Defined as cdf over total wealth x illiquid wealth x income
    m_a_prime::AbstractArray,           # Defined as cdf over liquid wealth x illiquid wealth x income
    k_a_prime::AbstractArray,           # Defined as cdf over liquid wealth x illiquid wealth x income
    cdf_initial_on_grid::AbstractArray, # Defined as cdf over total wealth x illiquid wealth x income
    grid_w::AbstractArray,
    grid_w_sorting::AbstractArray,
    n_par::NumericalParameters,
)   

    for i_y in 1:n_par.ny

        # cdf over total wealth unconditional of illiquid assets
        cdf_prime_totalwealth_given_y = view(cdf_initial_on_grid, :, n_par.nk, i_y)

        # get monotonically increasing policy functions for adjustment scenarios
        m_a_prime_grid_totalwealth_given_y = m_a_prime[:, :, i_y][grid_w_sorting]
        k_a_prime_grid_totalwealth_given_y = k_a_prime[:, :, i_y][grid_w_sorting]
        total_wealth_a_given_y = m_a_prime_grid_totalwealth_given_y .+ k_a_prime_grid_totalwealth_given_y

        # # REVIEW - Is this required here?
        # # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
        # k_at_max_cdf = k_a_prime_grid_totalwealth_given_y[end]
        # idx_last_increasing_cdf = findlast(diff(cdf_prime_totalwealth_given_y) .> eps())
        # if idx_last_increasing_cdf !== nothing
        #     m_at_max_cdf = m_n_prime_given_y_k[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
        # end

        # get index after which policy functions are strictly monotonically increasing
        idx_last_at_constraint_k = findlast(k_a_prime_grid_totalwealth_given_y .== n_par.grid_k[1])
        idx_last_at_constraint_k = isnothing(idx_last_at_constraint_k) ? 1 : idx_last_at_constraint_k
        idx_last_at_constraint_w = findlast(total_wealth_a_given_y .== grid_w[1])
        idx_last_at_constraint_w = isnothing(idx_last_at_constraint_w) ? 1 : idx_last_at_constraint_w

        # Start interpolation from last unique value (= last value at the constraint)
        k_to_w_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:end], total_wealth_a_given_y[idx_last_at_constraint_k:end])
        w_to_cdf_spline = Interpolator(total_wealth_a_given_y[idx_last_at_constraint_w:end], cdf_prime_totalwealth_given_y[idx_last_at_constraint_w:end])

        # Extrapolation for values below and above observed k_primes
        function k_to_w_spline_extr!(w_extr::AbstractVector, k::Vector{Float64})
            # indexes for values below lowest observed decision
            idx1 = findlast(k .< k_a_prime_grid_totalwealth_given_y[1])
            idx1 = isnothing(idx1) ? 0 : idx1
            # index for values above highest observed decision
            idx2 = findfirst(k .> k_a_prime_grid_totalwealth_given_y[end])
            idx2 = isnothing(idx2) ? length(k) + 1 : idx2
            # inter- and extrapolation
            w_extr[1:idx1] .= total_wealth_a_given_y[1] - 1e-6
            w_extr[idx2:end] .= total_wealth_a_given_y[end] + 1e-6
            w_extr[idx1+1:idx2-1] .= k_to_w_spline.(k[idx1+1:idx2-1])
        end

        # interpolate grid of  k to grid of total wealth
        grid_wk = similar(n_par.grid_k)
        k_to_w_spline_extr!(grid_wk, n_par.grid_k)

        # Extrapolation for values below and above observed w_primes
        function w_to_cdf_spline_extr!(cdf_extr::AbstractVector, w::Vector{Float64})
            # indexes for values below lowest observed decision
            idx1 = findlast(w .< total_wealth_a_given_y[1])
            idx1 = isnothing(idx1) ? 0 : idx1
            # index for values above highest observed decision
            idx2 = findfirst(w .> total_wealth_a_given_y[end])
            idx2 = isnothing(idx2) ? length(w) + 1 : idx2
            # inter- and extrapolation
            cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision
            cdf_extr[idx2:end] .= 1.0 * cdf_prime_totalwealth_given_y[end] # max mass above highest observed decision
            cdf_extr[idx1+1:idx2-1] .= w_to_cdf_spline.(w[idx1+1:idx2-1])
        end

        # evaluate cdf on grid for adjusters to get P(w <= x, y <= w)
        for i_k = 1:n_par.nk
            w_to_cdf_spline_extr!(view(cdf_prime_on_grid, :, i_k, i_y), min.(grid_w, grid_wk[i_k]))
        end
        w_to_cdf_spline_extr!(view(cdf_prime_on_grid, length(grid_w), :, i_y), grid_wk)
        w_to_cdf_spline_extr!(view(cdf_prime_on_grid, :, n_par.nk, i_y), grid_w)
        cdf_prime_on_grid[end,end,i_y] = cdf_prime_totalwealth_given_y[end]

    end
end



function DirectTransition_Splines_non_adjusters!(
    cdf_prime_on_grid::AbstractArray,       # Defined as cdf over total wealth x illiquid wealth x income
    m_n_prime::AbstractArray,               # Defined as cdf over liquid wealth x illiquid wealth x income
    cdf_initial_on_grid::AbstractArray,     # Defined as cdf over total wealth x illiquid wealth x income
    grid_w::AbstractArray,
    grid_w_sorting::AbstractArray,
    nk_map::AbstractArray,
    q,
    n_par::NumericalParameters,
)

    for i_y = 1:n_par.ny
        # transforms joint cdf in w,k to cdf in w conditional on k
        cdf_prime_given_y_diffk = [cdf_initial_on_grid[:,1,i_y] diff(cdf_initial_on_grid[:,:,i_y], dims=2)] 
        # NOTE Y: make CDF conditional on k (i.e. no normalization) to be consistent with the adjuster case
        # cdf_prime_given_y_diffk = cdf_prime_given_y_diffk ./ repeat(cdf_prime_given_y_diffk[end,:],outer=(n_par.nk,1)) # normalize conditional cdf
        # NOTE Y: will only be evaluated at nodes (k-grid)
        # cdf_k = Interpolator(
        #     n_par.grid_k,
        #     cdf_initial_on_grid[end,:,i_y]
        # )
        for i_k = 1:n_par.nk

            # print("i_y: ", i_y, " i_k: ", i_k, "\n")
            
            # select off-grid cdf, policy and wealth grid for given y and k
            cdf_prime_given_y_k = view(cdf_prime_given_y_diffk, :, i_k)
            # get cdf values that correspond to wealth grid given i_k
            cdf_prime_given_y_k_selected = cdf_prime_given_y_k[nk_map[grid_w_sorting] .== i_k]  # TODO: use view?
            # cdf today at those points is cdf prime at w_n_prime
            w_n_prime_given_y_k = m_n_prime[:, i_k, i_y] .+ q.*n_par.grid_k[i_k]

            # Find values where the constraint binds
            idx_last_at_constraint = findlast(w_n_prime_given_y_k .== n_par.grid_m[1] .+ q.*n_par.grid_k[i_k])    # REVIEW - Correct? -> q required?
            idx_last_at_constraint = isnothing(idx_last_at_constraint) ? 1 : idx_last_at_constraint

            # REVIEW - Review example file for this problem
            # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
            # w_at_max_cdf = w_n_prime_given_y_k[end]
            # idx_last_increasing_cdf = findlast(diff(cdf_prime_given_y_k) .> eps())
            # if idx_last_increasing_cdf !== nothing
            #     w_at_max_cdf = w_n_prime_given_y_k[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
            # end

            # Start interpolation from last unique value (= last value at the constraint)
            w_to_cdf_spline = Interpolator(
                w_n_prime_given_y_k[idx_last_at_constraint:end],
                cdf_prime_given_y_k_selected[idx_last_at_constraint:end])

             # Extrapolation for values below and above observed m_primes and interpolation as defined above otherwise
             function w_to_cdf_spline_extr!(cdf_values::AbstractVector, w::Vector{Float64})
                idx1 = findlast(w .< w_n_prime_given_y_k[1])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                # idx2 = findfirst(w .> min(w_at_max_cdf, w_n_prime_given_y_k[end]))
                idx2 = findfirst(w .> w_n_prime_given_y_k[end])
                idx2 = isnothing(idx2) ? length(w) + 1 : idx2
                # inter- and extrapolation
                # wealth below lowest observed policy (conditional on k) do not happen (would correspond to w<k_j)
                cdf_values[1:idx1] .= 0.0 
                # wealth beyond the highest observed policy (conditional on k) would imply liquid savings beyond m-grid
                # -> CDF = max(CDF|k_j)
                cdf_values[idx2:end] .= 1.0 * cdf_prime_given_y_k[end]
                cdf_values[idx1+1:idx2-1] .= w_to_cdf_spline.(w[idx1+1:idx2-1])
            end

            # REVIEW - Some lines unncecessary?
            # Evaluate cdf on grid
            cdf_prime_on_grid_given_k_y = view(cdf_prime_on_grid, :, i_k, i_y)
            cdfend = copy(cdf_prime_given_y_k[end])
            w_to_cdf_spline_extr!(cdf_prime_on_grid_given_k_y, grid_w)
            cdf_prime_on_grid_given_k_y .= min.(cdf_prime_on_grid_given_k_y, cdfend)
            cdf_prime_on_grid_given_k_y[end] = cdfend
        end

        # Sum up over k (as k is treated discretely, no proper integration for now)
        
        cdf_prime_on_grid[:,:,i_y] .= cumsum(cdf_prime_on_grid[:, :, i_y], dims=2)

        # integrate over k
        # computation in place (use result and overwrite with end result)

        # I believe this form of integration is flawed:
        # - Taking the derivative in the direction of k for 
        # - it always returns 0 for the first element of the grid by construction
        # - Also: The values for cdf_prime_on_grid are not monotonically increasing (I am not sure if this is problematic, but I remember vaguely that for some integration methods this can be)
        # for (i_w,w) in enumerate(grid_w)
        #     k_condition_spline = Interpolator(
        #         n_par.grid_k,
        #         cdf_prime_on_grid[i_w, :, i_y]
        #     )
        #     k_condition_diff_cdf_spline = Interpolator(
        #         n_par.grid_k,
        #         [ForwardDiff.derivative(k_condition_spline,k)*cdf_initial_on_grid[end,i_k,i_y] for (i_k,k) in enumerate(n_par.grid_k)]
        #         #vcat((diff(cdf_prime_on_grid[i_w, :, i_y]).*cdf_initial_on_grid[end,:,i_y][1:end-1])[:],ForwardDiff.derivative(k_condition_spline,n_par.grid_k[end]))
        #     )
        #     right_part(k) = integrate(k_condition_diff_cdf_spline, n_par.grid_k[1], k)
        #     left_part(k) = k_condition_spline(k) * cdf_k(k) - k_condition_spline(n_par.grid_k[1]) * cdf_k(n_par.grid_k[1])
        #     int_mass(k) = left_part(k) - right_part(k)
        #     cdf_prime_on_grid[i_w, :, i_y] .= int_mass.(n_par.grid_k)
        # end
        # cdf_prime_on_grid[:,:,i_y] .= cdf_prime_on_grid[:,:,i_y] ./ cdf_prime_on_grid[end,end,i_y] .* cdf_initial_on_grid[end,end,i_y] # fix normalization issue due to interpolation inaccuracy
    end
end


