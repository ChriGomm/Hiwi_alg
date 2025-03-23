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

    # add an additional dimension of zeros at second dimension  (dimension of aux: t[1] x t[2]+1 x t[3], with aux[:,1,:]=0) 
    t = collect(size(cdf_initial))
    t[2] = 1
    aux = zeros(eltype(cdf_initial), Tuple(t))
    # get cdf of m conditional on k and y
    cdf_initial_m = diff(cat([aux, cdf_initial]...; dims=2); dims=2)

    # create a vector of total wealth based on the grids of m and k
    total_wealth_unsorted = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
    for k = 1:n_par.nk
        for m = 1:n_par.nm
            total_wealth_unsorted[m+(k-1)*n_par.nm] = RB .* n_par.grid_m[m] .+ RK .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
        end
    end

    IX = sortperm(total_wealth_unsorted)
    # total_wealth_sorted = total_wealth_unsorted[IX]
    cdf_prime_on_grid_a = similar(cdf_prime_on_grid)

    DirectTransition_Splines_adjusters!(
        cdf_prime_on_grid_a,
        m_a_prime_grid,
        k_a_prime_grid,
        cdf_initial,
        IX,
        n_par,
    )

    cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Adjusters

    cdf_prime_on_grid_n = similar(cdf_prime_on_grid)

    DirectTransition_Splines_non_adjusters!(
        cdf_prime_on_grid_n,
        m_n_prime_grid,
        cdf_initial_m,
        n_par,
    )

    
    cdf_prime_on_grid_n .= cumsum(cdf_prime_on_grid_n, dims=2) * (1.0 - m_par.λ) # Non-adjusters

    cdf_prime_on_grid .= cdf_prime_on_grid_a .+ cdf_prime_on_grid_n

    # 2. Build expectation of cdf over income states
    n = size(cdf_prime_on_grid)
    cdf_prime_on_grid .= reshape(reshape(cdf_prime_on_grid, (n[1] .* n[2], n[3])) * Π, (n[1], n[2], n[3]))

end

function DirectTransition_Splines_adjusters!(
    cdf_prime_on_grid_a::AbstractArray,
    m_a_prime_grid::AbstractArray,
    k_a_prime_grid::AbstractArray,
    cdf_initial::AbstractArray,
    IX::AbstractArray,
    n_par::NumericalParameters,
)

    for i_y = 1:n_par.ny
        # REVIEW - problematic lines
        cdf_prime_given_y = cdf_initial[:, :, i_y]
        pdf_prime_given_y = cdf_to_pdf(cdf_prime_given_y)[:]
        cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

        # get monotonically increasing policy functions for adjustment scenarios
        m_a_prime_grid_totalwealth_given_y = m_a_prime_grid[:, :, i_y][IX]
        k_a_prime_grid_totalwealth_given_y = k_a_prime_grid[:, :, i_y][IX]
        total_wealth_a_given_y = m_a_prime_grid_totalwealth_given_y .+ k_a_prime_grid_totalwealth_given_y

        # get index after which policy functions are strictly monotonically increasing
        idx_last_at_constraint_k = findlast(k_a_prime_grid_totalwealth_given_y .== n_par.grid_k[1])
        idx_last_at_constraint_k = isnothing(idx_last_at_constraint_k) ? 1 : idx_last_at_constraint_k
        idx_last_at_constraint_m = findlast(m_a_prime_grid_totalwealth_given_y .== n_par.grid_m[1])
        idx_last_at_constraint_m = isnothing(idx_last_at_constraint_m) ? 1 : idx_last_at_constraint_m
        idx_last_at_constraint_w = findlast(total_wealth_a_given_y .== n_par.grid_m[1] .+ n_par.grid_k[1])
        idx_last_at_constraint_w = isnothing(idx_last_at_constraint_w) ? 1 : idx_last_at_constraint_w
            
        # Start interpolation from last unique value (= last value at the constraint)
        k_to_w_spline = Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:end], total_wealth_a_given_y[idx_last_at_constraint_k:end])
        m_to_w_spline = Interpolator(m_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_m:end], total_wealth_a_given_y[idx_last_at_constraint_m:end])
        w_to_cdf_spline = Interpolator(total_wealth_a_given_y[idx_last_at_constraint_w:end], cdf_prime_totalwealth_given_y[idx_last_at_constraint_w:end])

                
        # Extrapolation for values below and above observed m_primes
        function m_to_w_spline_extr!(w_extr::AbstractVector, m::Vector{Float64})
            # indexes for values below lowest observed decision 
            idx1 = findlast(m .< m_a_prime_grid_totalwealth_given_y[1])
            idx1 = isnothing(idx1) ? 0 : idx1
            # index for values above highest observed decision
            idx2 = findfirst(m .> m_a_prime_grid_totalwealth_given_y[end])
            idx2 = isnothing(idx2) ? length(m) + 1 : idx2
            # inter- and extrapolation
            w_extr[1:idx1] .= total_wealth_a_given_y[1] - 1e-6
            w_extr[idx2:end] .= total_wealth_a_given_y[end] + 1e-6
            w_extr[idx1+1:idx2-1] .= m_to_w_spline.(m[idx1+1:idx2-1])
        end
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

        # interpolate grid of m and k to grid of total wealth prime
        grid_wm = similar(n_par.grid_m)
        m_to_w_spline_extr!(grid_wm, n_par.grid_m)
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

        # evaluate cdf on grid for adjusters

        for i_k = 1:n_par.nk
            w_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, :, i_k, i_y), min.(grid_wm, grid_wk[i_k]))
        end
        w_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, n_par.nm, :, i_y), grid_wk)
        w_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, :, n_par.nk, i_y), grid_wm)

    end

end


# function DirectTransition_Splines_adjusters_test!(
#     cdf_prime_on_grid_a::AbstractArray,
#     m_a_prime_grid::AbstractArray,
#     k_a_prime_grid::AbstractArray,
#     cdf_initial::AbstractArray,
#     IX::AbstractArray,
#     n_par::NumericalParameters,
# )

#     for i_y = 1:n_par.ny
#         # REVIEW - problematic lines
#         cdf_prime_given_y = cdf_initial[:, :, i_y]
#         pdf_prime_given_y = cdf_to_pdf(cdf_prime_given_y)[:]
#         cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

#         # get monotonically increasing policy functions for adjustment scenarios
#         m_a_prime_grid_totalwealth_given_y = m_a_prime_grid[:, :, i_y][IX]
#         k_a_prime_grid_totalwealth_given_y = k_a_prime_grid[:, :, i_y][IX]
#         total_wealth_a_given_y = m_a_prime_grid_totalwealth_given_y .+ k_a_prime_grid_totalwealth_given_y

#         # get index after which policy functions are strictly monotonically increasing
#         idx_last_at_constraint_w = findlast(total_wealth_a_given_y .== n_par.grid_m[1] .+ n_par.grid_k[1])
#         idx_last_at_constraint_w = isnothing(idx_last_at_constraint_w) ? 1 : idx_last_at_constraint_w
            
#         # Start interpolation from last unique value (= last value at the constraint)
#         w_to_cdf_spline = Interpolator(total_wealth_a_given_y[idx_last_at_constraint_w:end], cdf_prime_totalwealth_given_y[idx_last_at_constraint_w:end])
  
#         # Extrapolation for values below and above observed w_primes
#         function w_to_cdf_spline_extr!(cdf_extr::AbstractVector, w::Vector{Float64})
#             # indexes for values below lowest observed decision
#             idx1 = findlast(w .< total_wealth_a_given_y[1])
#             idx1 = isnothing(idx1) ? 0 : idx1
#             # index for values above highest observed decision
#             idx2 = findfirst(w .> total_wealth_a_given_y[end])
#             idx2 = isnothing(idx2) ? length(w) + 1 : idx2
#             # inter- and extrapolation
#             cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision
#             cdf_extr[idx2:end] .= 1.0 * cdf_prime_totalwealth_given_y[end] # max mass above highest observed decision
#             cdf_extr[idx1+1:idx2-1] .= w_to_cdf_spline.(w[idx1+1:idx2-1])
#         end

#         # evaluate cdf on grid for adjusters
#         cdfend = copy(cdf_prime_totalwealth_given_y[end])
#         for i_k = 1:n_par.nk
#             w_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, :, i_k, i_y), n_par.grid_m .+ n_par.grid_k[i_k])
#         end
#         cdf_prime_on_grid_a[n_par.nm, n_par.nk, i_y] = cdfend
#     end

# end


function DirectTransition_Splines_adjusters_mgrid!(
    cdf_prime_on_grid_a::AbstractArray,
    m_a_prime::AbstractArray,
    cdf_initial::AbstractArray,
    n_par::NumericalParameters,
)

    for i_y = 1:n_par.ny
        for i_k = 1:n_par.nk

            # get conditional policies
            m_a_prime_given_y_k = m_a_prime[:, i_k, i_y]
            # for these endogenous policies, we know the next periods (off-grid) distribution
            cdf_mprime_given_y_kprime = cdf_initial[:, i_k, i_y]

            # interpolate cdf back on liquid asset grid

            # get index after which policy functions are strictly monotonically increasing
            idx_last_at_constraint_m = findlast(m_a_prime_given_y_k .== n_par.grid_m[1])
            idx_last_at_constraint_m = isnothing(idx_last_at_constraint_m) ? 1 : idx_last_at_constraint_m

            # Start interpolation from last unique value (= last value at the constraint)
            m_to_cdf_spline = Interpolator(
                m_a_prime_given_y_k[idx_last_at_constraint_m:end], 
                cdf_mprime_given_y_kprime[idx_last_at_constraint_m:end])
  
            # Extrapolation for values below and above observed w_primes
            function m_to_cdf_spline_extr!(cdf_extr::AbstractVector, m::Vector{Float64})
                # indexes for values below lowest observed decision
                idx1 = findlast(m .< m_a_prime_given_y_k[1])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                idx2 = findfirst(m .> m_a_prime_given_y_k[end])
                idx2 = isnothing(idx2) ? length(m) + 1 : idx2
                # inter- and extrapolation
                cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision
                cdf_extr[idx2:end] .= 1.0 * cdf_mprime_given_y_kprime[end] # max mass above highest observed decision
                cdf_extr[idx1+1:idx2-1] .= m_to_cdf_spline.(m[idx1+1:idx2-1])
            end

            # evaluate cdf on liquid asset grid for adjusters
            # the resulting cdf is defined on the wealth gird and on kprime
            m_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, :, i_k, i_y), n_par.grid_m)
            cdf_prime_on_grid_a[n_par.nm, n_par.nk, i_y] = copy(cdf_mprime_given_y_kprime[end])

        end
    end

    # Now we have the cdf defined on the liquid asset grid and illiquid asset endogenous policy grid
    # To bring it back to the illiquid asset grid, we can do the transition using the given cdf and the illiquid asset policy



end


function DirectTransition_Splines_adjusters_kgrid!(
    cdf_prime_on_grid_a::AbstractArray,
    k_a_prime::AbstractArray,
    cdf_initial::AbstractArray,
    n_par::NumericalParameters,
)

    for i_y = 1:n_par.ny
        for i_m = 1:n_par.nm

            # get conditional policies
            k_a_prime_given_y_m = k_a_prime[i_m, :, i_y]
            # for these endogenous policies, we know the next periods (off-grid) distribution
            cdf_kprime_given_y_m = cdf_initial[i_m, :, i_y]

            # interpolate cdf back on illiquid asset grid

            # get index after which policy functions are strictly monotonically increasing
            idx_last_at_constraint_k = findlast(k_a_prime_given_y_m .== n_par.grid_k[1])
            idx_last_at_constraint_k = isnothing(idx_last_at_constraint_k) ? 1 : idx_last_at_constraint_k

            # Start interpolation from last unique value (= last value at the constraint)
            k_to_cdf_spline = Interpolator(
                k_a_prime_given_y_m[idx_last_at_constraint_k:end], 
                cdf_kprime_given_y_m[idx_last_at_constraint_k:end])
  
            # Extrapolation for values below and above observed w_primes
            function k_to_cdf_spline_extr!(cdf_extr::AbstractVector, k::Vector{Float64})
                # indexes for values below lowest observed decision
                idx1 = findlast(k .< k_a_prime_given_y_m[1])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                idx2 = findfirst(k .> k_a_prime_given_y_m[end])
                idx2 = isnothing(idx2) ? length(k) + 1 : idx2
                # inter- and extrapolation
                cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision
                cdf_extr[idx2:end] .= 1.0 * cdf_kprime_given_y_m[end] # max mass above highest observed decision
                cdf_extr[idx1+1:idx2-1] .= k_to_cdf_spline.(k[idx1+1:idx2-1])
            end

            # evaluate cdf on liquid asset grid for adjusters
            # the resulting cdf is defined on the wealth gird and on kprime
            k_to_cdf_spline_extr!(view(cdf_prime_on_grid_a, i_m, :, i_y), n_par.grid_k)
            cdf_prime_on_grid_a[n_par.nm, n_par.nk, i_y] = copy(cdf_kprime_given_y_m[end])

        end
    end
end



function DirectTransition_Splines_non_adjusters!(
    cdf_prime_on_grid_n::AbstractArray,
    m_n_prime_grid::AbstractArray,
    cdf_initial_m::AbstractArray,
    n_par::NumericalParameters,
)

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
            idx_last_at_constraint = isnothing(idx_last_at_constraint) ? 1 : idx_last_at_constraint
            
            # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
            m_at_max_cdf = m_prime_given_y[end]
            idx_last_increasing_cdf = findlast(diff(cdf_prime_given_y) .> eps())
            if idx_last_increasing_cdf !== nothing
                m_at_max_cdf = m_prime_given_y[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
            end

            # Start interpolation from last unique value (= last value at the constraint)
            m_to_cdf_spline = Interpolator(
                m_prime_given_y[idx_last_at_constraint:end],
                cdf_prime_given_y[idx_last_at_constraint:end])

            # Extrapolation for values below and above observed m_primes and interpolation as defined above otherwise
            function m_to_cdf_spline_extr!(cdf_values::AbstractVector, m::Vector{Float64})
                idx1 = findlast(m .< m_prime_given_y[1])
                if idx1 !== nothing
                    cdf_values[1:idx1] .= 0.0
                else
                    idx1 = 0
                end
                idx2 = findfirst(m .> min(m_at_max_cdf, m_prime_given_y[end]))
                if idx2 !== nothing
                    cdf_values[idx2:end] .= 1.0 * cdf_prime_given_y[end]
                else
                    idx2 = length(m) + 1
                end
                cdf_values[idx1+1:idx2-1] .= m_to_cdf_spline.(m[idx1+1:idx2-1])
            end

            # 2b. Evaluate cdf at fixed grid
            cdfend = copy(cdf_prime_given_y[end])
            cdf_prime_on_grid_n_given_k_y = view(cdf_prime_on_grid_n, :, i_k, i_y)
            m_to_cdf_spline_extr!(cdf_prime_on_grid_n_given_k_y, n_par.grid_m)
            cdf_prime_on_grid_n_given_k_y .= min.(cdf_prime_on_grid_n_given_k_y, cdfend)
            # Cap cdf at maximum gridpoint
            cdf_prime_on_grid_n_given_k_y[end] = cdfend
        end
    end
end