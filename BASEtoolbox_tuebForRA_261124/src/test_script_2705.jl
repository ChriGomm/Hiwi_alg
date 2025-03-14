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
CDF_guess = ss_full_young.n_par.CDF_guess;
# CDF_guess = ss_full_young.CDFSS;
K_guess = ss_full_young.KSS;


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

        Vm = Vm_guess
        Vk = Vk_guess
        # CDF = CDF_guess
        # CDF = n_par.CDF_guess


RB_guess = m_par.RB
RB = RB_guess
R_guess = r
RK = R_guess

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



# DirectTransition_Splines! --------------------

cdf_prime_on_grid = cdf_initial

for i = 1:10

    cdf_initial = copy(cdf_prime_on_grid)
    cdf_prime_on_grid = similar(cdf_initial)

n = size(cdf_initial)

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


    # DirectTransition_Splines_adjusters_test!(
    DirectTransition_Splines_adjusters!(
        cdf_prime_on_grid_a,
        m_a_star,
        k_a_star,
        cdf_initial,
        IX,
        n_par,
    )

    cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Adjusters

    cdf_prime_on_grid_n = similar(cdf_prime_on_grid)


    DirectTransition_Splines_non_adjusters!(
        cdf_prime_on_grid_n,
        m_n_star,
        cdf_initial_m,
        n_par,
    )



    cdf_prime_on_grid_n .= cumsum(cdf_prime_on_grid_n, dims=2) * (1.0 - m_par.λ) # Non-adjusters

    cdf_prime_on_grid .= cdf_prime_on_grid_a .+ cdf_prime_on_grid_n

    # 2. Build expectation of cdf over income states
    n = size(cdf_prime_on_grid)
    cdf_prime_on_grid .= reshape(reshape(cdf_prime_on_grid, (n[1] .* n[2], n[3])) * n_par.Π, (n[1], n[2], n[3]))

    println("iteration: ", i)
    println("sum pdf y: ", sum(cdf_prime_on_grid[end, end, :   ]))
end

# ------------
    KSS, VmSS, VkSS, CDFSS, n_par, m_par = BASEforHANK.find_steadystate_splines(
        m_par, 
        K_guess = ss_full_young.KSS, 
        Vm_guess = ss_full_young.VmSS, 
        Vk_guess = ss_full_young.VkSS, 
        CDF_guess = ss_full_young.CDFSS)

# ----------------------------

cdf_initial = copy(CDF_guess)

cdf_prime_on_grid = cdf_initial

for i = 1:10

    cdf_initial = copy(cdf_prime_on_grid)
    cdf_prime_on_grid .= similar(cdf_initial)

    BASEforHANK.SteadyState.DirectTransition_Splines2!(
        cdf_prime_on_grid,
        m_n_star,
        m_a_star,
        k_a_star,
        cdf_initial,
        n_par.Π,
        RB,
        RK,
        n_par,
        m_par,
    )

    println("iteration: ", i)
    println("sum pdf y: ", sum(cdf_prime_on_grid[end, end, :   ]))
end




# ---------------------------

using Plots
p = surface(legend = true)

for i_k = 1:3

    m_a_prime_given_y_k_i = m_a_star[:, i_k, 1]
    k_a_prime_given_y_k_i = k_a_star[:, i_k, 1]
    cdf_prime_given_y_k_i = cdf_initial[:, i_k, 1]

    surface!(p, m_a_prime_given_y_k_i, k_a_prime_given_y_k_i, cdf_prime_given_y_k_i, label="k=$i_k")
end

display(p)

# ------------------------

cdf_initial = copy(CDF_guess)

# cdf_prime_on_grid = cdf_initial

cdf_prime_on_mgrid_kprime_a = similar(cdf_initial)

BASEforHANK.SteadyState.DirectTransition_Splines_adjusters_mgrid!(
    cdf_prime_on_mgrid_kprime_a,
    m_a_star,
    cdf_initial,
    n_par,
)

cdf_prime_on_m_k_a = similar(cdf_prime_on_grid)

BASEforHANK.SteadyState.DirectTransition_Splines_adjusters_kgrid!(
    cdf_prime_on_m_k_a,
    m_a_star,
    cdf_prime_on_mgrid_kprime_a,
    n_par,
)




# ---------------------------
# test the other way around

cdf_prime_on_kgrid_mprime_a = similar(cdf_initial)

BASEforHANK.SteadyState.DirectTransition_Splines_adjusters_kgrid!(
    cdf_prime_on_kgrid_mprime_a,
    m_a_star,
    cdf_initial,
    n_par,
)

cdf_prime_on_m_k_a = similar(cdf_prime_on_grid)

BASEforHANK.SteadyState.DirectTransition_Splines_adjusters_mgrid!(
    cdf_prime_on_m_k_a,
    m_a_star,
    cdf_prime_on_kgrid_mprime_a,
    n_par,
)



cdf_prime_on_m_k_a_backup = copy(cdf_prime_on_m_k_a)




# --------------------------

# create a vector of total wealth based on the grids of m and k
total_wealth_unsorted = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
total_wealth_unsorted2 = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
nm_index_mapping = Array{Int}(undef, n_par.nk .* n_par.nm)
nk_index_mapping = Array{Int}(undef, n_par.nk .* n_par.nm)
for k = 1:n_par.nk
    for m = 1:n_par.nm
        total_wealth_unsorted2[m+(k-1)*n_par.nm] = n_par.grid_m[m] .+ n_par.grid_k[k]
        total_wealth_unsorted[m+(k-1)*n_par.nm] = RB .* n_par.grid_m[m] .+ RK .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
        nm_index_mapping[m+(k-1)*n_par.nm] = m
        nk_index_mapping[m+(k-1)*n_par.nm] = k
    end
end

IX = sortperm(total_wealth_unsorted)
IX_2 = sortperm(total_wealth_unsorted2)

sum(IX - IX_2)

# check monoticity
total_wealth_sorted = total_wealth_unsorted[IX]
@assert all(diff(total_wealth_sorted) .>= 0)

m_a_star_sorted = m_a_star[IX]
@assert all(diff(m_a_star_sorted) .>= 0)
k_a_star_sorted = k_a_star[IX]
@assert all(diff(k_a_star_sorted) .>= 0)
m_n_star_sorted = m_n_star[IX]
@assert all(diff(m_n_star_sorted) .>= 0)

w_a_star = m_a_star .+ k_a_star 
w_a_star_sorted = w_a_star[IX]
@assert all(diff(w_a_star_sorted) .>= 0)

mesh_k = repeat(n_par.grid_k, 1, n_par.nm, n_par.ny)
w_n_star = m_n_star .+ mesh_k
w_n_star_sorted = w_n_star[IX]
@assert all(diff(w_n_star_sorted) .>= 0)


# --------
i_y = 1

cdf_prime_given_y = cdf_initial[:, :, i_y]
pdf_prime_given_y = BASEforHANK.Tools.cdf_to_pdf(cdf_prime_given_y)[:]
cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

@assert all(diff(cdf_prime_totalwealth_given_y) .>= 0)


nm_index_mapping_sorted = nm_index_mapping[IX]
nk_index_mapping_sorted = nk_index_mapping[IX]
cdf_prime_2_grids =  Array{eltype(cdf_initial)}(undef, n_par.nm, n_par.nk)
cdf_prime_m =  Array{eltype(cdf_initial)}(undef, n_par.nm)
cdf_prime_k =  Array{eltype(cdf_initial)}(undef, n_par.nk)
for i_w in 1:length(IX)
    # i_k = nk_index_mapping_sorted[i_w]
    # i_m = nm_index_mapping_sorted[i_w]
    i_k = nk_index_mapping[i_w]
    i_m = nm_index_mapping[i_w]
    # cdf_prime_2_grids[i_m, i_k] = IX[i_w]
    cdf_prime_2_grids[i_m, i_k] = cdf_prime_totalwealth_given_y[i_w]
    if i_k == n_par.nk
        cdf_prime_m[i_m] = cdf_prime_totalwealth_given_y[i_w]
    end
    if i_m == n_par.nm
        cdf_prime_k[i_k] = cdf_prime_totalwealth_given_y[i_w]
    end
end

@assert all(diff(cdf_prime_2_grids, dims=1) .>= 0)
@assert all(diff(cdf_prime_2_grids, dims=2) .>= 0)

CDF_guess[:,:,1]

cdf_prime_2_grids

# total_wealth_policy_unsorted = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
# for k = 1:n_par.nk
#     for m = 1:n_par.nm
#         total_wealth_policy_unsorted[m+(k-1)*n_par.nm] = m_a_star[m] .+ k_a_star[k]
#     end
# end
# IX_policy = sortperm(total_wealth_policy_unsorted)



# ------------------ # Tranition on total wealth x illiquid wealth

cdf_prime = zeros(eltype(CDF_guess), size(CDF_guess))
cdf_initial = copy(CDF_guess)

# grid for total wealth: w = m + k
# -> there is not one fixed grid for wealth as total and wealth and illiquid wealth are connected
# grid_w = n_par.grid_m .+ n_par.grid_k 

# mesh for total wealth w = m + k
mesh_w = repeat(n_par.grid_k', n_par.nm, 1) .+ repeat(n_par.grid_m, 1, n_par.nk)
# mesh_w2 = zeros(n_par.nm, n_par.nk)
# for i_k = 1:n_par.nk
#     mesh_w2[:, i_k] .= n_par.grid_m .+ n_par.grid_k[i_k]
# end

# create a vector of total wealth based on the grids of m and k
total_wealth_unsorted = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
total_wealth_unsorted2 = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
nm_index_mapping = Array{Int}(undef, n_par.nk .* n_par.nm)
nk_index_mapping = Array{Int}(undef, n_par.nk .* n_par.nm)
for k = 1:n_par.nk
    for m = 1:n_par.nm
        total_wealth_unsorted[m+(k-1)*n_par.nm] = n_par.grid_m[m] .+ n_par.grid_k[k]
        # total_wealth_unsorted[m+(k-1)*n_par.nm] = RB .* n_par.grid_m[m] .+ RK .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
        nm_index_mapping[m+(k-1)*n_par.nm] = m
        nk_index_mapping[m+(k-1)*n_par.nm] = k
    end
end

sum(total_wealth_unsorted .- mesh_w[:])


# IX = sortperm(total_wealth_unsorted)

IX = sortperm(mesh_w[:])
mesh_w_sorted = mesh_w[IX]
@assert all(diff(mesh_w_sorted) .>= 0)



# ---------
# example initial distribution

# cdf_initial = ones(n_par.nm * n_par.nk, n_par.nk, n_par.ny) ./ (n_par.nm * n_par.nk * n_par.nk * n_par.ny)
# cdf_over_total_wealth = cumsum(cdf_initial, dims=1)
# cdf_over_total_wealth_and_illiquid = cumsum(cdf_over_total_wealth, dims=2)

# -------

w_n_prime = m_n_prime .+ n_par.mesh_k
w_a_prime = m_a_star .+ k_a_star
m_a_prime = m_a_star
k_a_prime = k_a_star

i_y = 1

w_a_prime_given_y =  m_a_prime[:, :, i_y].+ k_a_prime[:, :, i_y]

using Plots
x  = n_par.grid_m
y = n_par.grid_k
surface(x, y, m_a_prime[:, :, i_y])
surface(x, y, k_a_prime[:, :, i_y])
surface(x, y, m_a_prime[:, :, i_y].+ k_a_prime[:, :, i_y])

x = m_a_prime[:, :, i_y]
y = k_a_prime[:, :, i_y]
surface(x, y, CDF_guess[:, :, i_y])

IX_helper = sortperm(w_a_prime_given_y[:])
w_a_prime_given_y = w_a_prime_given_y[IX_helper]

@assert all(diff(w_a_prime_given_y) .>= 0)
# findall(diff(w_a_prime_given_y) .< 0)

# ----------

idx_last_at_constraint_w = findlast(w_a_prime_given_y .== n_par.grid_m[1] .+ n_par.grid_k[1])
idx_last_at_constraint_w = isnothing(idx_last_at_constraint_w) ? 1 : idx_last_at_constraint_w

w_to_cdf_spline = Interpolator(w_a_prime_given_y[idx_last_at_constraint_w:end], cdf_over_total_wealth[idx_last_at_constraint_w:end])



function DirectTransition_Splines_liquid_assets!(
    cdf_prime_on_grid::AbstractArray,
    m_prime::AbstractArray, 
    cdf_initial_on_grid::AbstractArray,
    n_par::NumericalParameters,
)   

    for i_y = 1:n_par.ny
        for i_k = 1:n_par.nk

            cdf_prime_given_y_k = view(cdf_initial_on_grid, :, i_k, i_y)

            # get monotonically increasing policy functions for total wealth in case of adjustment
            m_prime_given_y_k = view(m_prime, :, i_k, i_y)

            # get index after which policy functions are strictly monotonically increasing
            idx_last_at_constraint = findlast(m_prime_given_y_k .== n_par.grid_m[1])
            idx_last_at_constraint = isnothing(idx_last_at_constraint) ? 1 : idx_last_at_constraint

            # REVIEW - Is this required here?
            # # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
            # m_at_max_cdf = m_n_prime_given_y_k[end]
            # idx_last_increasing_cdf = findlast(diff(cdf_prime_given_y_k) .> eps())
            # if idx_last_increasing_cdf !== nothing
            #     m_at_max_cdf = m_n_prime_given_y_k[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
            # end

            # Start interpolation from last unique value (= last value at the constraint)
            m_to_cdf_spline = Interpolator(m_prime_given_y_k[idx_last_at_constraint:end], cdf_prime_given_y_k[idx_last_at_constraint:end])

            # Extrapolation for values below and above observed w_primes
            function m_to_cdf_spline_extr!(cdf_extr::AbstractVector, m::Vector{Float64})
                # indexes for values below lowest observed decision
                idx1 = findlast(m .< m_prime_given_y_k[1])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                idx2 = findfirst(m .> m_prime_given_y_k[end])
                idx2 = isnothing(idx2) ? length(m) + 1 : idx2
                # inter- and extrapolation
                # (if idx1 == 0 or idx2 > end -> 0-element view, i.e. nothing is changed)
                cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision 
                cdf_extr[idx2:end] .= 1.0 * cdf_prime_given_y_k[end] # max mass above highest observed decision
                cdf_extr[idx1+1:idx2-1] .= m_to_cdf_spline.(m[idx1+1:idx2-1])
            end

            # evaluate cdf on grid
            m_to_cdf_spline_extr!(view(cdf_prime_on_grid, :, i_y), n_par.grid_m)
        end
    end
end

function DirectTransition_Splines_illiquid_assets!(
    cdf_prime_on_grid::AbstractArray,
    k_prime::AbstractArray, 
    cdf_initial_on_grid::AbstractArray,
    n_par::NumericalParameters,
)   

    for i_y = 1:n_par.ny
        for i_m = 1:n_par.nm

            cdf_prime_given_y_m = view(cdf_initial_on_grid, i_k, :, i_y)

            # get monotonically increasing policy functions for total wealth in case of adjustment
            k_prime_given_y_m = view(k_prime, i_k, :, i_y)

            # get index after which policy functions are strictly monotonically increasing
            idx_last_at_constraint = findlast(k_prime_given_y_m .== n_par.grid_k[1])
            idx_last_at_constraint = isnothing(idx_last_at_constraint) ? 1 : idx_last_at_constraint

            # REVIEW - Is this required here?
            # # Find cdf_prime_given_y where maximum cdf is reached to ensure strict monotonicity
            # m_at_max_cdf = m_n_prime_given_y_k[end]
            # idx_last_increasing_cdf = findlast(diff(cdf_prime_given_y_k) .> eps())
            # if idx_last_increasing_cdf !== nothing
            #     m_at_max_cdf = m_n_prime_given_y_k[idx_last_increasing_cdf+1] # idx+1 as diff function reduces dimension by 1
            # end

            # Start interpolation from last unique value (= last value at the constraint)
            k_to_cdf_spline = Interpolator(k_prime_given_y_m[idx_last_at_constraint:end], cdf_prime_given_y_m[idx_last_at_constraint:end])

            # Extrapolation for values below and above observed w_primes
            function k_to_cdf_spline_extr!(cdf_extr::AbstractVector, k::Vector{Float64})
                # indexes for values below lowest observed decision
                idx1 = findlast(k .< k_prime_given_y_m[1])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                idx2 = findfirst(k .> k_prime_given_y_m[end])
                idx2 = isnothing(idx2) ? length(k) + 1 : idx2
                # inter- and extrapolation
                # (if idx1 == 0 or idx2 > end -> 0-element view, i.e. nothing is changed)
                cdf_extr[1:idx1] .= 0.0   # no mass below lowest observed decision 
                cdf_extr[idx2:end] .= 1.0 * cdf_prime_given_y_m[end] # max mass above highest observed decision
                cdf_extr[idx1+1:idx2-1] .= k_to_cdf_spline.(k[idx1+1:idx2-1])
            end

            # evaluate cdf on grid
            k_to_cdf_spline_extr!(view(cdf_prime_on_grid, :, i_y), n_par.grid_m)
        end
    end
end

function DirectTransition_Splines2!(
    cdf_prime_on_grid::AbstractArray,   # Defined as cdf over liquid wealth x illiquid wealth x income
    m_n_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    m_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    k_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    cdf_initial_on_grid::AbstractArray,     # Defined as cdf over liquid wealth x illiquid wealth x income
    Π::AbstractArray,
    RB::Real,
    RK::Real,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)

    cdf_prime_on_grid_m = similar(cdf_prime_on_grid)
    cdf_prime_on_grid_m_a = similar(cdf_prime_on_grid)

    DirectTransition_Splines_liquid_assets!(
        cdf_prime_on_grid_a,
        m_a_prime, 
        cdf_initial_on_grid,
        n_par,
    )

    cdf_prime_on_grid_m_a .= cdf_prime_on_grid_m_a * m_par.λ # Adjusters

    cdf_prime_on_grid_m_n = similar(cdf_prime_on_grid)

    DirectTransition_Splines_liquid_assets!(
        cdf_prime_on_grid_m_a,
        m_n_prime, 
        cdf_initial_on_grid,
        n_par,
    )

    cdf_prime_on_grid_m_n .= cdf_prime_on_grid_m_n * (1.0 - m_par.λ) # Non-adjusters

    cdf_prime_on_grid_m .= cdf_prime_on_grid_m_a .+ cdf_prime_on_grid_m_n

    cdf_prime_on_grid_k = similar(cdf_prime_on_grid)

    DirectTransition_Splines_illiquid_assets!(
        cdf_prime_on_grid_k,
        k_a_prime, 
        cdf_initial_on_grid,
        n_par,
    )

    cdf_prime_on_grid_k .= cdf_prime_on_grid_k .* m_par.λ .+ cdf_initial_on_grid .* (1.0 - m_par.λ)

    # Back out cdf values for k 

    # Weight


    # transition along illiquid assets

end