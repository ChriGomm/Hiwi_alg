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

path_to_transition = "SubModules/SteadyState/IM_fcns/fcn_directtransition_total_wealth_illiquid.jl"
include(path_to_transition)


n = (n_par.nk, n_par.nm, n_par.ny)

    # add an additional dimension of zeros at second dimension  (dimension of aux: t[1] x t[2]+1 x t[3], with aux[:,1,:]=0) 
    t = collect(n)
    t[2] = 1
    aux = zeros(Tuple(t))

    # create a vector of total wealth based on the grids of m and k

    # mesh_w = repeat(n_par.grid_k', n_par.nm, 1) .+ repeat(n_par.grid_m, 1, n_par.nk)
    # total_wealth_unsorted = mesh_w[:]
    # total_wealth_sorting = sortperm(total_wealth_unsorted)
    # total_wealth_sorted = total_wealth_unsorted[total_wealth_sorting]

    total_wealth_unsorted = Array{eltype(cdf_initial)}(undef, n_par.nk .* n_par.nm)
    nm_map = Array{eltype(Int)}(undef, n_par.nk .* n_par.nm)
    nk_map = Array{eltype(Int)}(undef, n_par.nk .* n_par.nm)
    for k = 1:n_par.nk
        for m = 1:n_par.nm
            total_wealth_unsorted[m+(k-1)*n_par.nm] = RB .* n_par.grid_m[m] .+ RK .* n_par.grid_k[k] .+ m_par.Rbar .* n_par.grid_m[m] .* (n_par.grid_m[m] .< 0)
            nm_map[m+(k-1)*n_par.nm] = m
            nk_map[m+(k-1)*n_par.nm] = k
        end
    end
    total_wealth_sorting = sortperm(total_wealth_unsorted)
    total_wealth_sorted = total_wealth_unsorted[total_wealth_sorting]

    @assert all(diff(total_wealth_sorted) .>= 0)
    for i_y = 1:n_par.ny
        @assert all(diff(m_a_star[:, :, i_y][IX]) .>= 0)
        @assert all(diff(k_a_star[:, :, i_y][IX]) .>= 0)
        for i_k = 1:n_par.nk
            @assert all(diff(m_n_star[:, i_k, i_y] .+ n_par.grid_k[i_k]) .>= 0)
        end
    end


    pdf_guess = ones(n_par.nm * n_par.nk,  n_par.nk, n_par.ny) / sum(n_par.nm * n_par.nk * n_par.nk * n_par.ny)
    cdf_guess = cumsum(cumsum(pdf_guess, dims=1), dims=2)

    @assert all(diff(cdf_guess, dims=2) .>= 0)
    @assert all(diff(diff(cdf_guess, dims=2), dims=1) .>= 0)


cdf_prime = zeros(size(cdf_guess))
cdf_initial = copy(cdf_guess)


cdf_prime_on_grid_a = similar(cdf_initial)

DirectTransition_Splines_adjusters!(
    cdf_prime_on_grid_a,
    m_a_star, 
    k_a_star,
    cdf_initial,
    total_wealth_sorted,
    total_wealth_sorting,
    n_par,
)

cdf_prime_on_grid_a

cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Ad

cdf_prime_on_grid_n = similar(cdf_initial);

include(path_to_transition)

DirectTransition_Splines_non_adjusters!(
    cdf_prime_on_grid_n,
    m_n_star, 
    cdf_initial,
    total_wealth_sorted,
    total_wealth_sorting,
    nk_map,
    n_par,
)


# m_n_star[:, 127, i_y] .+ n_par.grid_k[127]
# findlast(m_n_star[:, 127, i_y] .+ n_par.grid_k[127] .== n_par.grid_m[1] + n_par.grid_k[127])

# ------------------------------------

cdf_prime = zeros(size(cdf_guess))
cdf_initial = copy(cdf_guess)


# Tolerance for change in cdf from period to period
tol = n_par.ϵ
# Maximum iterations to find steady state distribution
max_iter = 10000
# Init 
distance = 9999.0
counts = 0

# Iterate on distribution until convergence
while distance > tol && counts < max_iter
    counts = counts + 1
    cdf_old = copy(cdf_initial)

    DirectTransition_Splines!(
        cdf_initial,
        m_n_star,
        m_a_star,
        k_a_star,
        copy(cdf_initial),
        n_par.Π,
        total_wealth_sorted,
        total_wealth_sorting,
        nk_map,
        n_par,
        m_par
        )

    difference = cdf_old .- cdf_initial
    distance = maximum(abs, difference)

end

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

# for i in 1:10
#     println("i = $i")

#     DirectTransition_Splines!(
#         cdf_prime,
#         m_n_star,
#         m_a_star,
#         k_a_star,
#         cdf_initial,
#         n_par.Π,
#         total_wealth_sorted,
#         total_wealth_sorting,
#         nk_map,
#         n_par,
#         m_par
#         )

#     cdf_initial = copy(cdf_prime)

# end

