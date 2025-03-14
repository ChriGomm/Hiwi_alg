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
using BenchmarkTools, Revise, LinearAlgebra, PCHIPInterpolation, ForwardDiff, Plots
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
KSS, VmSS, VkSS, CDFSS, n_par, m_par = BASEforHANK.find_steadystate_splines(m_par)
ss_full_splines = BASEforHANK.SteadyStateStruct(KSS, VmSS, VkSS, CDFSS, n_par)
jldsave("Output/Saves/steadystate_splines.jld2", true; ss_full_splines) 

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

K_guess = 35.0

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

    # Vm = Vm_guess
    # Vk = Vk_guess
    # # CDF = CDF_guess
    # # CDF = n_par.CDF_guess

    c_guess =
        inc[1] .+ inc[2] .* (n_par.mesh_k .* r .> 0) .+ inc[3] .* (n_par.mesh_m .> 0)
    if any(any(c_guess .< 0.0))
        @warn "negative consumption guess"
    end
    Vm = eff_int .* BASEforHANK.IncomesETC.mutil(c_guess, m_par)
    Vk = (r + m_par.λ) .* BASEforHANK.IncomesETC.mutil(c_guess, m_par)
    


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

pdf_guess = ones(n_par.nm * n_par.nk,  n_par.nk, n_par.ny) / sum(n_par.nm * n_par.nk * n_par.nk * n_par.ny)
    cdf_guess = cumsum(cumsum(pdf_guess, dims=1), dims=2)

    @assert all(diff(cdf_guess, dims=2) .>= 0)
    @assert all(diff(diff(cdf_guess, dims=2), dims=1) .>= 0)

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

    total_wealth_unsorted = Array{eltype(cdf_guess)}(undef, n_par.nk .* n_par.nm)
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
        # @assert all(diff(m_a_star[:, :, i_y][total_wealth_sorting]) .>= 0)
        @assert all(diff(k_a_star[:, :, i_y][total_wealth_sorting]) .>= 0)
        @assert all(diff(k_a_star[:, :, i_y][total_wealth_sorting] .+ m_a_star[:, :, i_y][total_wealth_sorting]) .>= 0)
        for i_k = 1:n_par.nk
            @assert all(diff(m_n_star[:, i_k, i_y] .+ n_par.grid_k[i_k]) .>= 0)
        end
    end


    


cdf_prime = zeros(size(cdf_guess));
cdf_initial = copy(cdf_guess);


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

# include(path_to_transition)

DirectTransition_Splines_non_adjusters!(
    cdf_prime_on_grid_n,
    m_n_star, 
    cdf_initial,
    total_wealth_sorted,
    total_wealth_sorting,
    nk_map,
    1.0,
    n_par,
)


# m_n_star[:, 127, i_y] .+ n_par.grid_k[127]
# findlast(m_n_star[:, 127, i_y] .+ n_par.grid_k[127] .== n_par.grid_m[1] + n_par.grid_k[127])

# ------------------------------------
using PCHIPInterpolation
include(path_to_transition)


# cdf_prime = zeros(size(cdf_guess))
cdf_initial = copy(cdf_guess)

# Tolerance for change in cdf from period to period
tol = n_par.ϵ
# Maximum iterations to find steady state distribution
max_iter = 100
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

    println("$counts - distance: $distance")

end

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

CDF_TotalWealth = sum(cdf_initial[:, end, :], dims=2)[:]



K = BASEforHANK.SteadyState.expected_value(sum(cdf_initial[end, :, :], dims=2)[:], n_par.grid_k)
TotalWealth = BASEforHANK.SteadyState.expected_value(sum(cdf_initial[:, end, :], dims=2)[:], total_wealth_sorted)
B = TotalWealth - K

# ---------- convergence for adjusters and non-adjusters


# cdf_prime = zeros(size(cdf_guess))
cdf_initial = copy(cdf_guess)


# Tolerance for change in cdf from period to period
tol = n_par.ϵ
# Maximum iterations to find steady state distribution
max_iter = 50
# Init 
distance = 9999.0
counts = 0


# Iterate on distribution until convergence
while distance > tol && counts < max_iter
    counts = counts + 1
    cdf_old = copy(cdf_initial)

    DirectTransition_Splines_adjusters!(
        cdf_initial,
        m_a_star, 
        k_a_star,
        copy(cdf_initial),
        total_wealth_sorted,
        total_wealth_sorting,
        n_par,
    )

    # DirectTransition_Splines!(
    #     cdf_initial,
    #     m_n_star,
    #     m_a_star,
    #     k_a_star,
    #     copy(cdf_initial),
    #     n_par.Π,
    #     total_wealth_sorted,
    #     total_wealth_sorting,
    #     nk_map,
    #     n_par,
    #     m_par
    #     )

    difference = cdf_old .- cdf_initial
    distance = maximum(abs, difference)

    println("$counts - distance: $distance")

end

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

CDF_TotalWealth = sum(cdf_initial[:, end, :], dims=2)[:]


# non adjusters
include(path_to_transition)
cdf_initial = copy(cdf_guess)


# Tolerance for change in cdf from period to period
tol = n_par.ϵ
# Maximum iterations to find steady state distribution
max_iter = 50
# Init 
distance = 9999.0
counts = 0


# Iterate on distribution until convergence
while distance > tol && counts < max_iter
    counts = counts + 1
    cdf_old = copy(cdf_initial)

    DirectTransition_Splines_non_adjusters!(
        cdf_initial,
        m_n_star, 
        copy(cdf_initial),
        total_wealth_sorted,
        total_wealth_sorting,
        nk_map,
        1.0,
        n_par,
    )

    # DirectTransition_Splines!(
    #     cdf_initial,
    #     m_n_star,
    #     m_a_star,
    #     k_a_star,
    #     copy(cdf_initial),
    #     n_par.Π,
    #     total_wealth_sorted,
    #     total_wealth_sorting,
    #     nk_map,
    #     n_par,
    #     m_par
    #     )

    difference = cdf_old .- cdf_initial
    distance = maximum(abs, difference)

    println("$counts - distance: $distance")

end

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

CDF_TotalWealth = sum(cdf_initial[:, end, :], dims=2)[:]


# -----

m_n_star_y = copy(m_n_star[:,:,1])

w_n_mesh = n_par.mesh_k[:,:,1] .+ m_n_star_y 

w_n_mesh_sorted = w_n_mesh[total_wealth_sorting]

all(diff(w_n_mesh_sorted) .>= 0)
findall(diff(w_n_mesh_sorted) .< 0)

m_n_star_end_y .+ n_par.mesh_k


total_wealth_mesh = n_par.grid_m .+ n_par.mesh_k[:,:,1]

i_k = n_par.nk
cdf_prime_given_y_k = cdf_initial[:,i_k,1]
cdf_prime_given_y_k_selected = cdf_prime_given_y_k[nk_map[total_wealth_sorting] .== i_k]
w_n_prime_given_y_k = m_n_star[:,i_k,1] .+ n_par.grid_k[i_k]



# ---------------------

cdf_initial_on_grid = copy(cdf_guess)
cdf_prime_on_grid = similar(cdf_initial_on_grid)
grid_w_sorting = copy(total_wealth_sorting)
grid_w = copy(total_wealth_sorted)
m_n_prime = copy(m_n_star)
q = 1.0
i_y = 1

cdf_prime_given_y_diffk = [cdf_initial_on_grid[:,1,i_y] diff(cdf_initial_on_grid[:,:,i_y], dims=2)] 
        # cdf_prime_given_y_diffk = cdf_prime_given_y_diffk ./ repeat(cdf_prime_given_y_diffk[end,:],outer=(n_par.nk,1)) # normalize conditional cdf
        # cdf_k = Interpolator(
        #     n_par.grid_k,
        #     cdf_initial_on_grid[end,:,i_y]
        # )

# transforms joint cdf in w,k to cdf in w conditional on k
cdf_prime_given_y_diffk = [cdf_initial_on_grid[:,1,i_y] diff(cdf_initial_on_grid[:,:,i_y], dims=2)] 
# normalize conditional cdf # REVIEW: atm. normalization wrt. to k and y pdf -> should it still be conditional on y?
# cdf_prime_given_y_diffk = cdf_prime_given_y_diffk ./ repeat(cdf_prime_given_y_diffk[end,:],outer=(n_par.nk,1)) 
# # mapping of k grid to cdf of k (unconditional on m, but conditional on y)
# cdf_k = Interpolator(
#     n_par.grid_k,
#     cdf_initial_on_grid[end,:,i_y]
# )


# for (i_w,w) in enumerate(grid_w)
cdf_prime_on_grid = similar(cdf_initial_on_grid)

# for i_k = 1:n_par.nk

i_k = 1

    cdf_prime_given_y_k = view(cdf_prime_given_y_diffk, :, i_k)
    cdf_prime_given_y_k_selected = cdf_prime_given_y_k[nk_map[grid_w_sorting] .== i_k]  # TODO: use view?

    w_n_prime_given_y_k = m_n_prime[:, i_k, i_y] .+ q.*n_par.grid_k[i_k]

    idx_last_at_constraint = findlast(w_n_prime_given_y_k .== n_par.grid_m[1] .+ q.*n_par.grid_k[i_k])    # REVIEW - Correct? -> q required?
                idx_last_at_constraint = isnothing(idx_last_at_constraint) ? 1 : idx_last_at_constraint

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
        cdf_values[1:idx1] .= 0.0 #cdf_prime_given_y_k_selected[1]
        cdf_values[idx2:end] .= 1.0 * cdf_prime_given_y_k_selected[end] # REVIEW: Should it be 1.0 or cdf_prime_given_y_k_selected[end]?
        cdf_values[idx1+1:idx2-1] .= w_to_cdf_spline.(w[idx1+1:idx2-1])
    end

    cdf_prime_on_grid_given_k_y = view(cdf_prime_on_grid, :, i_k, i_y)
                # cdfend = copy(cdf_prime_given_y_k[end])
    w_to_cdf_spline_extr!(cdf_prime_on_grid_given_k_y, grid_w)

    # cdf_prime_on_grid_given_k_y

    # grid_w[end]
    # w_n_prime_given_y_k

# end

cdf_prime_on_grid

cdf_prime_on_grid_backup = copy(cdf_prime_on_grid)




for (i_w,w) in enumerate(grid_w)
    # i_w = 200
    k_condition_spline = Interpolator(
        n_par.grid_k,
        cdf_prime_on_grid[i_w, :, i_y]
    )

    k_condition_diff_cdf_spline = Interpolator(
        n_par.grid_k,
        [ForwardDiff.derivative(k_condition_spline,k)*cdf_initial_on_grid[end,i_k,i_y] for (i_k,k) in enumerate(n_par.grid_k)]
        #vcat((diff(cdf_prime_on_grid[i_w, :, i_y]).*cdf_initial_on_grid[end,:,i_y][1:end-1])[:],ForwardDiff.derivative(k_condition_spline,n_par.grid_k[end]))
    )
    right_part(k) = integrate(k_condition_diff_cdf_spline, n_par.grid_k[1], k)
    left_part(k) = k_condition_spline(k) * cdf_k(k) - k_condition_spline(n_par.grid_k[1]) * cdf_k(n_par.grid_k[1])
    int_mass(k) = left_part(k) - right_part(k)
    cdf_prime_on_grid[i_w, :, i_y] .= int_mass.(n_par.grid_k)
end
cdf_prime_on_grid[:,:,i_y] .= cdf_prime_on_grid[:,:,i_y] ./ cdf_prime_on_grid[end,end,i_y] .* cdf_initial_on_grid[end,end,i_y] # fix normalization issue due to interpolation inaccuracy
# end

left_part(n_par.grid_k[2])
right_part(n_par.grid_k[2])


plot(k_grid_fine, k_condition_spline.(k_grid_fine))


for (i_k, k) in enumerate(n_par.grid_k)
    println(integrate(k_condition_diff_cdf_spline, n_par.grid_k[1], i))
end