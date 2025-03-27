#------------------------------------------------------------------------------
# Header: load module
#------------------------------------------------------------------------------
# ATTENTION: make sure that your present working directory pwd() is set to the folder
# containing script.jl and BASEforHANK.jl. Otherwise adjust the load path.
# cd("./src")
# push!(LOAD_PATH, pwd())
# pre-process user inputs for model setup
using Printf
function printArray(a::AbstractArray)
    for i in 1:size(a)[1]
        for j in 1:size(a)[2]
            if j==size(a)[2]
                @printf("%.3f\n",a[i,j])
            else
                @printf("%.3f\t",a[i,j])
            end
        end
    end
end

function saveArray(filename::String,a::AbstractArray)
    dim = size(a)
    println("$dim")
    open(filename,"w") do file
    if length(dim)==1
        for i in 1:dim[1]
            write(file,"$(@sprintf("%.5f;\n",a[i]))")
        end
    elseif length(dim)==2
        for i in 1:dim[1]
            for j in 1:dim[2]
                write(file,"$(@sprintf("%.5f;",a[i,j]))")
                if j==dim[2]
                    write(file,"\n")
                else
                    write(file,"\t")
                end
            end
        end
    end
    end
end


include("Preprocessor/PreprocessInputs.jl")
# include("BASEforHANK.jl")
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
# println("Calculating the steady state")
# KSS, VmSS, VkSS, distrSS, n_par, m_par = BASEforHANK.find_steadystate_splines(m_par)

# ss_full_splines = BASEforHANK.SteadyStateStruct(KSS, VmSS, VkSS, CDFSS, n_par)
# jldsave("Output/Saves/steadystate_splines.jld2", true; ss_full_splines) 

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

K_guess = 25.0018

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

    c_a_star, m_a_star, k_a_star, c_n_star, m_n_star, Vm, Vk, m_a_aux, w_bar, aux_c = BASEforHANK.SteadyState.find_ss_policies(
            Vm,
            Vk,
            inc,
            eff_int,
            n_par,
            m_par,
            RB_guess,
            R_guess,
        )
    i_l, w_r = BASEforHANK.SteadyState.MakeWeightsLight(m_a_aux[1,:],n_par.grid_m) 
    w_k = ones(n_par.ny)
    for i_y in 1:n_par.ny
        if isempty(w_bar[i_y])
            w_k[i_y] = NaN
        else
            w_k[i_y] = w_bar[i_y][end] + w_r[i_y]*(n_par.grid_m[i_l[i_y]+1]+aux_c[i_l[i_y]+1,1]-(n_par.grid_m[i_l[i_y]]+aux_c[i_l[i_y],1]))
        end
    end
    w_m = [isempty(w_bar[i_y]) ? NaN : w_bar[i_y][1] for i_y in 1:n_par.ny]

# find_ss_distribution_splines --------------------------
# DirectTransitionSplines(

path_to_transition = "SubModules/SteadyState/IM_fcns/fcn_directtransition_conditionals.jl"
locate = BASEforHANK.Tools.locate
mylinearinterpolate = BASEforHANK.Tools.mylinearinterpolate
mylinearcondcdf = BASEforHANK.Tools.mylinearcondcdf
include(path_to_transition)

pdf_b_cond_k = ones(n_par.nm, n_par.nk, n_par.ny) / sum(n_par.nm * n_par.ny)
cdf_b_cond_k = cumsum(pdf_b_cond_k, dims=1)
cdf_k = cumsum(pdf_b_cond_k[1,:,:], dims=1)

for i_y = 1:n_par.ny
    cdf_b_cond_k[:,:,i_y] = distr_y[i_y].*cdf_b_cond_k[:,:,i_y] ./ cdf_b_cond_k[end,:,i_y]
    cdf_k[:,i_y] = distr_y[i_y].*cdf_k[:,i_y] ./ cdf_k[end,i_y]
end


cdf_b_cond_k_initial = copy(cdf_b_cond_k);
cdf_k_initial = copy(cdf_k);


cdf_b_cond_k_prime_on_grid_a = similar(cdf_b_cond_k_initial)
cdf_k_prime_on_grid_a = similar(cdf_k_initial)

# k_a_prime = k_a_star
# m_a_prime = m_a_star

w_eval_grid = [    (RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ RK .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)) /(RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar) for i_b in 1:length(n_par.w_sel_m), i_k in 1:length(n_par.w_sel_k), j_k in 1:n_par.nk    ]
#calc sorting for wealth
for i_b in 1:length(n_par.w_sel_m)
    for i_k in 1:length(n_par.w_sel_k)
        for j_k in 1:n_par.nk
            println([i_b,i_k,j_k]," ",(RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ RK .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)))
            println(1/(RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar))
        end
    end
end
# println("w_evalgrid: ",w_eval_grid)
w = NaN*ones(length(n_par.w_sel_m)*length(n_par.w_sel_k))
for (i_w_b,i_b) in enumerate(n_par.w_sel_m)
    for (i_w_k,i_k) in enumerate(n_par.w_sel_k)
        w[i_w_b + length(n_par.w_sel_m)*(i_w_k-1)] = RB .* n_par.grid_m[i_b] .+ RK .* n_par.grid_k[i_k] .+ m_par.Rbar .* n_par.grid_m[i_b] .* (n_par.grid_m[i_b] .< 0)
    end
end
sortingw = sortperm(w)

@load "Output/Saves/young250x250.jld2" ss_full_young
pdf_k_young = reshape(sum(ss_full_young.distrSS, dims = 1),ss_full_young.n_par.nk,ss_full_young.n_par.ny)
cdf_k_young = cumsum(pdf_k_young,dims=1)
cdf_m_young = cumsum(reshape(sum(ss_full_young.distrSS, dims = 2),ss_full_young.n_par.nm,ss_full_young.n_par.ny),dims=1)

@timev cdf_w, cdf_k_prime_dep_b = DirectTransition_Splines_adjusters!(
    cdf_b_cond_k_prime_on_grid_a,
    cdf_k_prime_on_grid_a,
    m_a_star, 
    k_a_star,
    cdf_b_cond_k_initial,
    cdf_k_initial,
    distr_y,
    RB,
    RK,
    sortingw,
    w[sortingw],
    w_eval_grid,
    m_a_aux,
    w_k,
    w_m,
    n_par,
    m_par;
    speedup = true
)

cdf_b_cond_k_prime_on_grid_n = similar(cdf_b_cond_k_initial)

# include(path_to_transition)

@timev DirectTransition_Splines_non_adjusters!(
    cdf_b_cond_k_prime_on_grid_n,
    m_n_star, 
    cdf_b_cond_k_initial,
    distr_y,
    n_par,
)

cdf_b_cond_k_prime_on_grid = m_par.λ .* cdf_b_cond_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n
cdf_k_prime_on_grid = m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

# TEST
distr_prime_on_grid = zeros(n_par.nm+1,n_par.nk,n_par.ny)
distr_prime_on_grid[n_par.nm+1,:,:] .= cdf_k_prime_on_grid
for i_y in 1:n_par.ny
    distr_prime_on_grid[1:n_par.nm,1,i_y] .= (m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .* cdf_k_prime_on_grid_a[1,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,1,i_y] .* cdf_k_initial[1,i_y])./ distr_prime_on_grid[n_par.nm+1,1,i_y]
    for i_k in 2:n_par.nk
        distr_prime_on_grid[1:n_par.nm,i_k,i_y] .= (m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) .+ (1.0 - m_par.λ) .* .5*(cdf_b_cond_k_prime_on_grid_n[:,i_k,i_y] + cdf_b_cond_k_prime_on_grid_n[:,i_k-1,i_y]).*(cdf_k_initial[i_k,i_y] .- cdf_k_initial[i_k-1,i_y]))./ (distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y])
    end
end

# m_n_star[:, 127, i_y] .+ n_par.grid_k[127]
# findlast(m_n_star[:, 127, i_y] .+ n_par.grid_k[127] .== n_par.grid_m[1] + n_par.grid_k[127])

# ------------------------------------
using PCHIPInterpolation
include(path_to_transition)

cdf_k_young_grid = cdf_k_young #similar(cdf_k_young)
cdf_b_cond_k_young = NaN*ones(ss_full_young.n_par.nm,ss_full_young.n_par.nk,ss_full_young.n_par.ny)
cdf_b_cond_k_young_grid = similar(cdf_b_cond_k_initial)
for i_y in 1:n_par.ny
    for i_k in 1:n_par.nk
        cdf_b_cond_k_young[:,i_k,i_y] = cumsum(ss_full_young.distrSS[:,i_k,i_y],dims=1)/pdf_k_young[i_k,i_y]
        cdf_b_cond_k_young_itp = Interpolator(ss_full_young.n_par.grid_m,cdf_b_cond_k_young[:,i_k,i_y])
        cdf_b_cond_k_young_grid[:,i_k,i_y] = cdf_b_cond_k_young_itp.(n_par.grid_m)
    end
end
cdf_k_young_grid = NaN*ones(n_par.nm,n_par.ny)
for i_y in 1:n_par.ny
    cdf_k_young_itp = Interpolator(ss_full_young.n_par.grid_k,cdf_k_young[:,i_y])
    cdf_k_young_grid[:,i_y] = cdf_k_young_itp.(n_par.grid_k)
end

cdf_m_young_grid = NaN*ones(n_par.nm,n_par.ny)
for i_y in 1:n_par.ny
    cdf_m_young_itp = Interpolator(ss_full_young.n_par.grid_m,cdf_m_young[:,i_y])
    cdf_m_young_grid[:,i_y] = cdf_m_young_itp.(n_par.grid_m)
end


# cdf_prime = zeros(size(cdf_guess))
distr_initial = NaN*ones(n_par.nm+1,n_par.nk,n_par.ny)
distr_initial[1:n_par.nm,:,:] = cdf_b_cond_k_initial
distr_initial[end,:,:] = cdf_k_young_grid



cdf_b_cond_k_initial = copy(distr_initial[1:n_par.nm,:,:])
cdf_k_initial = copy(reshape(distr_initial[n_par.nm+1,:,:], (n_par.nk, n_par.ny)));

cdf_b_cond_k_prime_on_grid_a = similar(cdf_b_cond_k_initial)
cdf_k_prime_on_grid_a = similar(cdf_k_initial)


cdf_w, cdf_k_prime_dep_b = DirectTransition_Splines_adjusters!(
        cdf_b_cond_k_prime_on_grid_a,
        cdf_k_prime_on_grid_a,
        m_a_star, 
        k_a_star,
        cdf_b_cond_k_initial,
        cdf_k_initial,
        distr_y,
        RB,
        RK,
        sortingw,
        w[sortingw],
        w_eval_grid,
        m_a_aux,
        w_k,
        w_m,
        n_par,
        m_par;
        speedup = true
    )

    cdf_b_cond_k_prime_on_grid_n = similar(cdf_b_cond_k_initial)


    DirectTransition_Splines_non_adjusters!(
        cdf_b_cond_k_prime_on_grid_n,
        m_n_star, 
        cdf_b_cond_k_initial,
        distr_y,
        n_par,
    )

    # println("distro adj: ")
    # printArray(cdf_b_cond_k_prime_on_grid_a[:,:,1])
    # println("distro non-adj")
    # printArray(cdf_b_cond_k_prime_on_grid_n[:,:,1])

    distr_prime_on_grid[n_par.nm+1,:,:] .= m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

    for i_y in 1:1#n_par.ny
        distr_prime_on_grid[1:n_par.nm,1,i_y] .= (m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .* cdf_k_prime_on_grid_a[1,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,1,i_y] .* cdf_k_initial[1,i_y])./ distr_prime_on_grid[n_par.nm+1,1,i_y]
        for i_k in 2:n_par.nk
            println(i_k,(distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y]))
            
            distr_prime_on_grid[1:n_par.nm,i_k,i_y] .= (m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) .+ (1.0 - m_par.λ) .* .5*(cdf_b_cond_k_prime_on_grid_n[:,i_k,i_y] + cdf_b_cond_k_prime_on_grid_n[:,i_k-1,i_y]).*(cdf_k_initial[i_k,i_y] .- cdf_k_initial[i_k-1,i_y]))./ (distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y])
            # println("row of dist: ",m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) )
        end
    end
    # println("distr_prime_on_grid[:,:,1])
    # println("distr k")
    # helper_fk = distr_prime_on_grid[n_par.nm+1,2:end,1]-distr_prime_on_grid[n_par.nm+1,1:end-1,1]
    # println(cdf_b_cond_k_prime_on_grid_a)
    n = size(distr_prime_on_grid)
    distr_prime_on_grid .= reshape(reshape(distr_prime_on_grid, (n[1] .* n[2], n[3])) * n_par.Π, (n[1], n[2], n[3]))










# Tolerance for change in cdf from period to period
tol = n_par.ϵ
# Maximum iterations to find steady state distribution
max_iter = 400
# Init 
distance = 9999.0 
counts = 0
# println("initial distro: ")
# printArray(distr_initial[:,:,1])
while distance > tol && counts < max_iter
    global counts, distance, distr_initial, cdf_w
    counts = counts + 1
    distr_old = copy(distr_initial)

    cdf_w = DirectTransition_Splines!(
        distr_initial,
        m_n_star,
        m_a_star,
        k_a_star,
        copy(distr_initial),
        n_par.Π,
        distr_y,
        RB,
        RK,
        w_eval_grid,
        sortingw,
        w[sortingw],
        m_a_aux,
        w_k,
        w_m,
        n_par,
        m_par;
        speedup = true
        )

    difference = distr_old .- distr_initial
    distance = maximum(abs, difference)

    println("$counts - distance: $distance")

    # mix in young marginal k distribution
    # distr_initial[end,:,:] = 0.5* distr_initial[end,:,:] .+ 0.5 .* cdf_k_young_grid

end

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

println("final distr: ")
printArray(distr_initial[:,:,1])
K = BASEforHANK.SteadyState.expected_value(sum(distr_initial[end,:,:],dims=2)[:],n_par.grid_k)
struct ssStruc
    distrSS
    n_par
    KSS
end
ss = ssStruc(distr_initial,n_par,K)
jldsave("Output/Saves/ssKfixed_250x2500.jld2", true; ss)

# compute marginal cdf of b
cdf_b = NaN*ones(n_par.nm,n_par.ny)
for i_y in 1:n_par.ny
    diffcdfk = diff(distr_initial[end,:,i_y],dims=1)/distr_initial[end,end,i_y]
    for i_b = 1:n_par.nm
        for i_k = 1:n_par.nk
            cdf_b[i_b,i_y] = distr_initial[end,1,i_y]/distr_initial[end,end,i_y]*distr_initial[i_b,1,i_y] + .5*sum((distr_initial[i_b,2:end,i_y] .+ distr_initial[i_b,1:end-1,i_y]).*diffcdfk)
        end
    end
end

B = BASEforHANK.SteadyState.expected_value(sum(cdf_b,dims=2)[:],n_par.grid_m)
# plot(n_par.grid_k,sum(distr_initial[end,:,:],dims=2))
# plot(n_par.grid_m,sum(cdf_b,dims=2))

for i_y = 1:n_par.ny
    for i_k = 1:10
        plot(n_par.grid_m,cdf_b_cond_k_young_grid[:,i_k,1],label="young",xlims=(0,20))
        plot!(n_par.grid_m,distr_initial[1:end-1,i_k,1]/distr_y[1],label="degm")
        vline!([m_a_aux[i_k,1]],label="m_a",linestyle=:dash)
        i_n = findfirst(m_n_star[:,2,1] .> m_a_aux[2,1])
        vline!([n_par.grid_m[i_n]],label="m_n^{-1}(m_a)",linestyle=:dash)
        savefig(string("Output/Figures/comp_cdf_b_k$i_k","_$i_y.pdf"))
    end
end