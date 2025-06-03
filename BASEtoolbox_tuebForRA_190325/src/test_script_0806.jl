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
            write(file,"$(@sprintf("%.15f\n",a[i]))")
        end
    elseif length(dim)==2
        for i in 1:dim[1]
            for j in 1:dim[2]
                write(file,"$(@sprintf("%.15f",a[i,j]))")
                if j==dim[2]
                    write(file,"\n")
                else
                    write(file,";\t")
                end
            end
        end
    end
    end
end


include("Preprocessor/PreprocessInputs.jl")
# include("BASEforHANK.jl")
using .BASEforHANK
using BenchmarkTools, Revise, LinearAlgebra, PCHIPInterpolation, ForwardDiff, Plots, Interpolations
# set BLAS threads to the number of Julia threads.
# prevents BLAS from grabbing all threads on a machine
BASEforHANK.LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

function merge_distr!(final::AbstractArray,merger::AbstractArray,cut::Int64,scale::Int64)
    scale = cut-1<scale ? cut-1 : scale
    weight = [i*1/scale for i in 0:scale]
    final[cut-scale:cut] .= (1 .-weight).*merger[cut-scale:cut].+ weight.*final[cut-scale:cut]
    final[1:cut-scale]= merger[1:cut-scale]
end

    
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
#    n_par.grid_k[1:n_par.nk] = exp.(range(log(n_par.kmin+ 1.0), stop = log(n_par.kmax+420  + 1.0), length = n_par.nk)) .- 1.0
# #    n_par.grid_k[n_par.nk]= 500
#    n_par.grid_m[1:n_par.nm] = exp.(range(0, stop = log(n_par.mmax+450 - n_par.mmin + 1.0), length = n_par.nm)) .+ n_par.mmin .- 1.0
#    n_par.grid_m[n_par.nm] = 500
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
    # func_dir  = "update_funk_500_on_500_each_500"
    # mkdir(func_dir)
    # for i in 1:n_par.ny
    # saveArray(func_dir*"/m_star$i"*"_a.csv",m_a_star[:,:,i])
    # saveArray(func_dir*"/m_star$i"*"_n.csv",m_n_star[:,:,i])
    # saveArray(func_dir*"/k_star$i"*"_a.csv",k_a_star[:,:,i])
    # end
    # assert(0==1)
    function pdf_from_spline!(cdf_initial::AbstractArray,pdf_initial::AbstractArray,neg_cut::AbstractArray,zero_occurance::AbstractArray,counting::Int64,pos::Int64)
        for i_y in 1:n_par.ny
        # i_y =1
        num_der = diff(cdf_initial[:,i_y])[1:end]
        # initial_cut = findfirst(num_der[40:end].==0)
        # initial_cut = isnothing(initial_cut) ? length(cdf_initial[:,i_y]) : initial_cut+40-1
        # cdf_k_initial_intp = Interpolator(vcat(n_par.grid_k[2:initial_cut-1],n_par.grid_k[end]),vcat(cdf_initial[2:initial_cut-1,i_y],cdf_initial[end,i_y]))
        # deriv_initial = k -> ForwardDiff.derivative(cdf_k_initial_intp,k)
        # pdf_k_initial_y = vcat(num_der[1]*ones(1),deriv_initial.(n_par.grid_k[3:end]))

        # cut_merge = findfirst(num_der.<0.001)
        # cut_merge = isnothing(cut_merge) ? initial_cut-2 : cut_merge-1
        # cutoff[counting,pos,i_y] = cut_merge
        # merge_distr!(pdf_k_initial_y,num_der,cut_merge,5)

        pdf_k_initial_y = num_der

        neg_index = findfirst(pdf_k_initial_y.<0)
        if ! isnothing(neg_index)
            # println("neg found: ",pos)
            cut_neg = findlast(pdf_k_initial_y[1:neg_index].>0)
            neg_cut[counting,pos,i_y] = cut_neg
            # endval =  
            resid = [i*(pdf_k_initial_y[cut_neg]-minimum(abs, [pdf_k_initial_y[cut_neg]/4,1e-20]))/(length(pdf_k_initial_y)-cut_neg) for i in 1:(length(pdf_k_initial_y)-cut_neg)]
            pdf_k_initial_y[cut_neg+1:end] = resid
        end
        zero_index = findfirst(pdf_k_initial_y[2:end].==0)
        if ! isnothing(zero_index)
            # println("zero found, ",pos," i_y=",i_y)
            indices = findall(pdf_k_initial_y[2:end].==0)
            zero_occurance[counting,pos,i_y,1:length(indices)] .= indices .+ 1
            # println("zero at: ",indices,"zero_index: ",zero_index)
            for index in indices
                index +=1
                if index ==2
                    pdf_k_initial_y[index] = pdf_k_initial_y[index+1]
                elseif index == length(pdf_k_initial_y)
                    pdf_k_initial_y[index] = pdf_k_initial_y[index-1]
                else
                    pdf_k_initial_y[index] = (pdf_k_initial_y[index+1]+pdf_k_initial_y[index-1])/2
                end
            end
        end
            

        pdf_initial[2:end,i_y] = pdf_k_initial_y
        pdf_initial[1,i_y] = cdf_k_initial[1,i_y]
        end
        
        
        # return initial_cut, cut_merge
    end


# find_ss_distribution_splines --------------------------
# DirectTransitionSplines(

path_to_transition = "SubModules/SteadyState/IM_fcns/fcn_directtransition_conditionals.jl"
locate = BASEforHANK.Tools.locate
mylinearinterpolate = BASEforHANK.Tools.mylinearinterpolate
mylinearcondcdf = BASEforHANK.Tools.mylinearcondcdf
include(path_to_transition)

pdf_b_cond_k = ones(n_par.nm, n_par.nk-1, n_par.ny) / sum(n_par.nm * n_par.ny)
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

w_eval_grid = [    (RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ RK .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)) /(RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar)*((RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ RK .* n_par.grid_k[n_par.w_sel_k[i_k]] ).>=(RK.*n_par.grid_k[j_k])) for i_b in 1:length(n_par.w_sel_m), i_k in 1:length(n_par.w_sel_k), j_k in 1:n_par.nk    ]
#calc sorting for wealth
# for i_b in 1:length(n_par.w_sel_m)
#     for i_k in 1:length(n_par.w_sel_k)
#         for j_k in 1:n_par.nk
#             println([i_b,i_k,j_k]," ",(RB .* n_par.grid_m[n_par.w_sel_m[i_b]] .+ RK .* (n_par.grid_k[n_par.w_sel_k[i_k]]-n_par.grid_k[j_k]) .+ m_par.Rbar .* n_par.grid_m[n_par.w_sel_m[i_b]] .* (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0)))
#             println(1/(RB .+ (n_par.grid_m[n_par.w_sel_m[i_b]] .< 0) .* m_par.Rbar))
#         end
#     end
# end
# println("w_evalgrid: ",w_eval_grid)

w_test = Array{Float64,2}(undef,length(n_par.w_sel_k)*length(n_par.w_sel_m),n_par.nk)
for ib in 1:length(n_par.w_sel_m)
    for ik in 1:length(n_par.w_sel_k)
        w_test[ib + length(n_par.w_sel_m)*(ik-1),:]= w_eval_grid[ib,ik,:]
    end
end

test = [i*j*l for i in ["1","2","3"], j in ["1","2","3"], l in ["1","2","3"]]

w = NaN*ones(length(n_par.w_sel_m)*length(n_par.w_sel_k))
for (i_w_b,i_b) in enumerate(n_par.w_sel_m)
    for (i_w_k,i_k) in enumerate(n_par.w_sel_k)
        w[i_w_b + length(n_par.w_sel_m)*(i_w_k-1)] = RB .* n_par.grid_m[i_b] .+ RK .* n_par.grid_k[i_k] .+ m_par.Rbar .* n_par.grid_m[i_b] .* (n_par.grid_m[i_b] .< 0)
    end
end
sortingw = sortperm(w)
# assert(0==1)
@load "Output/Saves/young250x250.jld2" ss_full_young
pdf_k_young = reshape(sum(ss_full_young.distrSS, dims = 1),ss_full_young.n_par.nk,ss_full_young.n_par.ny)
cdf_k_young = cumsum(pdf_k_young,dims=1)
cdf_m_young = cumsum(reshape(sum(ss_full_young.distrSS, dims = 2),ss_full_young.n_par.nm,ss_full_young.n_par.ny),dims=1)
check_young = ones(ss_full_young.n_par.nm,ss_full_young.n_par.nk,ss_full_young.n_par.ny)
pdf_k_young_grid = ones(size(n_par.grid_k_cdf)[1],n_par.ny)
cdf_b_cond_k_young_grid = ones(n_par.nm,n_par.nk-1,n_par.ny)
stretch = 25
for i_y in 1:4
square_grid = ones(ss_full_young.n_par.nm*stretch,ss_full_young.n_par.nk)
for i_k in 1:ss_full_young.n_par.nk
    intp_pdf = Interpolator(ss_full_young.n_par.grid_m,ss_full_young.distrSS[:,i_k,i_y])
    a = intp_pdf.(range(0,ss_full_young.n_par.grid_m[end],length=ss_full_young.n_par.nm*stretch))
    # println("len of intp ",size(a))
     square_grid[:,i_k] =a
end

square_grid_k = ones(ss_full_young.n_par.nm*stretch,ss_full_young.n_par.nk*stretch)
for i_m in 1:ss_full_young.n_par.nm*stretch
    intp_pdf = Interpolator(ss_full_young.n_par.grid_k,square_grid[i_m,:])
    square_grid_k[i_m,:]= intp_pdf.(range(0,ss_full_young.n_par.grid_k[end],length=ss_full_young.n_par.nk*stretch))
end

hom_k = range(0,ss_full_young.n_par.grid_k[end],length=ss_full_young.n_par.nk*stretch)
hom_m = range(0,ss_full_young.n_par.grid_m[end],length=ss_full_young.n_par.nm*stretch)

pdf_b_k_combined_intp = LinearInterpolation((hom_m,hom_k),square_grid_k)
# pdf_b_k_combined_intp = cubic_spline_interpolation((hom_m,hom_k),square_grid_k)
pdf_grid_for_k = [pdf_b_k_combined_intp(n_par.grid_m[ib],n_par.grid_k_cdf[ik]) 
for ib in 1:n_par.nm, ik in 1:size(n_par.grid_k_cdf)[1]]
pdf_grid_for_cond = [pdf_b_k_combined_intp(n_par.grid_m[ib],n_par.grid_k[ik]) 
for ib in 1:n_par.nm, ik in 1:n_par.nk-1]
pdf_k_young_grid[:,i_y] = sum(pdf_grid_for_k,dims=1)
cdf_b_cond_k_young_grid[:,:,i_y] = cumsum(pdf_grid_for_cond,dims=1) ./ sum(pdf_grid_for_k,dims=1)
cdf_b_cond_k_young_grid[:,:,i_y] = cdf_b_cond_k_young_grid[:,:,i_y] ./ cdf_b_cond_k_young_grid[end,:,i_y]'
# Delta = 0.00
# check_young[:,:,i_y] = [pdf_b_k_combined_intp(ss_full_young.n_par.grid_m[ib]+Delta,ss_full_young.n_par.grid_k[ik]+Delta) for ib in 1:ss_full_young.n_par.nm, ik in 1:ss_full_young.n_par.nk]
# ss_full_young.distrSS[1:end-1,1:end-1,1]-check_young

    # saveArray("out/pdf_young_comb_sqrgrid_y$i_y.csv",square_grid_k[:,:])
    # saveArray("out/pdf_young_comb_org_y$i_y.csv",ss_full_young.distrSS[:,:,i_y])
    # saveArray("out/pdf_young_comb_intp_y$i_y.csv",check_young[:,:,i_y])
    saveArray("out/cdf_b_cond_k_young_y$i_y.csv",cdf_b_cond_k_young_grid[:,:,i_y])
end
saveArray("out/pdf_k_young_grid.csv",pdf_k_young_grid)
# assert(1==0)
# @timev cdf_w, cdf_k_prime_dep_b = DirectTransition_Splines_adjusters!(
#     cdf_b_cond_k_prime_on_grid_a,
#     cdf_k_prime_on_grid_a,
#     m_a_star, 
#     k_a_star,
#     cdf_b_cond_k_initial,
#     cdf_k_initial,
#     distr_y,
#     RB,
#     RK,
#     sortingw,
#     w[sortingw],
#     w_eval_grid,
#     m_a_aux,
#     w_k,
#     w_m,
#     n_par,
#     m_par;
#     speedup = false
# )

# cdf_b_cond_k_prime_on_grid_n = similar(cdf_b_cond_k_initial)


# # include(path_to_transition)

# @timev DirectTransition_Splines_non_adjusters!(
#     cdf_b_cond_k_prime_on_grid_n,
#     m_n_star, 
#     cdf_b_cond_k_initial,
#     distr_y,
#     n_par,
# )
# distr_prime_on_grid = zeros(n_par.nm+1,n_par.nk,n_par.ny)
# cdf_k_initial_orig = copy(cdf_k_initial)
# cdf_k_initial = cdf_k_initial ./ cdf_k_initial[end,:]'
# distr_prime_on_grid[n_par.nm+1,:,:] .= m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

#     for i_y in 1:n_par.ny
#         distr_prime_on_grid[1:n_par.nm,1,i_y] .= (m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .* cdf_k_prime_on_grid_a[1,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,1,i_y] .* cdf_k_initial[1,i_y])./ distr_prime_on_grid[n_par.nm+1,1,i_y]
#         for i_k in 2:n_par.nk
#             distr_k_finite = distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y]
#             distr_k_initial_finite = cdf_k_initial[i_k,i_y] .- cdf_k_initial[i_k-1,i_y]
#             if distr_k_finite ==0
#                 distr_k_finite= 1e-16
#             end
#             if distr_k_initial_finite==0
#                 distr_k_initial_finite=1e-16
#             end
#             distr_prime_on_grid[1:n_par.nm,i_k,i_y] .= (m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) .+ (1.0 - m_par.λ) .* .5*(cdf_b_cond_k_prime_on_grid_n[:,i_k,i_y] + cdf_b_cond_k_prime_on_grid_n[:,i_k-1,i_y]).*distr_k_initial_finite)./ (distr_k_finite)
#         end
#     end
#     # println("")
#     # println("distr_bcondk prior normalisation: ")
#     # printArray(distr_prime_on_grid[:,:,1])
    
#     for i_y in 1:n_par.ny
        
#             distr_prime_on_grid[:,:,i_y] .= distr_prime_on_grid[:,:,i_y].*distr_y[i_y]
        
#     end


# saveArray("out/m_star_a.csv",m_a_star[:,:,1])
# saveArray("out/k_star_a.csv",k_a_star[:,:,1])
# saveArray("out/m_star_n.csv",m_n_star[:,:,1])
# saveArray("out/cdf_k_initial.csv",cdf_k_initial)


# saveArray("out/cdfb_condk.csv",cdf_b_cond_k_initial[:,:,1])
# saveArray("out/bcondk_a.csv",cdf_b_cond_k_prime_on_grid_a[:,:,1])
# saveArray("out/bcondk_n.csv",cdf_b_cond_k_prime_on_grid_n[:,:,1])
# saveArray("out/k_a.csv",cdf_k_prime_on_grid_a)
# saveArray("out/cdf_prime.csv",distr_prime_on_grid[:,:,1]/distr_y[1])



# cdf_b_cond_k_prime_on_grid = m_par.λ .* cdf_b_cond_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n
# cdf_k_prime_on_grid = m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

# # TEST
# distr_prime_on_grid = zeros(n_par.nm+1,n_par.nk,n_par.ny)
# distr_prime_on_grid[n_par.nm+1,:,:] .= cdf_k_prime_on_grid
# for i_y in 1:n_par.ny
#     distr_prime_on_grid[1:n_par.nm,1,i_y] .= (m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .* cdf_k_prime_on_grid_a[1,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,1,i_y] .* cdf_k_initial[1,i_y])./ distr_prime_on_grid[n_par.nm+1,1,i_y]
#     for i_k in 2:n_par.nk
#         distr_prime_on_grid[1:n_par.nm,i_k,i_y] .= (m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) .+ (1.0 - m_par.λ) .* .5*(cdf_b_cond_k_prime_on_grid_n[:,i_k,i_y] + cdf_b_cond_k_prime_on_grid_n[:,i_k-1,i_y]).*(cdf_k_initial[i_k,i_y] .- cdf_k_initial[i_k-1,i_y]))./ (distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y])
#     end
# end

# m_n_star[:, 127, i_y] .+ n_par.grid_k[127]
# findlast(m_n_star[:, 127, i_y] .+ n_par.grid_k[127] .== n_par.grid_m[1] + n_par.grid_k[127])

# ------------------------------------
using PCHIPInterpolation
include(path_to_transition)

# cdf_k_young_grid = cdf_k_young #similar(cdf_k_young)
# cdf_b_cond_k_young = NaN*ones(ss_full_young.n_par.nm,ss_full_young.n_par.nk,ss_full_young.n_par.ny)
# cdf_b_cond_k_young_grid = similar(cdf_b_cond_k_initial)
# # pdf_b_k_combined_intp
# for i_y in 1:n_par.ny
#     for i_k in 1:n_par.nk-1
#         cdf_b_cond_k_young[:,i_k,i_y] = cumsum(ss_full_young.distrSS[:,i_k,i_y],dims=1)/pdf_k_young[i_k,i_y]
#         cdf_b_cond_k_young_itp = Interpolator(ss_full_young.n_par.grid_m,cdf_b_cond_k_young[:,i_k,i_y])
#         cdf_b_cond_k_young_grid[:,i_k,i_y] = cdf_b_cond_k_young_itp.(n_par.grid_m)
#     end
# end
# cdf_k_young_grid = NaN*ones(size(n_par.grid_k_cdf)[1],n_par.ny)
# for i_y in 1:n_par.ny
#     cdf_k_young_itp = Interpolator(ss_full_young.n_par.grid_k,cdf_k_young[:,i_y])
#     cdf_k_young_grid[:,i_y] = cdf_k_young_itp.(n_par.grid_k_cdf)
# end
# # cdf_k_young_grid = cdf_k_young_grid ./ cdf_k_young_grid[end,:]'
# pdf_k_young_grid = NaN*ones(n_par.nk,n_par.ny)
# for i_y in 1:n_par.ny
#     pdf_k_young_itp = Interpolator(ss_full_young.n_par.grid_k,pdf_k_young[:,i_y])
#     pdf_k_young_grid[:,i_y] = pdf_k_young_itp.(n_par.grid_k)
# end
# pdf_k_young_grid = pdf_k_young_grid ./ cdf_k_young[end,:]'
# cdf_m_young_grid = NaN*ones(n_par.nm,n_par.ny)
# for i_y in 1:n_par.ny
#     cdf_m_young_itp = Interpolator(ss_full_young.n_par.grid_m,cdf_m_young[:,i_y])
#     cdf_m_young_grid[:,i_y] = cdf_m_young_itp.(n_par.grid_m)
# end


# cdf_prime = zeros(size(cdf_guess))
distr_initial = NaN*ones(n_par.nm+1,n_par.nk-1,n_par.ny)
distr_initial[1:n_par.nm,:,:] = cdf_b_cond_k_initial
distr_initial[end,:,:] = cumsum(pdf_k_young_grid,dims=1)

# saveArray("cdf_k_young_800k_in_200steps.csv",cdf_k_young_grid)
# for i_y in 1:4
# saveArray("out/cdf_bcondk_young_800k_500b_200steps_iy$i_y.csv",cdf_b_cond_k_young_grid[:,:,i_y])
# end
# for i_y in 1:4
#     saveArray("out/pdf_bcondk_young_org_iy$i_y.csv",ss_full_young.distrSS[:,:,i_y])
# end

# saveArray("out/pdf_k_young_org.csv",pdf_k_young./cdf_k_young[end,:]')
# saveArray("out/pdf_k_young_grid.csv",pdf_k_young_grid)

# saveArray("out/k_young.csv",ss_full_young.n_par.grid_k)
# assert(0==1)
# for i_y in 1:4
#     saveArray("out/cdf_bcondk_young_grid_iy$i_y.csv",cdf_b_cond_k_young_grid[:,:,i_y])
# end
# assert(1==0)
# cdf_b_cond_k_initial = copy(distr_initial[1:n_par.nm,:,:])
# cdf_k_initial = copy(reshape(distr_initial[n_par.nm+1,:,:], (n_par.nk, n_par.ny)));

# cdf_b_cond_k_prime_on_grid_a = similar(cdf_b_cond_k_initial)
# cdf_k_prime_on_grid_a = similar(cdf_k_initial)


# cdf_w, cdf_k_prime_dep_b = DirectTransition_Splines_adjusters!(
#         cdf_b_cond_k_prime_on_grid_a,
#         cdf_k_prime_on_grid_a,
#         m_a_star, 
#         k_a_star,
#         cdf_b_cond_k_initial,
#         cdf_k_initial,
#         distr_y,
#         RB,
#         RK,
#         sortingw,
#         w[sortingw],
#         w_eval_grid,
#         m_a_aux,
#         w_k,
#         w_m,
#         n_par,
#         m_par;
#         speedup = true
#     )

#     cdf_b_cond_k_prime_on_grid_n = similar(cdf_b_cond_k_initial)


#     DirectTransition_Splines_non_adjusters!(
#         cdf_b_cond_k_prime_on_grid_n,
#         m_n_star, 
#         cdf_b_cond_k_initial,
#         distr_y,
#         n_par,
#     )

    # println("distro adj: ")
    # printArray(cdf_b_cond_k_prime_on_grid_a[:,:,1])
    # println("distro non-adj")
    # printArray(cdf_b_cond_k_prime_on_grid_n[:,:,1])

    # distr_prime_on_grid[n_par.nm+1,:,:] .= m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

    # for i_y in 1:1#n_par.ny
    #     distr_prime_on_grid[1:n_par.nm,1,i_y] .= (m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .* cdf_k_prime_on_grid_a[1,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,1,i_y] .* cdf_k_initial[1,i_y])./ distr_prime_on_grid[n_par.nm+1,1,i_y]
    #     for i_k in 2:n_par.nk
    #         # println(i_k,(distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y]))
            
    #         distr_prime_on_grid[1:n_par.nm,i_k,i_y] .= (m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) .+ (1.0 - m_par.λ) .* .5*(cdf_b_cond_k_prime_on_grid_n[:,i_k,i_y] + cdf_b_cond_k_prime_on_grid_n[:,i_k-1,i_y]).*(cdf_k_initial[i_k,i_y] .- cdf_k_initial[i_k-1,i_y]))./ (distr_prime_on_grid[n_par.nm+1,i_k,i_y] .- distr_prime_on_grid[n_par.nm+1,i_k-1,i_y])
    #         # println("row of dist: ",m_par.λ .*(cdf_k_prime_dep_b[:,i_k,i_y]-cdf_k_prime_dep_b[:,i_k-1,i_y]) )
    #     end
    # end
    # println("distr_prime_on_grid[:,:,1])
    # println("distr k")
    # helper_fk = distr_prime_on_grid[n_par.nm+1,2:end,1]-distr_prime_on_grid[n_par.nm+1,1:end-1,1]
    # println(cdf_b_cond_k_prime_on_grid_a)
    # n = size(distr_prime_on_grid)
    # distr_prime_on_grid .= reshape(reshape(distr_prime_on_grid, (n[1] .* n[2], n[3])) * n_par.Π, (n[1], n[2], n[3]))






# printArray(cutof[:,:,1])


# Tolerance for change in cdf from period to period
# tol = n_par.ϵ
tol = 1e-7
# Maximum iterations to find steady state distribution
max_iter =200
# Init 
convergence_course = NaN*ones(max_iter)
distance = 9999.0 
counts = 0
cut_negatives = ones(Int64,(max_iter,3,n_par.ny)) 
zero_occurances = ones(Int64,(max_iter,3,n_par.ny,50)) 
cut_negatives .*= (-1)
zero_occurances .*= (-1)
# println("initial distro: ")
# printArray(distr_initial[:,:,1])
distr_y_update = transpose(copy(distr_y))
while distance > tol && counts < max_iter
    global counts, distance, distr_initial, cdf_w, distr_y_update
    counts = counts + 1
    distr_old = copy(distr_initial)
    

    cdf_w, counts = DirectTransition_Splines!(
        distr_initial,
        m_n_star,
        m_a_star,
        k_a_star,
        copy(distr_initial),
        n_par.Π,
        distr_y_update,
        RB,
        RK,
        w_eval_grid,
        sortingw,
        w[sortingw],
        m_a_aux,
        w_k,
        w_m,
        n_par,
        m_par,
        counts,
        distance,
        cut_negatives,
        zero_occurances,
        pdf_k_young_grid,
        cdf_k_young_grid;
        speedup = false
        )
    # distr_y_update = distr_y_update * n_par.Π
    # distr_y_update .= distr_y_update./ sum(distr_y_update)
    difference = distr_old .- distr_initial
    distance = maximum(abs, difference)
    convergence_course[counts] = distance
    println("$counts - distance: $distance")
    
    # mix in young marginal k distribution
    # distr_initial[end,:,:] = 0.5* distr_initial[end,:,:] .+ 0.5 .* cdf_k_young_grid

end

saveArray("out/difference.csv",convergence_course)

println("Distribution Iterations: ", counts)
println("Distribution Dist: ", distance)

# println("final distr: ")
# printArray(distr_initial[:,:,1])
K = BASEforHANK.SteadyState.expected_value(sum(distr_initial[end,:,:],dims=2)[:],n_par.grid_k[1:end-1])
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
mkdir("out/Figures")
for i_y = 1:n_par.ny
    for i_k = 1:40
        plot(n_par.grid_m,cdf_b_cond_k_young_grid[:,i_k,1],label="young",xlims=(0,20))
        plot!(n_par.grid_m,distr_initial[1:end-1,i_k,1]/distr_y[1],label="degm")
        vline!([m_a_aux[i_k,1]],label="m_a",linestyle=:dash)
        i_n = findfirst(m_n_star[:,2,1] .> m_a_aux[2,1])
        vline!([n_par.grid_m[i_n]],label="m_n^{-1}(m_a)",linestyle=:dash)
        savefig(string("out/Figures/comp_cdf_b_k$i_k","_$i_y.pdf"))
    end
end