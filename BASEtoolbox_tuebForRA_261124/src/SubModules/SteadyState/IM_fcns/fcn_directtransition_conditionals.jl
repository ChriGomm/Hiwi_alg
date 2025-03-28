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
    RB,
    R,
    w_eval_grid,
    n_par::NumericalParameters,
    m_par::ModelParameters,
)
    pos = findall(isapprox.(eigvals(Π'), 1.0; atol = 1e-8))[1]
    eigvec = eigvecs(Π')[:,pos]
    pdf_inc = eigvec * sign(eigvec[1])./norm(eigvec,1)

    dPrime = zeros(eltype(distr), size(distr))
    distr_prime = zeros(eltype(distr), size(distr))

    distr_initial = distr

    DirectTransition_Splines!(distr_prime, m_n_star, m_a_star, k_a_star, copy(distr_initial), Π, pdf_inc, RB, R, w_eval_grid, n_par, m_par)

    return distr_prime
end


function DirectTransition_Splines!(
    distr_prime_on_grid::AbstractArray,   
    m_n_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    m_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    k_a_prime::AbstractArray,   # Defined as policy over liquid wealth x illiquid wealth x income
    distr_initial_on_grid::AbstractArray,     
    Π::AbstractArray,
    pdf_inc::AbstractArray,
    RB,
    R,
    w_eval_grid,
    w_grid_sort,
    wgrid,
    m_a_aux,
    w_k,
    w_m,
    n_par::NumericalParameters,
    m_par::ModelParameters;
    speedup::Bool = true
)

    cdf_b_cond_k_initial = copy(distr_initial_on_grid[1:n_par.nm,:,:])
    cdf_k_initial = copy(reshape(distr_initial_on_grid[n_par.nm+1,:,:], (n_par.nk, n_par.ny)));

    cdf_b_cond_k_prime_on_grid_a = similar(cdf_b_cond_k_initial)
    cdf_k_prime_on_grid_a = similar(cdf_k_initial)

    cdf_w = DirectTransition_Splines_adjusters!(
        cdf_b_cond_k_prime_on_grid_a,
        cdf_k_prime_on_grid_a,
        m_a_prime, 
        k_a_prime,
        cdf_b_cond_k_initial,
        cdf_k_initial,
        pdf_inc,
        RB,
        R,
        w_grid_sort,
        wgrid,
        w_eval_grid,
        m_a_aux,
        w_k,
        w_m,
        n_par,
        m_par;
        speedup = speedup
    )

    cdf_b_cond_k_prime_on_grid_n = similar(cdf_b_cond_k_initial)

    DirectTransition_Splines_non_adjusters!(
        cdf_b_cond_k_prime_on_grid_n,
        m_n_prime, 
        cdf_b_cond_k_initial,
        pdf_inc,
        n_par,
    )
    # To add up conditional distribution, have to be careful with small k that are not reached by adjusters with high incomes!
    # Conditioning on those k already implies non-adjustment status!
    for i_y in 1:n_par.ny
        i_k_start = findfirst(cdf_k_prime_on_grid_a[:,i_y] .> 0.0) #locate(k_a_prime[1,1,i_y],n_par.grid_k)
        i_k_end = locate(k_a_prime[end,end,i_y],n_par.grid_k)
        distr_prime_on_grid[1:n_par.nm,i_k_start:i_k_end,i_y] .= m_par.λ .* cdf_b_cond_k_prime_on_grid_a[:,i_k_start:i_k_end,i_y] .+ (1.0 - m_par.λ) .* cdf_b_cond_k_prime_on_grid_n[:,i_k_start:i_k_end,i_y]
        distr_prime_on_grid[1:n_par.nm,1:i_k_start-1,i_y] .= cdf_b_cond_k_prime_on_grid_n[:,1:i_k_start-1,i_y]
        distr_prime_on_grid[1:n_par.nm,i_k_end+1:end,i_y] .= cdf_b_cond_k_prime_on_grid_n[:,i_k_end+1:end,i_y]
    end
    distr_prime_on_grid[n_par.nm+1,:,:] .= m_par.λ .* cdf_k_prime_on_grid_a .+ (1.0 - m_par.λ) .* cdf_k_initial

    n = size(distr_prime_on_grid)
    distr_prime_on_grid .= reshape(reshape(distr_prime_on_grid, (n[1] .* n[2], n[3])) * Π, (n[1], n[2], n[3]))

    return cdf_w

end

function DirectTransition_Splines_adjusters!(
    cdf_b_cond_k_prime_on_grid_a::AbstractArray,
    cdf_k_prime_on_grid_a::AbstractArray,   
    m_a_prime::AbstractArray,           
    k_a_prime::AbstractArray,           
    cdf_b_cond_k_initial::AbstractArray,
    cdf_k_initial::AbstractArray,
    pdf_inc::AbstractArray,
    RB,
    R,
    w_grid_sort::AbstractArray,
    wgrid::AbstractArray,
    w_eval_grid::AbstractArray,
    m_a_aux::AbstractArray,
    w_k::AbstractArray,
    w_m::AbstractArray,
    n_par::NumericalParameters,
    m_par::ModelParameters;
    speedup::Bool = true
)   
    cdf_w = NaN*ones(eltype(cdf_k_initial),length(n_par.w_sel_k)*length(n_par.w_sel_m), n_par.ny)
    cdfend = 1.0
    for i_y in 1:n_par.ny
        # 0. normalize
        pdf_y = cdf_k_initial[end,i_y]
        # 1. Need to generate total wealth distribution from cdf_b_cond_k_initial, cdf_k_initial
        cdf_w_y = view(cdf_w,:,i_y)
        m_a_aux_y = view(m_a_aux,:,i_y)
        cdf_b_cond_k_intp = Array{eltype(cdf_k_initial)}(undef, n_par.nk,length(n_par.w_sel_k)*length(n_par.w_sel_m))
        if speedup
            for i_k = 1:n_par.nk
                cdf_b_cond_k_intp[i_k,:] = mylinearcondcdf(n_par.grid_m,cdf_b_cond_k_initial[:,i_k,i_y]./pdf_y,reshape(w_eval_grid[:,:,i_k],length(n_par.w_sel_k)*length(n_par.w_sel_m)))
                
            end      
        else
            for i_k = 1:n_par.nk
                intp_cond =  b -> b < n_par.grid_m[1] ? 0.0 : (b > n_par.grid_m[end] ? cdf_b_cond_k_initial[end,i_k,i_y]./pdf_y : Interpolator(
                    n_par.grid_m,
                    cdf_b_cond_k_initial[:,i_k,i_y]./pdf_y
                )(b))
                cdf_b_cond_k_intp[i_k,:] = intp_cond.(reshape(w_eval_grid[:,:,i_k],length(n_par.w_sel_k)*length(n_par.w_sel_m)))
                
            end      
        end             
        cdf_b_cond_k_intp[1,1] = cdf_b_cond_k_initial[1,1,i_y]./pdf_y     
        diffcdfk = diff(cdf_k_initial[:,i_y],dims=1)/pdf_y
        for i_w_b = 1:length(n_par.w_sel_m)
            for i_w_k = 1:length(n_par.w_sel_k)
                cdf_w_y[i_w_b + (i_w_k-1)*length(n_par.w_sel_m)] = cdf_k_initial[1,i_y]*cdf_b_cond_k_intp[1,i_w_b+(i_w_k-1)*length(n_par.w_sel_m)]/pdf_y + .5*sum((cdf_b_cond_k_intp[2:end,i_w_b+(i_w_k-1)*length(n_par.w_sel_m)] .+ cdf_b_cond_k_intp[1:end-1,i_w_b+(i_w_k-1)*length(n_par.w_sel_m)]).*diffcdfk)
            end
        end
        optk_unsorted = view(k_a_prime,n_par.w_sel_m,n_par.w_sel_k,i_y)[:]
        optb_unsorted = view(m_a_prime,n_par.w_sel_m,n_par.w_sel_k,i_y)[:]
        optk_sorted = optk_unsorted[w_grid_sort]
        optb_sorted = optb_unsorted[w_grid_sort]
        # normalize cdf_w
        cdf_w_y .= min.(cdf_w_y,cdfend)
        cdf_w_y[end] = cdfend
        # compute spline of cdf_w
        cdf_w_y_spl = Interpolator(wgrid,cdf_w_y[w_grid_sort])
        # 2. Compute cdf over k' with DEGM
        if isnan(w_k[i_y]) | (w_k[i_y] < wgrid[1]) # k=0 is zero probability event
            nodes_k = optk_sorted
            values_k = cdf_w_y[w_grid_sort]
        else
            w_li = locate(w_k[i_y],wgrid)
            # make sure that k*=0 is included
            w_li2 = findlast(optk_sorted .== n_par.grid_k[1])
            w_li = min(w_li,w_li2)
            nodes_k = optk_sorted[w_li:end]
            values_k = cdf_w_y[w_grid_sort][w_li:end]
        end
        cdf_k_int = k -> k < nodes_k[1] ? 0.0 : Interpolator(nodes_k, values_k)(k) #
        cdf_k_prime_on_grid_a[:,i_y] .= cdf_k_int.(n_par.grid_k)
        # 3. Compute cdf over b' conditional on k'
        # 3.1 Start with k'>0
        for i_k = 2:n_par.nk
            cdf_b_cond_k_prime_on_grid_a[:,i_k,i_y] .= Float64.(m_a_aux_y[i_k] .<= n_par.grid_m)
        end
        # 3.2 Add k'=0
        if isnan(w_k[i_y]) | (w_k[i_y] < wgrid[1]) # k=0 is zero probability event, but with extrapolation in cdf_k_int, it now matters! However, m_a_aux just puts everything on m=0.
            cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .= Float64.(m_a_aux_y[1] .<= n_par.grid_m)
        else
            w_li = locate(w_k[i_y],wgrid)
            i_wk = w_li+1
            cdf_k_prime_on_grid_a[1,i_y] = cdf_w_y_spl(w_k[i_y])
            if w_m[i_y] < wgrid[1]
                i_wb = 1
            else
                i_wb = locate(w_m[i_y],wgrid)
            end
            nodes2 = optb_sorted[max(i_wb-1,i_wk)+1:end]
            values2 = ones(length(nodes2))
            nodes1 = optb_sorted[i_wb:i_wk]
            values1 = (cdf_w_y[w_grid_sort][i_wb:i_wk]./cdf_k_prime_on_grid_a[1,i_y])

            cdf_b_cond_k_prime_int = b -> b > nodes2[end] ? 1.0 : (b < nodes1[1] ? (1 - (nodes1[1] - b)/(nodes1[1]-n_par.grid_m[1]))*values1[1] : Interpolator(vcat(nodes1,nodes2),vcat(values1,values2))(b))
            cdf_b_cond_k_prime_on_grid_a[:,1,i_y] .= cdf_b_cond_k_prime_int.(n_par.grid_m)
            if w_m[i_y] >= wgrid[1]
                cdf_b_cond_k_prime_on_grid_a[1,1,i_y] = cdf_w_y_spl(w_m[i_y])/cdf_k_prime_on_grid_a[1,i_y] 
            end
        end
        # normalize
        cdf_k_prime_on_grid_a[:,i_y] .= min.(cdf_k_prime_on_grid_a[:,i_y],cdfend)
        cdf_k_prime_on_grid_a[end,i_y] = cdfend
        cdf_b_cond_k_prime_on_grid_a[:,:,i_y] .= min.(cdf_b_cond_k_prime_on_grid_a[:,:,i_y],cdfend)
        cdf_b_cond_k_prime_on_grid_a[end,:,i_y] .= cdfend
        # 4. Times pdf(i_y)
        cdf_b_cond_k_prime_on_grid_a[:,:,i_y] .= cdf_b_cond_k_prime_on_grid_a[:,:,i_y] .* pdf_inc[i_y]
        cdf_k_prime_on_grid_a[:,i_y] .= cdf_k_prime_on_grid_a[:,i_y] .* pdf_inc[i_y]
    end
    return cdf_w
end



function DirectTransition_Splines_non_adjusters!(
    cdf_b_cond_k_prime_on_grid_n::AbstractArray,      
    m_n_prime::AbstractArray,               
    cdf_b_cond_k_initial::AbstractArray,  
    pdf_inc::AbstractArray,
    n_par::NumericalParameters,
)
    for i_y = 1:n_par.ny
        cdfend = pdf_inc[i_y]
        for i_k = 1:n_par.nk
            cdf_b_cond_k_given_y_k = view(cdf_b_cond_k_prime_on_grid_n,:,i_k,i_y)
            i_mmin = findlast(m_n_prime[:,i_k,i_y] .== n_par.grid_m[1])
            i_mmin_adj = isnothing(i_mmin) ? 1 : i_mmin
            intp = Interpolator(m_n_prime[i_mmin_adj:end,i_k,i_y], cdf_b_cond_k_initial[i_mmin_adj:end,i_k,i_y])
            function m_to_cdf_spline_extr!(cdf_values::AbstractVector, m::Vector{Float64})
                idx1 = findlast(m .< m_n_prime[1,i_k,i_y])
                idx1 = isnothing(idx1) ? 0 : idx1
                # index for values above highest observed decision
                # idx2 = findfirst(w .> min(w_at_max_cdf, w_n_prime_given_y_k[end]))
                idx2 = findfirst(m .> m_n_prime[end,i_k,i_y])
                idx2 = isnothing(idx2) ? length(m) + 1 : idx2
                # inter- and extrapolation
                # wealth below lowest observed policy (conditional on k) do not happen (would correspond to w<k_j)
                cdf_values[1:idx1] .= cdf_b_cond_k_initial[1,i_k,i_y]
                # wealth beyond the highest observed policy (conditional on k) would imply liquid savings beyond m-grid
                # -> CDF = max(CDF|k_j)
                cdf_values[idx2:end] .= 1.0 * cdf_b_cond_k_initial[end,i_k,i_y]
                cdf_values[idx1+1:idx2-1] .= intp.(m[idx1+1:idx2-1])
                #return cdf_values
            end
            m_to_cdf_spline_extr!(cdf_b_cond_k_given_y_k,n_par.grid_m)
            cdf_b_cond_k_given_y_k[1] = isnothing(i_mmin) ? cdf_b_cond_k_initial[1,i_k,i_y] : cdf_b_cond_k_initial[i_mmin,i_k,i_y]
            # normalize cdf_b_cond_k_given_y_k
            cdf_b_cond_k_given_y_k .= min.(cdf_b_cond_k_given_y_k,cdfend)
            cdf_b_cond_k_given_y_k[end] = cdfend
        end
    end
end


