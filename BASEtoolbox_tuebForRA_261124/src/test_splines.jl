
using Plots
sr_full=ss_full_young
RB=exp(sr_full.XSS[sr_full.indexes.RBSS])
RK=exp(sr_full.XSS[sr_full.indexes.rSS])


total_wealth = Array{eltype(0.0)}(undef, sr_full.n_par.nk .* sr_full.n_par.nm)
for k = 1:sr_full.n_par.nk
    for m = 1:sr_full.n_par.nm
        total_wealth[m+(k-1)*sr_full.n_par.nm] = RB .* sr_full.n_par.grid_m[m] .+ RK .* sr_full.n_par.grid_k[k] .+ m_par.Rbar .* sr_full.n_par.grid_m[m] .* (sr_full.n_par.grid_m[m] .< 0)
    end
end

IX = sortperm(total_wealth)
total_wealth = total_wealth[IX]



cdf_initial=sr_full.distrSS
cdf_initial_m=cumsum(cdf_initial, dims=1)
cdf_initial=cumsum(cumsum(cdf_initial, dims=1), dims=2)
cdf_old = copy(cdf_initial)

cdf_prime_given_y = cdf_initial[:, :, 1]
pdf_prime_given_y = BASEforHANK.SteadyState.cdf_to_pdf(cdf_prime_given_y)[:]
cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])
# cdf_prime_totalwealth_given_y_2 = cdf_initial[:, :, 1][IX]

cdf_prime_m = cdf_prime_given_y[:, end]
pdf_prime_m = BASEforHANK.SteadyState.cdf_to_pdf(cdf_prime_m)[:]
cdf_prime_cond_m = cdf_prime_given_y ./ cdf_prime_m

cdf_prime_k = cdf_prime_given_y[end, :]
pdf_prime_k = BASEforHANK.SteadyState.cdf_to_pdf(cdf_prime_k)
cdf_prime_cond_k = cdf_prime_given_y ./ cdf_prime_k'

function cutF(F, x, grid)
    y = map(x) do xi
        if xi < grid[1]
            0.0
        elseif xi > grid[end]
            1.0
        else
            F(xi)
        end
    end
    return y
end


Fwk = zeros(size(total_wealth)   )
Fwm = zeros(size(total_wealth)   )

for i_k = 1:sr_full.n_par.nk
    # Fk = BASEforHANK.SteadyState.Interpolator(sr_full.n_par.grid_k, cdf_prime_cond_m[i_m, :])
    # Fw += cutF(Fk, (total_wealth .- RB.*sr_full.n_par.grid_m[i_m])./RK,sr_full.n_par) .* pdf_prime_m[i_m]
    Fm = BASEforHANK.SteadyState.Interpolator(sr_full.n_par.grid_k, cdf_prime_cond_k[:, i_k])
    Fwk += cutF(Fm, (total_wealth .- RK.*sr_full.n_par.grid_k[i_k])./RB,sr_full.n_par.grid_m) .* pdf_prime_k[i_k]
end
for i_m = 1:sr_full.n_par.nm
    Fk = BASEforHANK.SteadyState.Interpolator(sr_full.n_par.grid_k, cdf_prime_cond_m[i_m, :])
    Fwm += cutF(Fk, (total_wealth .- RB.*sr_full.n_par.grid_m[i_m])./RK,sr_full.n_par.grid_k) .* pdf_prime_m[i_m]
    # Fm = BASEforHANK.SteadyState.Interpolator(sr_full.n_par.grid_k, cdf_prime_cond_k[:, i_k])
    # Fw += cutF(Fm, (total_wealth .- RK.*sr_full.n_par.grid_k[i_k])./RB,sr_full.n_par) .* pdf_prime_k[i_k]
end

n = size(cdf_initial)
cdf_old_Pi = reshape(reshape(cdf_old, (n[1] .* n[2], n[3])) * sr_full.n_par.Π, (n[1], n[2], n[3]))
cdf_prime_given_y = cdf_old_Pi[:, :, 1]

pdf_prime_given_y = BASEforHANK.Tools.cdf_to_pdf(cdf_prime_given_y)[:]

cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

# S_a, T_a, W_a, S_n, T_n, W_n =
# BASEforHANK.SteadyState.MakeTransition(sr_full.m_a_star, sr_full.m_n_star, sr_full.k_a_star, sr_full.n_par.Π, sr_full.n_par)

# TransitionMat_a = BASEforHANK.SteadyState.sparse(
# S_a,
# T_a,
# W_a,
# sr_full.n_par.nm * sr_full.n_par.nk * sr_full.n_par.ny,
# sr_full.n_par.nm * sr_full.n_par.nk * sr_full.n_par.ny,
# )

# distr_young_a = real.(BASEforHANK.SteadyState.eigsolve(TransitionMat_a', sr_full.distrSS[:], 1)[2][1])
# distr_young_a = (reshape((distr_young_a[:]) ./ sum((distr_young_a[:])), (sr_full.n_par.nm, sr_full.n_par.nk, sr_full.n_par.ny)))

# distr_young_a=reshape(distr_young_a ,sr_full.n_par.nm, sr_full.n_par.nk, sr_full.n_par.ny)
# cdf_initial=distr_young_a
# cdf_initial_m=cumsum(cdf_initial, dims=1)
# cdf_initial=cumsum(cumsum(cdf_initial, dims=1), dims=2)
# cdf_old = copy(cdf_initial)

countw=0
max_iter=100
while countw < max_iter
    countw = countw + 1
    cdf_old = copy(cdf_initial)

cdf_initial = BASEforHANK.DirectTransition_Splines!(cdf_initial,sr_full.m_n_star, sr_full.m_a_star, sr_full.k_a_star, copy(cdf_initial), sr_full.n_par.Π, RB, RK, sr_full.n_par,sr_full.m_par)
difference = cdf_old .- cdf_initial
distance = maximum(abs, difference)
println("Distribution Iterations: ", countw)
println("Distribution Dist: ", distance)
end

XX=[cdf_initial[:,:,1][IX] total_wealth sr_full.m_a_star[IX][:,:,1][:] sr_full.k_a_star[IX][:,:,1][:] sr_full.m_a_star[IX][:,:,1][:].+sr_full.k_a_star[IX][:,:,1][:] cdf_prime_totalwealth_given_y]


function expected_value(cdf::AbstractArray, grid)

    cdf_splines = BASEforHANK.SteadyState.Interpolator(grid, cdf)

    right_part = BASEforHANK.SteadyState.integrate(cdf_splines, grid[1], grid[end])
    left_part = grid[end] * cdf_splines(grid[end]) - grid[1] * cdf_splines(grid[1])

    EV = left_part - right_part

    return EV
end

K = expected_value(sum(cdf_initial[end, :, :],dims=2)[:], sr_full.n_par.grid_k)
B = expected_value(sum(cdf_initial[:, end, :],dims=2)[:], sr_full.n_par.grid_m)

pdf_splines = BASEforHANK.Tools.cdf_to_pdf(cumsum(cdf_initial,dims=3))

m_pdf_splines = sum(pdf_splines, dims = (2,3))[:]
m_pdf_hist = sum(sr_full.distrSS, dims = (2,3))[:]

k_pdf_splines = sum(pdf_splines, dims = (1,3))[:]
k_pdf_hist = sum(sr_full.distrSS, dims = (1,3))[:]

y_pdf_splines = sum(pdf_splines, dims = (1,2))[:]
y_pdf_hist = sum(sr_full.distrSS, dims = (1,2))[:]

n_par = sr_full.n_par

total_wealth = zeros( n_par.nk .* n_par.nm)
for k = 1:n_par.nk
    for m = 1:n_par.nm
        total_wealth[m+(k-1)*n_par.nm] = RB.*n_par.grid_m[m] .+  RK.*n_par.grid_k[k]
    end
end

IX = sortperm(total_wealth)
total_wealth = total_wealth[IX]

m_a_prime_grid = sr_full.m_a_star
k_a_prime_grid = sr_full.k_a_star



cdf_prime_on_grid_a = similar(cdf_initial)


i_y = 1


    cdf_prime_given_y = cdf_initial[:, :, i_y]
    println(size(cdf_initial))

    pdf_prime_given_y = BASEforHANK.cdf_to_pdf(cdf_prime_given_y)[:]


    cdf_prime_totalwealth_given_y = cumsum(pdf_prime_given_y[IX])

    m_a_prime_grid_totalwealth_given_y = m_a_prime_grid[:, :, i_y][IX]
    k_a_prime_grid_totalwealth_given_y = k_a_prime_grid[:, :, i_y][IX]

    idx_last_at_constraint_k = findlast(k_a_prime_grid_totalwealth_given_y .== n_par.grid_k[1])
    idx_last_on_grid_k = min.(findlast(k_a_prime_grid_totalwealth_given_y .< n_par.grid_k[end]) .+ 1, n_par.nk*n_par.nm)

    idx_last_at_constraint_m = findlast(m_a_prime_grid_totalwealth_given_y .== n_par.grid_m[1])
    idx_last_on_grid_m = min.(findlast(m_a_prime_grid_totalwealth_given_y .< n_par.grid_m[end]) .+ 1, n_par.nm*n_par.nk)

    # Start interpolation from last unique value (= last value at the constraint)
    if isnothing(idx_last_at_constraint_k)
        k_to_cdf_spline = BASEforHANK.Interpolator(k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k], cdf_prime_totalwealth_given_y[1:idx_last_on_grid_k])
        k_to_m_spline = BASEforHANK.Interpolator(k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k], m_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_k])

    else
        k_to_cdf_spline = BASEforHANK.Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k],
        cdf_prime_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k])
        k_to_m_spline = BASEforHANK.Interpolator(k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k],
        m_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_k:idx_last_on_grid_k])
    end
    if isnothing(idx_last_at_constraint_m)
        m_to_k_spline = BASEforHANK.Interpolator(m_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_m], k_a_prime_grid_totalwealth_given_y[1:idx_last_on_grid_m])
    else
        m_to_k_spline = BASEforHANK.Interpolator(m_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_m:idx_last_on_grid_m], k_a_prime_grid_totalwealth_given_y[idx_last_at_constraint_m:idx_last_on_grid_m])
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

    for i_k = 1:n_par.nk
        for i_m = 1:n_par.nm
            if n_par.grid_m[i_m] > k_to_m_spline(n_par.grid_k[i_k])
                cdf_prime_on_grid_a[i_m, i_k, i_y] = k_to_cdf_spline_extr.(n_par.grid_k[i_k])
            else
                cdf_prime_on_grid_a[i_m, i_k, i_y] = k_to_cdf_spline_extr.(m_to_k_spline(n_par.grid_m[i_m]))
            end
        end
    end


    



cdf_prime_on_grid_a .= cdf_prime_on_grid_a * m_par.λ # Adjusters

# cdf_prime_on_grid_n = similar(cdf_prime_on_grid)

