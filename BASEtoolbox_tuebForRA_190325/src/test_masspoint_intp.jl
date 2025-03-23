using PCHIPInterpolation, Plots

grid = collect(1:0.01:5)
masspoints = [1.5, 2.5, 3.5, 4.5]
cdf = similar(grid)
cdf .= 0.0
mass_mps = 1.0 / length(masspoints)
for mp in masspoints
    ix = findfirst(grid .>= mp)
    cdf[ix:end] .= cdf[ix] + mass_mps
end

rand_nodes = unique(rand(1:length(grid),100))
sorting = sortperm(grid[rand_nodes])

cdf_intp = Interpolator(grid[rand_nodes][sorting], cdf[rand_nodes][sorting])

plot(grid, cdf, label="CDF", lw=2)
plot!(grid, cdf_intp.(grid), label="Interpolated CDF", lw=2)
savefig("intp_stepwise.png")