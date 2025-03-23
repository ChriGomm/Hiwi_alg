using JLD2, PCHIPInterpolation, ForwardDiff, Plots, CodecZlib
cd("./src")
JLD2.@load "Output/Saves/ssKfixed_80x1600.jld2" ss
ss80 = ss;
JLD2.@load "Output/Saves/ssKfixed_80x729_wit.jld2" ss
ss80_wit = ss;
JLD2.@load "Output/Saves/ssKfixed_100x2500_wit.jld2" ss
ss100_wit = ss;
@load "Output/Saves/ssKfixed_100x10000.jld2" ss
ss100 = ss;
@load "Output/Saves/ssKfixed_240x3600.jld2" ss
ss240 = ss;
@load "Output/Saves/ssKfixed_240x3600_wit.jld2" ss
ss240_wit = ss;
@load "Output/Saves/ssKfixed_250x2500.jld2" ss
ss250 = ss;
@load "Output/Saves/ssKfixed_500x10000.jld2" ss
ss500 = ss;

JLD2.@load "Output/Saves/young250x250.jld2" ss_full_young
ss250y = ss_full_young;

cdfsky = Dict{String,Array{Float64,2}}()
cdfsby = Dict{String,Array{Float64,2}}()
for ssy in [ss250y]
    distr_m = reshape(sum(ssy.distrSS, dims = 2),ssy.n_par.nm,ssy.n_par.ny)
    distr_k = reshape(sum(ssy.distrSS, dims = 1),ssy.n_par.nk,ssy.n_par.ny)
    cdfsky[string(ssy.n_par.nm)] = cumsum(distr_k,dims=1)
    cdfsby[string(ssy.n_par.nm)] = cumsum(distr_m,dims=1)
end
for i_y = 1:ss80.n_par.ny
    pdf = ss80.distrSS[end,end,i_y]
    k_distrs = plot(title="Distribution over k, y=$i_y",xlabel="k",ylabel="CDF",size=(400,300),xlims=(0,2000),legend=:bottomright)
    KSS = ceil(ss80.KSS;digits=2)
    plot!(k_distrs,ss80.n_par.grid_k,ss80.distrSS[end,:,i_y]/pdf,label="80x80x1600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss80_wit.KSS;digits=2)
    plot!(k_distrs,ss80_wit.n_par.grid_k,ss80_wit.distrSS[end,:,i_y]/pdf,label="80x80x729 (wit,K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss100_wit.KSS;digits=2)
    plot!(k_distrs,ss100_wit.n_par.grid_k,ss100_wit.distrSS[end,:,i_y]/pdf,label="100x100x2500 (wit,K=$KSS)",linewidth=1.5, linestyle=:solid)
    KSS = ceil(ss240.KSS;digits=2)
    plot!(k_distrs,ss240.n_par.grid_k,ss240.distrSS[end,:,i_y]/pdf,label="240x240x3600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss240_wit.KSS;digits=2)
    plot!(k_distrs,ss240_wit.n_par.grid_k,ss240_wit.distrSS[end,:,i_y]/pdf,label="240x240x3600 (wit,K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    # KSS = ceil(ss100.KSS;digits=2)
    # plot!(k_distrs,ss100.n_par.grid_k,ss100.distrSS[end,:,i_y]/pdf,label="100x100x10000 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    KSS = ceil(ss250.KSS;digits=2)
    plot!(k_distrs,ss250.n_par.grid_k,ss250.distrSS[end,:,i_y]/pdf,label="250x250x2500 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss500.KSS;digits=2)
    plot!(k_distrs,ss500.n_par.grid_k,ss500.distrSS[end,:,i_y]/pdf,label="500x500x10000 (K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss250y.KSS;digits=2)
    plot!(k_distrs,ss250y.n_par.grid_k,cdfsky["250"][:,i_y]/pdf,label="Hstgr250x250 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(k_distrs,"Output/Figures/k_distrs_y$i_y.pdf")

    k_distrs = plot(title="Distribution over k, y=$i_y",xlabel="k",ylabel="CDF",size=(400,300),xlims=(0,150),legend=:bottomright)
    KSS = ceil(ss80.KSS;digits=2)
    plot!(k_distrs,ss80.n_par.grid_k,ss80.distrSS[end,:,i_y]/pdf,label="80x80x1600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss80_wit.KSS;digits=2)
    plot!(k_distrs,ss80_wit.n_par.grid_k,ss80_wit.distrSS[end,:,i_y]/pdf,label="80x80x729 (wit,K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss100_wit.KSS;digits=2)
    plot!(k_distrs,ss100_wit.n_par.grid_k,ss100_wit.distrSS[end,:,i_y]/pdf,label="100x100x2500 (wit,K=$KSS)",linewidth=1.5, linestyle=:solid)
    KSS = ceil(ss240.KSS;digits=2)
    plot!(k_distrs,ss240.n_par.grid_k,ss240.distrSS[end,:,i_y]/pdf,label="240x240x3600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss240_wit.KSS;digits=2)
    plot!(k_distrs,ss240_wit.n_par.grid_k,ss240_wit.distrSS[end,:,i_y]/pdf,label="240x240x3600 (wit,K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    # KSS = ceil(ss100.KSS;digits=2)
    # plot!(k_distrs,ss100.n_par.grid_k,ss100.distrSS[end,:,i_y]/pdf,label="100x100x10000 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    KSS = ceil(ss250.KSS;digits=2)
    plot!(k_distrs,ss250.n_par.grid_k,ss250.distrSS[end,:,i_y]/pdf,label="250x250x2500 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss500.KSS;digits=2)
    plot!(k_distrs,ss500.n_par.grid_k,ss500.distrSS[end,:,i_y]/pdf,label="500x500x10000 (K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss250y.KSS;digits=2)
    plot!(k_distrs,ss250y.n_par.grid_k,cdfsky["250"][:,i_y]/pdf,label="Hstgr250x250 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(k_distrs,"Output/Figures/k_small_y$i_y.pdf")

    k_distrs = plot(title="Distribution over k, y=$i_y",xlabel="k",ylabel="CDF",size=(400,300),xlims=(0,1))
    KSS = ceil(ss80.KSS;digits=2)
    plot!(k_distrs,ss80.n_par.grid_k,ss80.distrSS[end,:,i_y]/pdf,label="80x80x1600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss80_wit.KSS;digits=2)
    plot!(k_distrs,ss80_wit.n_par.grid_k,ss80_wit.distrSS[end,:,i_y]/pdf,label="80x80x729 (wit,K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss100_wit.KSS;digits=2)
    plot!(k_distrs,ss100_wit.n_par.grid_k,ss100_wit.distrSS[end,:,i_y]/pdf,label="100x100x2500 (wit,K=$KSS)",linewidth=1.5, linestyle=:solid)
    KSS = ceil(ss240.KSS;digits=2)
    plot!(k_distrs,ss240.n_par.grid_k,ss240.distrSS[end,:,i_y]/pdf,label="240x240x3600 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss240_wit.KSS;digits=2)
    plot!(k_distrs,ss240_wit.n_par.grid_k,ss240_wit.distrSS[end,:,i_y]/pdf,label="240x240x3600 (wit,K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    # KSS = ceil(ss100.KSS;digits=2)
    # plot!(k_distrs,ss100.n_par.grid_k,ss100.distrSS[end,:,i_y]/pdf,label="100x100x10000 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    KSS = ceil(ss250.KSS;digits=2)
    plot!(k_distrs,ss250.n_par.grid_k,ss250.distrSS[end,:,i_y]/pdf,label="250x250x2500 (K=$KSS)",linewidth=1.5, linestyle=:dash)
    KSS = ceil(ss500.KSS;digits=2)
    plot!(k_distrs,ss500.n_par.grid_k,ss500.distrSS[end,:,i_y]/pdf,label="500x500x10000 (K=$KSS)",linewidth=1.5, linestyle=:dot)
    KSS = ceil(ss250y.KSS;digits=2)
    plot!(k_distrs,ss250y.n_par.grid_k,cdfsky["250"][:,i_y]/pdf,label="Hstgr250x250 (K=$KSS)",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(k_distrs,"Output/Figures/k_null_y$i_y.pdf")
end

cdfbs = Dict{Int,Array{Float64,2}}()
for (i_ss,ss) in enumerate([ss80,ss80_wit,ss100_wit,ss100,ss240,ss240_wit,ss250,ss500])
    global cdfbs
    n_par = ss.n_par;
    distr_initial = ss.distrSS;
    cdf_b = NaN*ones(n_par.nm,n_par.ny)
    for i_y in 1:n_par.ny
        diffcdfk = diff(distr_initial[end,:,i_y],dims=1)/distr_initial[end,end,i_y]
        for i_b = 1:n_par.nm
            for i_k = 1:n_par.nk
                cdf_b[i_b,i_y] = distr_initial[end,1,i_y]/distr_initial[end,end,i_y]*distr_initial[i_b,1,i_y] + .5*sum((distr_initial[i_b,2:end,i_y] .+ distr_initial[i_b,1:end-1,i_y]).*diffcdfk)
            end
        end
    end
    cdfbs[i_ss] = cdf_b
end
for i_y = 1:ss80.n_par.ny
    pdf = ss80.distrSS[end,end,i_y]
    b_distrs = plot(title ="Distribution over b, y=$i_y",xlabel="b",ylabel="CDF",size=(400,300),xlims=(0,500),legend=:bottomright)
    plot!(b_distrs,ss80.n_par.grid_m,cdfbs[1][:,i_y]/pdf,label="80x80x1600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss80_wit.n_par.grid_m,cdfbs[2][:,i_y]/pdf,label="80x80x729 (wit)",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss100_wit.n_par.grid_m,cdfbs[3][:,i_y]/pdf,label="100x100x2500 (wit)",linewidth=1.5, linestyle=:dashdot)
    #plot!(b_distrs,ss100.n_par.grid_m,cdfbs[4][:,i_y]/pdf,label="100x100x10000",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240.n_par.grid_m,cdfbs[5][:,i_y]/pdf,label="240x240x3600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240_wit.n_par.grid_m,cdfbs[6][:,i_y]/pdf,label="240x240x3600 (wit)",linewidth=1.5, linestyle=:dashdot)
    plot!(b_distrs,ss250.n_par.grid_m,cdfbs[7][:,i_y]/pdf,label="250x250x2500",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss500.n_par.grid_m,cdfbs[8][:,i_y]/pdf,label="500x500x10000",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss250y.n_par.grid_m,cdfsby["250"][:,i_y]/pdf,label="Hstgr250x250",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(b_distrs,"Output/Figures/b_distrs_y$i_y.pdf")

    pdf = ss80.distrSS[end,end,i_y]
    b_distrs = plot(title ="Distribution over b, y=$i_y",xlabel="b",ylabel="CDF",size=(400,300),xlims=(0,150))
    plot!(b_distrs,ss80.n_par.grid_m,cdfbs[1][:,i_y]/pdf,label="80x80x1600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss80_wit.n_par.grid_m,cdfbs[2][:,i_y]/pdf,label="80x80x729 (wit)",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss100_wit.n_par.grid_m,cdfbs[3][:,i_y]/pdf,label="100x100x2500 (wit)",linewidth=1.5, linestyle=:dashdot)
    #plot!(b_distrs,ss100.n_par.grid_m,cdfbs[4][:,i_y]/pdf,label="100x100x10000",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240.n_par.grid_m,cdfbs[5][:,i_y]/pdf,label="240x240x3600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240_wit.n_par.grid_m,cdfbs[6][:,i_y]/pdf,label="240x240x3600 (wit)",linewidth=1.5, linestyle=:dashdot)
    plot!(b_distrs,ss250.n_par.grid_m,cdfbs[7][:,i_y]/pdf,label="250x250x2500",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss500.n_par.grid_m,cdfbs[8][:,i_y]/pdf,label="500x500x10000",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss250y.n_par.grid_m,cdfsby["250"][:,i_y]/pdf,label="Hstgr250x250",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(b_distrs,"Output/Figures/b_small_y$i_y.pdf")


    b_distrs = plot(title ="Distribution over b, y=$i_y",xlabel="b",ylabel="CDF",size=(400,300),xlims=(0,1))
    plot!(b_distrs,ss80.n_par.grid_m,cdfbs[1][:,i_y]/pdf,label="80x80x1600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss80_wit.n_par.grid_m,cdfbs[2][:,i_y]/pdf,label="80x80x729 (wit)",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss100_wit.n_par.grid_m,cdfbs[3][:,i_y]/pdf,label="100x100x2500 (wit)",linewidth=1.5, linestyle=:dashdot)
    #plot!(b_distrs,ss100.n_par.grid_m,cdfbs[4][:,i_y]/pdf,label="100x100x10000",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240.n_par.grid_m,cdfbs[5][:,i_y]/pdf,label="240x240x3600",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss240_wit.n_par.grid_m,cdfbs[6][:,i_y]/pdf,label="240x240x3600 (wit)",linewidth=1.5, linestyle=:dashdot)
    plot!(b_distrs,ss250.n_par.grid_m,cdfbs[7][:,i_y]/pdf,label="250x250x2500",linewidth=1.5, linestyle=:dash)
    plot!(b_distrs,ss500.n_par.grid_m,cdfbs[8][:,i_y]/pdf,label="500x500x10000",linewidth=1.5, linestyle=:dot)
    plot!(b_distrs,ss250y.n_par.grid_m,cdfsby["250"][:,i_y]/pdf,label="Hstgr250x250",linewidth=1.5, linestyle=:dashdot)
    Plots.savefig(b_distrs,"Output/Figures/b_null_y$i_y.pdf")
end
