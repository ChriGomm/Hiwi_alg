  ############################################################################
    # Prepare elements used for uncompression
    ############################################################################
    # Matrices to take care of reduced degree of freedom in marginal distributions
    Γ = BASEforHANK.PerturbationSolution.shuffleMatrix(sr_full.distrSS, sr_full.n_par)
    # Matrices for discrete cosine transforms
    DC = Array{Array{Float64,2},1}(undef, 3)
    DC[1] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.nm)
    DC[2] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.nk)
    DC[3] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.ny)
    IDC = [DC[1]', DC[2]', DC[3]']

    DCD = Array{Array{Float64,2},1}(undef, 3)
    DCD[1] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.nm_copula)
    DCD[2] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.nk_copula)
    DCD[3] = BASEforHANK.PerturbationSolution.mydctmx(sr_full.n_par.ny_copula)
    IDCD = [DCD[1]', DCD[2]', DCD[3]']

    ############################################################################
    # Check whether Steady state solves the difference equation
    ############################################################################
    length_X0 = sr_full.n_par.ntotal
    X0 = zeros(length_X0) .+ BASEforHANK.PerturbationSolution.ForwardDiff.Dual(0.0, 0.0)
    F = BASEforHANK.PerturbationSolution.Fsys(
        X0,
        X0,
        sr_full.XSS,
        m_par,
        sr_full.n_par,
        sr_full.indexes,
        Γ,
        sr_full.compressionIndexes,
        DC,
        IDC,
        DCD,
        IDCD,
    )

    FR = BASEforHANK.PerturbationSolution.realpart.(F)
    println(findall(abs.(FR) .> 0.001))
    println("Number of States and Controls")
    println(length(F))
    println("Max error on Fsys:")
    println(maximum(abs.(FR[:])))
    println("Max error of COP in Fsys:")
    println(maximum(abs.(FR[sr_full.indexes.COP])))
    println("Max error of Vm in Fsys:")
    println(maximum(abs.(FR[sr_full.indexes.Vm])))

    println("Max error of Vk in Fsys:")
    println(maximum(abs.(FR[sr_full.indexes.Vk])))
    FR[sr_full.indexes.distr_m]