#------------------------------------------------------------------------------
# THIS FILE CONTAINS THE "AGGREGATE" MODEL EQUATIONS, I.E. EVERYTHING  BUT THE 
# HOUSEHOLD PLANNING PROBLEM. THE lATTER IS DESCRIBED BY ONE EGM BACKWARD STEP AND 
# ONE FORWARD ITERATION OF THE DISTRIBUTION.
#
# AGGREGATE EQUATIONS TAKE THE FORM 
# F[EQUATION NUMBER] = lhs - rhs
#
# EQUATION NUMBERS ARE GENEREATED AUTOMATICALLY AND STORED IN THE INDEX STRUCT
# FOR THIS THE "CORRESPONDING" VARIABLE NEEDS TO BE IN THE LIST OF STATES 
# OR CONTROLS.
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# AUXILIARY VARIABLES ARE DEFINED FIRST
#------------------------------------------------------------------------------
ιΠ = (1.0 ./ 40.0 - 1.0 ./ 800.0) .* m_par.shiftΠ .+ 1.0 ./ 800.0
ωΠ = ιΠ ./ m_par.ιΠ .* m_par.ωΠ

# Elasticities and steepness from target markups for Phillips Curves
η = μ / (μ - 1.0)                                 # demand elasticity
κ = η * (m_par.κ / m_par.μ) * (m_par.μ - 1.0)     # implied steepness of phillips curve
ηw = μw / (μw - 1.0)                               # demand elasticity wages
κw = ηw * (m_par.κw / m_par.μw) * (m_par.μw - 1.0) # implied steepness of wage phillips curve

# Capital Utilization
MPKSS = exp(XSS[indexes.rSS]) - 1.0 + m_par.δ_0       # stationary equil. marginal productivity of capital
δ_1 = MPKSS                                        # normailzation of utilization to 1 in stationary equilibrium
δ_2 = δ_1 .* m_par.δ_s                              # express second utilization coefficient in relative terms
# Auxiliary variables
Kserv = K * u                                         # Effective capital
MPKserv = interest(Kserv, mc.*Z, N, m_par) .+ m_par.δ_0 # mc .* Z .* m_par.α .* (Kserv ./ N) .^ (m_par.α - 1.0)      # marginal product of Capital
depr = m_par.δ_0 + δ_1 * (u - 1.0) + δ_2 / 2.0 * (u - 1.0)^2.0   # depreciation

Wagesum = N * w                                         # Total wages in economy t
WagesumPrime = NPrime * wPrime                               # Total wages in economy t+1

YREACTION = Ygrowth                                  # Policy reaction function to Y

distr_y = sum(distrSS, dims = (1, 2))

# tax progressivity variabels used to calculate e.g. total taxes
tax_prog_scale = (m_par.γ + m_par.τprog ) / ((m_par.γ + τprog))                        # scaling of labor disutility including tax progressivity
incgross = ((n_par.grid_y ./ n_par.H) .^ tax_prog_scale .* mcw .* w .* N ./ Ht)  # capital liquidation Income (q=1 in steady state)
incgross[end] = (n_par.grid_y[end] .* profits)                         # gross profit income
inc = τlev .* (incgross .^ (1.0 .- τprog))                                 # capital liquidation Income (q=1 in steady state)
taxrev = incgross .- inc                                                 # tax revenues

TaxAux = dot(distr_y, taxrev)
IncAux = dot(distr_y, incgross)

Htact = dot(
    distr_y[1:end-1],
    (n_par.grid_y[1:end-1] / n_par.H) .^ ((m_par.γ + m_par.τprog ) / (m_par.γ + τprog)),
)
############################################################################
#           Error term calculations (i.e. model starts here)          #
############################################################################

#-------- States -----------#
# Error Term on exogeneous States
# Shock processes
F[indexes.Gshock]       = log.(GshockPrime) - m_par.ρ_Gshock * log.(Gshock)     # primary deficit shock
F[indexes.Tprogshock]   = log.(TprogshockPrime) - m_par.ρ_Pshock * log.(Tprogshock) # tax shock

F[indexes.Rshock]       = log.(RshockPrime) - m_par.ρ_Rshock * log.(Rshock)     # Taylor rule shock
F[indexes.Sshock]       = log.(SshockPrime) - m_par.ρ_Sshock * log.(Sshock)     # uncertainty shock

# Stochastic states that can be directly moved (no feedback)
F[indexes.A]            = log.(APrime) - m_par.ρ_A * log.(A)                # (unobserved) Private bond return fed-funds spread (produces goods out of nothing if negative)
F[indexes.Z]            = log.(ZPrime) - m_par.ρ_Z * log.(Z)                # TFP
F[indexes.ZI]           = log.(ZIPrime) - m_par.ρ_ZI * log.(ZI)             # Investment-good productivity

F[indexes.μ]            = log.(μPrime ./ m_par.μ) - m_par.ρ_μ * log.(μ ./ m_par.μ)      # Process for markup target
F[indexes.μw]           = log.(μwPrime ./ m_par.μw) - m_par.ρ_μw * log.(μw ./ m_par.μw)   # Process for w-markup target

# Endogeneous States (including Lags)
F[indexes.σ] =
    log.(σPrime) -
    (m_par.ρ_s * log.(σ) + (1.0 - m_par.ρ_s) * m_par.Σ_n * log(Ygrowth) + log(Sshock))                     # Idiosyncratic income risk (contemporaneous reaction to business cycle)

F[indexes.Ylag] = log(YlagPrime) - log(Y)
F[indexes.Bgovlag] = log(BgovlagPrime) - log(Bgov)
F[indexes.Ilag] = log(IlagPrime) - log(I)
F[indexes.wlag] = log(wlagPrime) - log(w)
F[indexes.Tlag] = log(TlagPrime) - log(T)
F[indexes.qlag] = log(qlagPrime) - log(q)
F[indexes.Clag] = log(ClagPrime) - log(C)
F[indexes.av_tax_ratelag] = log(av_tax_ratelagPrime) - log(av_tax_rate)
F[indexes.τproglag] = log(τproglagPrime) - log(τprog)
F[indexes.qΠlag] = log(qΠlagPrime) - log(qΠ)

# Growth rates
F[indexes.Ygrowth] = log(Ygrowth) - log(Y / Ylag)
F[indexes.Tgrowth] = log(Tgrowth) - log(T / Tlag)
F[indexes.Bgovgrowth] = log(Bgovgrowth) - log(Bgov / Bgovlag)
F[indexes.Igrowth] = log(Igrowth) - log(I / Ilag)
F[indexes.wgrowth] = log(wgrowth) - log(w / wlag)
F[indexes.Cgrowth] = log(Cgrowth) - log(C / Clag)

#  Taylor rule and interest rates
F[indexes.RB] =
    log(RBPrime) - XSS[indexes.RBSS] - ((1 - m_par.ρ_R) * m_par.θ_π) .* log(π) -
    ((1 - m_par.ρ_R) * m_par.θ_Y) .* log(YREACTION) -
    m_par.ρ_R * (log.(RB) - XSS[indexes.RBSS]) - log(Rshock)

# Tax rule
F[indexes.τprog] =
    log(τprog) - m_par.ρ_P * log(τproglag) - (1.0 - m_par.ρ_P) * (XSS[indexes.τprogSS]) -
    (1.0 - m_par.ρ_P) * m_par.γ_YP * log(YREACTION) -
    (1.0 - m_par.ρ_P) * m_par.γ_BP * (log(Bgov) - XSS[indexes.BgovSS]) - log(Tprogshock)


F[indexes.τlev] = av_tax_rate - TaxAux ./ IncAux  # Union profits are taxed at average tax rate
F[indexes.T] = log(T) - log(TaxAux + av_tax_rate * unionprofits)


F[indexes.av_tax_rate] =
    log(av_tax_rate) - m_par.ρ_τ * log(av_tax_ratelag) -
    (1.0 - m_par.ρ_τ) * XSS[indexes.av_tax_rateSS] -
    (1.0 - m_par.ρ_τ) * m_par.γ_Yτ * log(YREACTION) -
    (1.0 - m_par.ρ_τ) * m_par.γ_Bτ * (log(Bgov) - log(Bgovlag))#XSS[indexes.BgovSS])

# --------- Controls ------------
# Deficit rule
F[indexes.π] =
    log(BgovgrowthPrime) + m_par.γ_B * (log(Bgov) - XSS[indexes.BgovSS]) -
    m_par.γ_Y * log(YREACTION) - m_par.γ_π * log(π) - log(Gshock)

F[indexes.G] = log(G) - log(BgovPrime + T - RB / π * Bgov)             # Government Budget Constraint

# Phillips Curve to determine equilibrium markup, output, factor incomes 
F[indexes.mc] =
    (log.(π) - XSS[indexes.πSS]) - κ * (mc - 1 ./ μ) -
    m_par.β * ((log.(πPrime) - XSS[indexes.πSS]) .* YPrime ./ Y)

# Wage Phillips Curve 
F[indexes.mcw] =
    (log.(πw) - XSS[indexes.πwSS]) - (
        κw * (mcw - 1 ./ μw) +
        m_par.β * ((log.(πwPrime) - XSS[indexes.πwSS]) .* WagesumPrime ./ Wagesum)
    )
# worker's wage = mcw * firm's wage

# Wage Dynamics
F[indexes.πw] = log.(w ./ wlag) - log.(πw ./ π)                   # Definition of real wage inflation

# Capital utilisation
F[indexes.u] = MPKserv - q * (δ_1 + δ_2 * (u - 1.0))           # Optimality condition for utilization

# Prices
F[indexes.r] = log.(r) - log.(1 + MPKserv * u - q * depr)       # rate of return on capital

F[indexes.mcww] = log.(mcww) - log.(mcw * w)                        # wages that workers receive

F[indexes.w] = log.(w) - log.(wage(Kserv, Z * mc, N, m_par))     # wages that firms pay

F[indexes.unionprofits] = log.(unionprofits) - log.(w .* N .* (1.0 - mcw))  # profits of the monopolistic unions

# firm_profits: price setting profits + investment profits. The latter are zero and do not show up up to first order (K'-(1-δ)K = I).
F[indexes.firm_profits] =
    log.(firm_profits) - log.(Y .* (1.0 - mc) .+ q .* (KPrime .- (1.0 .- depr) .* K) .- I)
F[indexes.profits] = log.(profits) - log.((1.0 .- ωΠ) .* firm_profits .+ ιΠ .* (qΠ .- 1.0)) # distributed profits to entrepreneurs
F[indexes.qΠ] =
    log.(RBPrime ./ πPrime) .-
    log.(((qΠPrime .- 1.0) .* (1 - ιΠ) .+ ωΠ .* firm_profitsPrime) ./ (qΠ .- 1.0))

F[indexes.RL] =
    log.(RL) -
    log.((RB .* Bgov .+ π .* ((qΠ .- 1.0) .* (1 - ιΠ) .+ ωΠ .* firm_profits)) ./ B)

F[indexes.Bgov] = log.(B) - log.(Bgov + (qΠlag .- 1.0))                                 # total liquidity demand


F[indexes.q] =
    1.0 -
    ZI *
    q *
    (1.0 - m_par.ϕ / 2.0 * (Igrowth - 1.0)^2.0 - # price of capital investment adjustment costs
     m_par.ϕ * (Igrowth - 1.0) * Igrowth) -
    m_par.β * ZIPrime * qPrime * m_par.ϕ * (IgrowthPrime - 1.0) * (IgrowthPrime)^2.0

# Asset market premia
F[indexes.LP] = log.(LP) - (log((q + r - 1.0) / qlag) - log(RB / π))                   # Ex-post liquidity premium           
F[indexes.LPXA] = log.(LPXA) - (log((qPrime + rPrime - 1.0) / q) - log(RBPrime / πPrime))  # ex-ante liquidity premium

# Aggregate Quantities
F[indexes.I] =
    KPrime .- K .* (1.0 .- depr) .-
    ZI .* I .* (1.0 .- m_par.ϕ ./ 2.0 .* (Igrowth - 1.0) .^ 2.0)           # Capital accumulation equation

F[indexes.N] = log.(N) - log.(labor_supply(w.*mcw, τlev, τprog, Ht, m_par))
    # log.(
    #     ((1.0 - τprog) * τlev * (mcw .* w) .^ (1.0 - τprog)) .^ (1.0 / (m_par.γ + τprog)) .*
    #     Ht
    # )   # labor supply
F[indexes.Y] = log.(Y) - log.(output(Kserv, Z, N, m_par))    #log.(Z .* N .^ (1.0 .- m_par.α) .* Kserv .^ m_par.α) # Z .* N .^ (1.0 .- m_par.α) .* Kserv .^ m_par.α                                      # production function
F[indexes.C] = log.(Y .- G .- I  .+ (A .- 1.0) .* RL .* B ./ π) .- log(C)                            # Resource constraint

# Error Term on prices/aggregate summary vars (logarithmic, controls), here difference to SS value averages
F[indexes.BY] = log.(BY) - log.(B / Y)                                                               # Bond to Output ratio
F[indexes.TY] = log.(TY) - log.(T / Y)                                                               # Tax to output ratio

# Distribution summary statistics used in this file (using the steady state distrubtion in case). 
# Lines here generate a unit derivative (distributional summaries do not change with other aggregate vars).
F[indexes.K] = log.(K) - XSS[indexes.KSS]                                                        # Capital market clearing
F[indexes.B] = log.(B) - XSS[indexes.BSS]                                                        # Bond market clearing

# Add distributional summary stats that do change with other aggregate controls/prices and with estimated parameters
F[indexes.Ht] = log.(Ht) - log.(Htact)

# other distributional statistics not used in other aggregate equations and not changing with parameters, 
# but potentially with other aggregate variables are NOT included here. They are found in FSYS.
