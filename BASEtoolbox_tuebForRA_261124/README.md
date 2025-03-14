# How to run conditional method (without wealth-grid in iteration)

The main algorithm is in *fcn_directtransition_conditionals.jl*, in the functions
- DirectTransition_Splines!
- DirectTransition_Splines_adjusters!
- DirectTransition_Splines_non_adjusters!

This follows the overleaf file (base version). The code for **Alternative B**, where wealth grid is in iteration, is not here. It runs slower and in my experiments yields a very similar distribution. You can check that yourself: In Output/Saves, all solution-files with a *_wit* at the end of the name are outcomes of the **Alternative B**-algorithm.

The function *DirectTransition_Splines!* needs also inputs from EGM. For this, *EGM_policyupdate* returns some objects, most importantly *m_a_aux*.

There are two use-cases: 
1. compute the steady-state distribution for given asset returns/household policies, or 
2. compute the steady-state, where asset returns have to be consistent with aggregate capital.

For 1., run the file *test_script_0806.jl*. In the beginning of the script, the household policies are computed for a given capital stock/asset return. Here, K=25.04 corresponds to that found in the steady state computed with the histogram method and 250x250 grids. Then, auxiliary variables as inputs for the algorithm are computed. Then, the time of the two main algorithms, adjusters and non-adjusters, is taken. Finally, the iteration takes place, where the convergence threshold is set at \epsilon=1.0e-13, and the result is saved.

For 2., run *script.jl*, which calls *find_steadystate_splines*. The functions for finding the steady state, found in *fcn_kdiff.jl* and *fcn_ksupply.jl*, are already adapted to be able to run the conditional method for finding the distribution.

# Final note on "speedup"
In function *DirectTransition_Splines_adjusters*, a "speedup" is preselected. It refers to the way the wealth-distribution is computed from the initial conditional distribution *F_{b|k}* that is given. For finding the probabilities of total wealth levels, one has to evaluate *F_{b|k}* at non-gridpoints. The "speedup" refers to using linear interpolation of the cdf to do that evaluation. Without speedup, instead, one uses a cubic spline for this. I found in my experiments that the linear interpolation is much faster, while the results barely differ. Note that this step is only about constructing the initial wealth-distribution; no optimal policy choices enter. Hence, using a linear interpolation here should not make the method subject to the Bhandari et al-critique (thinking already ahead about how to implement the perturbation).
