##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np

# Import custom functions
from utils.datadesc import custom_plot, pred_ps
from utils.simfuncs import dgp, sim_data, moms_fewvars
from utils.datamoms import data_moms
from utils.esthelpers import model_moms, unstack_all
from utils.estgmm import gmm, estimate
from utils.inference import indv_moms_ipw, indv_moms, std_errs

# Other options
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1118)

##########################################################
# Parameters and checks
##########################################################

# DGP
T, J = 4, 2
n = 1000000
dgpopt = 'fewvars'
psiMtr, mu_tr, nuP, betaL, betaPhi, xmeans, covx1to3, pL = dgp(T, J, dgpopt)
psi_true = psiMtr @ pL

# Average nu-moments for the DGP
avg_moms_X = moms_fewvars(T, pull_prev=True)

# Model moments
h_model, exit_mdl, surv_mdl = model_moms(psiMtr, avg_moms_X, out='all')

# Simulate data
data, nu, pL_X, phiX = sim_data(n, T+1, J, dgpopt, _print=True, out='all')
nL = data['notice'].value_counts().sort_index().values

##########################################################
# STEP 1: Predict propensity scores & estimate hazard
##########################################################

# Propensity scores & adjusted & unadjusted moments
ps, coefs = pred_ps(data)
h, h_se, *_ = data_moms(data, ps, purpose='output')
h_unadj, *_ = data_moms(data, purpose='output')

# (Check) Compare model and data moments
custom_plot([h_model[:,1], h[:,1]], legendlabs=['Model', 'Data'])

# Plot data moments
custom_plot([h[:, j] for j in range(J)], 
            [h_se[:, j] for j in range(J)])

##########################################################
# STEP 2: Estimation and Inference
##########################################################

# Specify options
nrm = avg_moms_X[0]
ffopt = 'np'
se_adj = True

# Estimate
thta_hat = gmm(data, nrm, ffopt, ps)

# Inference
thta_all = np.append(thta_hat, coefs)
if se_adj:
    se, Jtest = std_errs(thta_all, data, nrm, ffopt, MomsFunc=indv_moms_ipw)
    se = se[:len(thta_hat)]
else:
    se, Jtest = std_errs(thta_hat, data, nrm, ffopt, ps, MomsFunc=indv_moms)

# Unstack standard errors and parameters
psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
    unstack_all(T, J, nL, thta_hat, se, nrm, ffopt)

##########################################################
# Alternatively Step 1 & 2 together
##########################################################

direct = False
if direct:
    r = estimate(data, nrm, ffopt, adj='ipw')

##########################################################
# Plot
##########################################################

# Normalize observed hazard for plotting
h_avg = h @ nL/nL.sum()
h_avg = psi[0] * h_avg/h_avg[0]

# Plot structural hazard
custom_plot([psi_true, psi, h_avg], [None, psiSE, None],
           legendlabs=['True', 'Est', 'Data'], 
           xlab='Duration', ylab='Hazard Rate')

# Estimated moments (mu)
custom_plot([avg_moms_X, mu], [None, None], legendlabs=['True', 'Est'], 
           xlab='Duration', ylab='$E[\\phi(X)^dE(\\nu^d|X)]$', ydist=0.3)

##########################################################