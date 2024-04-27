##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np

# Import custom functions
from utils.DataDesc import CustomPlot, PredPS
from utils.SimFunctions import DGP, SimData, MomsFewVarsDGP
from utils.DataMoms import DataMoms
from utils.EstHelpers import ModelMoms, UnstackAll
from utils.EstGMM import GMM, Estimate
from utils.Inference import IndvMoms, IndvMomsIPW, StdErrors

# Other options
np.set_printoptions(precision=4, suppress=True)
np.random.seed(1118)

##########################################################
# Parameters and checks
##########################################################

# DGP
T, J = 6, 2
n = 500000
dgpopt = 'fewvars'
psiMtr, mu_tr, nuP, betaL, betaPhi, xmeans, covx1to3, pL = DGP(T, J, dgpopt)
psi_true = psiMtr @ pL

# Average nu-moments for the DGP
avg_moms_X = MomsFewVarsDGP(T)

# Model moments
h_model, exit_mdl, surv_mdl = ModelMoms(psiMtr, avg_moms_X, out='all')

# Simulate data
data, nu, pL_X, phiX = SimData(n, T+1, J, dgpopt, _print=True, out='all')
nL = data['notice'].value_counts().sort_index().values

##########################################################
# STEP 1: Predict propensity scores & estimate hazard
##########################################################

# Propensity scores & adjusted & unadjusted moments
ps, coefs = PredPS(data)
h, h_se, *_ = DataMoms(data, ps, purpose='output')
h_unadj, *_ = DataMoms(data, purpose='output')

# (Check) Compare model and data moments
CustomPlot([h_model[:,1], h[:,1]], legendlabs=['Model', 'Data'])

# Plot data moments
CustomPlot([h[:, j] for j in range(J)], 
            [h_se[:, j] for j in range(J)])

##########################################################
# STEP 2: Estimation and Inference
##########################################################

# Specify options
nrm = avg_moms_X[0]
ffopt = 'baseline'
se_adj = True

# Estimate
thta_hat = GMM(data, nrm, ffopt, ps)

# Inference
thta_all = np.append(thta_hat, coefs)
if se_adj:
    se = StdErrors(thta_all, data, nrm, ffopt, MomsFunc=IndvMomsIPW)
    se = se[:len(thta_hat)]
else:
    se = StdErrors(thta_hat, data, nrm, ffopt, MomsFunc=IndvMoms)

# Unstack standard errors and parameters
psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
    UnstackAll(T, J, nL, thta_hat, se, nrm, ffopt)

##########################################################
# Alternatively Step 1 & 2 together
##########################################################

direct = False
if direct:
    r = Estimate(data, nrm, ffopt, adj='ipw')

##########################################################
# Plot
##########################################################

# Normalize observed hazard for plotting
h_avg = h @ nL/nL.sum()
h_avg = psi[0] * h_avg/h_avg[0]

# Plot structural hazard
CustomPlot([psi_true, psi, h_avg], [None, psiSE, None],
           legendlabs=['True', 'Est', 'Data'], 
           xlab='Duration', ylab='Hazard Rate')

# Estimated moments (mu)
CustomPlot([avg_moms_X, mu], [None, None], legendlabs=['True', 'Est'], 
           xlab='Duration', ylab='$E[\\phi(X)^dE(\\nu^d|X)]$', ydist=0.3)

##########################################################