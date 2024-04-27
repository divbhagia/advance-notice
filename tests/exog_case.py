##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import matplotlib.pyplot as plt

# Import custom functions
from utils.SimFunctions import DGP, SimData
from utils.DataDesc import CustomPlot
from utils.DataMoms import DataMoms
from utils.EstHelpers import ModelMoms, Unstack, UnstackPsiM, NumGrad
from utils.EstGMM import AvgMoms, ObjFunc, GMM
from utils.Inference import IndvMoms, AvgMomsInference, OptimalWeightMat, StdErrors

# from utils.Inference import IndvMoms, StdErrors
# from utils.EstNuisancePars import PredPS, RegAdj

# Seed
np.random.seed(1118)

##########################################################
# DGP and data simulation
##########################################################

# DGP
T, J = 8, 3
n = 100000
dgp_opt = 'no_obs' 
psiM, mu, nuP, betaL, betaPhi, x_means, cov_x1to3 = DGP(T, J, dgp_opt)
nrm = mu[0]

# Simulate data
print(f'Simulating data for n={n} with T={T}, J={J} and DGP={dgp_opt}...')
data, nu, pL_X, phiX = SimData(n, T+1, J, dgp_opt, out='all')
nL = np.sum(pL_X, axis=0)
psi = psiM @ nL/n

##########################################################
# Compare model & data moments
##########################################################

# Raw data moments
h, exit, surv, exit_i, surv_i = DataMoms(data)
for var in ['h', 'exit', 'surv', 'exit_i', 'surv_i']:
    exec(f"{var} = {var}['raw']")
h_avg = h @ nL/n

# Compare data and model moments
h_model, exit_mdl, surv_mdl = ModelMoms(psiM, mu, out='all')
CustomPlot([h[:,0], h_model[:,0]], legendlabs=['Data', 'Model'], 
           ydist=0.1, ylims=[0,0.4])

##########################################################
# Check other GMM related functions
##########################################################

# True parameter values
thta = np.concatenate([psiM[0,:].flatten(), mu[1:].flatten(),
                    psiM[1:,0].flatten()])

# GMM moments and objective function at true parameters
m = AvgMoms(thta, exit, surv, nrm)
obj = ObjFunc(thta, exit, surv, nrm)
dobj_dx = NumGrad(ObjFunc, thta, exit, surv, nrm)
dm_dx = NumGrad(AvgMoms, thta, exit, surv, nrm)
print(f'Objective function at true parameters: {obj:.6f}')
print(f'Number of moments (T={T}, J={J}): T x J = {T*J}')
print(f'Number of parameters: (2 * (T-1) + J) = {2 * (T-1) + J}')
print(f'Shape of gradient for objective function: {dobj_dx.shape}')
print(f'Shape of Jacobian for moments: {dm_dx.shape}')

# Estimate the model using true moments
thta_hat = GMM(exit_mdl, surv_mdl, nrm=nrm)
psiM_hat, mu_hat = Unstack(T, J, thta_hat, nrm)

# Inference related functions
m_i = IndvMoms(thta, data, nrm)
m_chk = AvgMomsInference(thta, data, nrm, MomsFunc=IndvMoms)
W = OptimalWeightMat(thta, data, nrm, MomsFunc=IndvMoms)
print(f'Shape of individual moments: {m_i.shape}')
print(f'Shape of optimal weight matrix: {W.shape}')

# Calculate & plot standard errors
se = StdErrors(thta_hat, data, nrm, MomsFunc=IndvMoms)
psiSE, muSE = Unstack(T, J, se, nrm)
psi_hat, psin_hat, psiSE, psinSE = UnstackPsiM(nL, psiM, psiSE)

# Plot estimates with confidence intervals
plt.figure(figsize=[3, 2.5])
plt.errorbar(np.arange(0, T), psi_hat, yerr=1.96*psiSE,
              color='red', capsize=2, label='Est True Moms', alpha=0.75)
plt.plot(psi, label='True', color='black', linestyle='--')
plt.legend()
plt.show()

##########################################################
# Estimation and inference on the true data
##########################################################

# Estimate the model on simulated data
print('Estimating the model on simulated data...')
thta_hat = GMM(exit, surv, nrm=nrm)
psiM_hat, mu_hat = Unstack(T, J, thta_hat, nrm)

# Standard errors
print('Calculating standard errors...')
se = StdErrors(thta_hat, data, nrm, MomsFunc=IndvMoms)
psiSE, muSE = Unstack(T, J, se, nrm)

# Unstack estimates 
psi_hat, psin_hat, psiSE, psinSE = UnstackPsiM(nL, psiM_hat, psiSE)

# Plot estimates with confidence intervals
plt.figure(figsize=[3, 2.5])
plt.errorbar(np.arange(0, T), psi_hat, yerr=1.96*psiSE,
              color='red', capsize=2, label='Estimate', alpha=0.75)
plt.plot(psi, label='True', color='black', linestyle='--')
plt.legend()
plt.show()

##########################################################
