##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import matplotlib.pyplot as plt

# Import custom functions
from utils.simfuncs import dgp, sim_data
from utils.customplot import custom_plot
from utils.datamoms import data_moms
from utils.esthelpers import model_moms, unstack, unstack_all
from utils.estgmm import gmm, estimate
from utils.inference import std_errs, indv_moms
import numpy as np

# Seed
np.random.seed(1118)

##########################################################
# DGP and data simulation
##########################################################

# DGP
T, J = 4, 2
n = 100000
dgp_opt = 'no_obs' 
psiMtr, mu, nuP, betaL, betaPhi, x_means, cov_x1to3, pL = dgp(T, J, dgp_opt)
nrm = mu[0]

# Simulate data
print(f'Simulating data for n={n} with T={T}, J={J} and DGP={dgp_opt}...')
data, nu, pL_X, phiX = sim_data(n, T+1, J, dgp_opt, out='all')
nL = np.sum(pL_X, axis=0)
psi_true = psiMtr @ nL/n

##########################################################
# Compare model & data moments
##########################################################

# Raw data moments
h, *_ = data_moms(data, purpose='output')
h_avg = h @ nL/n

# Compare data and model moments
h_model, *_ = model_moms(psiMtr, mu, out='all')
custom_plot([h[:,0], h_model[:,0]], legendlabs=['Data', 'Model'])

##########################################################
# Estimation and inference on the true data
##########################################################

# True parameter values
thta = np.concatenate([psiMtr[0,:].flatten(), mu[1:].flatten(),
                    psiMtr[1:,0].flatten()])

# Estimate using GMM
ffopt = 'np'
print('Estimating the model on simulated data...')
thta_hat = gmm(data, nrm, ffopt)[0]
psiM_hat, mu_hat = unstack(T, J, thta_hat, nrm, ffopt)
se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
        unstack_all(T, J, nL, thta_hat, se, nrm, ffopt)

# Plot estimates with confidence intervals
plt.figure(figsize=[3, 2.5])
plt.errorbar(np.arange(0, T), psi, yerr=1.96*psiSE,
              color='red', capsize=2, label='Estimate', alpha=0.75)
plt.plot(psi_true, label='True', color='black', linestyle='--')
plt.legend()
plt.show()

# Or directly
r = estimate(data, nrm, ffopt)
plt.figure(figsize=[3, 2.5])
plt.errorbar(np.arange(0, T), r['psi'], yerr=1.96*r['psiSE'],
              color='red', capsize=2, label='Estimate', alpha=0.75)
plt.plot(psi_true, label='True', color='black', linestyle='--')
plt.legend()
plt.show()

##########################################################
# Extension 
##########################################################

gamma = np.ones((T-1, J))
kappa = np.zeros((T, J))

K = 50; lim = [0.75, 1.25];
par = np.linspace(lim[0], lim[1], K)
thta_list, Jstat_list = [], []
for k in range(K):
    ffopt = {'opt': 'baseline', 'gamma': gamma, 'kappa': kappa}
    ffopt['gamma'][:, 1] = par[k]
    thta_hat, Jstat = gmm(data, nrm, ffopt, print_=False)
    thta_list.append(thta_hat)
    Jstat_list.append(Jstat)

# Find index of the best model
Jstat_list = np.array(Jstat_list)
best = np.argmin(Jstat_list)
print(f'Best model: {par[best]}')

# Plot distance
plt.figure(figsize=[3, 2.5])
plt.plot(par, Jstat_list, color='black', linestyle='--')
plt.axvline(par[best], color='red', linestyle='--')

    
