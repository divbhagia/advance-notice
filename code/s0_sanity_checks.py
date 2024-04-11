# Housekeeping
import sys
sys.path.append('code')
import numpy as np
import pandas as pd
from utils.SimFns import DGP, SimData
from utils.DataFns import DurDistByNotice, CustomPlot
from utils.DDML import DDML, ImpliedMoms, IPW, RegAdj
from utils.GMM import Unstack, GMM, ModelMoments
import matplotlib.pyplot as plt
import multiprocessing as mp
np.random.seed(1118)

##########################################################
# Simple case with notice exogenous to nu
##########################################################

# DGP
T, J = 4, 2
n = 100000
dgp_opt = 'exog' 
psiM, mu, nuP, betaL, betaPhi, x_means, cov_x1to3 = DGP(T, J, dgp_opt)

# Simulate data
data = SimData(n, T+1, J, dgp_opt)
data, nu, pL_X, phiX = SimData(n, T+1, J, dgp_opt, out='all')

# Compare data and model moments
g_data = ImpliedMoms(data)[0]['raw']
g_model = ModelMoments(psiM, mu)

# Plot to compare model and data moments
labs = ['Data Moments', 'Model Moments']
CustomPlot([g_data[:,0], g_model[:,0]], legendlabs=labs)

# Estimate the model directly using gmm
nrm = mu[0]
psiM_gmm, mu_gmm = GMM(g_data, nrm, unstack=True)

# Implement DDML 
#psiM_hat, mu_hat, ps, h_i = DDML(data, nrm=nrm)
#g, h, S = ImpliedMoms(data, ps, h_i)

# Plot estimates vs true 
series1 = [psiM[1:,0], psiM_gmm[1:,0]]
series2 = [mu, mu_gmm]
labs = ['True', 'Estimate']
CustomPlot(series1, legendlabs=labs, title='Structural Hazard')
CustomPlot(series2, legendlabs=labs, title='Moments of nu', ydist=1)

##########################################################
# Try DDML with DGP: few_vars
##########################################################

# DGP
np.random.seed(1118)
T, J = 4, 2
n = 500000
dgp_opt = 'fewvars'
psiM, mu, nuP, betaL, betaPhi, x_means, cov_x1to3 = DGP(T, J, dgp_opt)
nrm = mu[0]

# Simulate data
data = SimData(n, T, J, dgp_opt, print_=True)

# DDML Parameters
model_ps = 'logit'
model_ra = 'logit'
remove_sparse = True
not_X_vars = ['dur', 'cens', 'notice', 'cens_ind']
X = data[[col for col in data.columns if col not in not_X_vars]]

if remove_sparse:
    X = X.loc[:, (betaL[:,1] != 0)]
    data = pd.concat([data[not_X_vars], X], axis=1)

# Estimate with nfolds=1
psiM_hat1, mu_hat, ps, h_i = DDML(data, model_ps, model_ra, nrm=nrm) # nfold=1
ps_ = IPW(data, model_ps)
h_i_ = RegAdj(data, model_ra)
g1 = ImpliedMoms(data, ps, h_i)[0]
psiM_hat2 = {x: GMM(g1[x], nrm, unstack=True)[0] for x in g1.keys()}

# Estimate with nfolds=2
folds = np.random.choice(2, len(data))
psiM_hat3, mu_hat, ps, h_i = DDML(data, model_ps, model_ra, folds, nrm)
ps_ = IPW(data, model_ps, folds)
h_i_ = RegAdj(data, model_ra, folds)
g2 = ImpliedMoms(data, ps, h_i)[0]
psiM_hat4 = {x: GMM(g2[x], nrm, unstack=True)[0] for x in g2.keys()}

# Normalize
psiM_nrm = psiM[1:,0]/psiM[1,0]
psiM_hat1_nrm = {x: psiM_hat1[x][1:,0]/psiM_hat1[x][1,0] for x in psiM_hat1.keys()}
psiM_hat2_nrm = {x: psiM_hat2[x][1:,0]/psiM_hat2[x][1,0] for x in psiM_hat2.keys()}
psiM_hat3_nrm = {x: psiM_hat3[x][1:,0]/psiM_hat3[x][1,0] for x in psiM_hat3.keys()}
psiM_hat4_nrm = {x: psiM_hat4[x][1:,0]/psiM_hat4[x][1,0] for x in psiM_hat4.keys()}

# Plot (Note: est 4 is same as 1 and 2 because g1 and g2 should be the same)
for key in ['dr', 'ipw', 'ra']:
    plt.figure(figsize=[3, 2.5])
    plt.plot(psiM_hat1_nrm[key], label='est1')
    plt.plot(psiM_hat2_nrm[key], label='est2', linestyle='--')
    plt.plot(psiM_hat3_nrm[key], label='est3', linestyle='-.')
    plt.plot(psiM_hat4_nrm[key], label='est4', linestyle=':')
    plt.plot(psiM_nrm, label='True', color='black')
    plt.title(key)
    plt.legend()
    plt.show()

# Compare DR, IPW, and RA Estimates 
# Note: DR should be closer to IPW if model_ps = logit & dgp_opt=default
plt.figure(figsize=[3, 2.5])
plt.plot(psiM_hat1_nrm['dr'], label='dr')
plt.plot(psiM_hat1_nrm['ipw'], label='ipw', linestyle='--')
plt.plot(psiM_hat1_nrm['ra'], label='ra', linestyle='-.')
plt.plot(psiM_nrm, label='True', color='black')
plt.legend()
plt.show()

##########################################################