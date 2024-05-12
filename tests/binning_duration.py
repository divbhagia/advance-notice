import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils.simdgp import dgp
from utils.dataclean import group_dur
from utils.esthelpers import model_moms, unstack, numgrad
from utils.customplot import set_plot_aes, custom_plot
from utils.config import Colors, TEST_QUANTS_DIR, SIMBIN_AGAIN
np.random.seed(1118)
set_plot_aes()

######################################################
# Simplified functions to estimate the model using h 
######################################################

def objfun(thta, h_data, nrm, ffopt='np'):
    T, J = h_data.shape
    h_model = model_moms(*unstack(T, J, thta, nrm, ffopt)[:2])
    m = (h_data - h_model).reshape(T*J, 1).flatten()
    obj = m.T @ m 
    return obj

def numgrad_wrapper(x, *args):
    return numgrad(objfun, x, *args).flatten()

def gmm(h_data, nrm, ffopt='np'):
    T, J = h_data.shape
    thta0 = 0.25*np.ones(2 * (T-1) + J)
    opts = {'disp': False, 'maxiter': 100000}
    results = minimize(objfun, thta0, method= 'BFGS', tol=1e-32,
                       args=(h_data, nrm, ffopt), options=opts, 
                       jac=numgrad_wrapper)
    psiM, _ = unstack(T, J, results.x, nrm, ffopt)
    return psiM

######################################################
# Functions to get binned quantities
######################################################

def bin_hazard(h, interval):
    S = np.insert(np.cumprod((1 - h), axis=0), 0, 1, axis=0)
    S_bin = S[::interval,:]
    h_bin = ((S_bin[:-1,:] - S_bin[1:,:]) / S_bin[:-1])
    return h_bin

def bin_psi(psi, interval):
    T = len(psi)
    durvals = np.arange(0, T)
    grpd_durvals = group_dur(durvals, interval)
    new_durvals = np.unique(grpd_durvals)
    psi_bin = np.zeros(len(new_durvals))
    for i, t in enumerate(new_durvals):
        idx_ = grpd_durvals == t
        psi_bin[i] = dgpqnts['psi'][idx_].sum()
    return psi_bin    

######################################################
# Main program
######################################################

# DGP parameters
T, J = 12, 2
psin = np.array([0.1, 0.2, 0.3, 0.5])[:J]
betaL = np.array([1, 1])
betaP = np.array([0, 0]) 
psiopts = ['nm', 'inc', 'dec', 'cons']

# Initialize vectors
intervals = [1, 2, 3, 4]
psi_hats = [np.zeros((T, len(intervals))) for i in range(len(psiopts))]
h_obs = [np.zeros(T) for i in range(len(psiopts))]

# Estimate model for different psiopts and intervals
if SIMBIN_AGAIN:
    for i in range(len(psiopts)):
        dgpqnts = dgp(T, psin, psiopts[i], betaL, betaP)
        psiM, mu, psi = dgpqnts['psiM'], dgpqnts['mu'], dgpqnts['psi']
        nrm = dgpqnts['mu'][0]
        h_model = model_moms(psiM, mu)
        for j in range(len(intervals)):
            h_bin = bin_hazard(h_model, intervals[j])
            psiM_hat = gmm(h_bin, nrm, ffopt='np')
            psi_hat = psiM_hat @ dgpqnts['pL']
            psi_hat = np.repeat(psi_hat, intervals[j])/intervals[j]
            psi_hats[i][:,j] = psi_hat
        h_obs[i] = h_model @ dgpqnts['pL']
    np.save(f'{TEST_QUANTS_DIR}/sim_binning_psi.npy', psi_hats)
    np.save(f'{TEST_QUANTS_DIR}/sim_binning_h.npy', h_obs)
else:
    psi_hats = np.load(f'{TEST_QUANTS_DIR}/sim_binning_psi.npy')
    h_obs = np.load(f'{TEST_QUANTS_DIR}/sim_binning_h.npy')

# Plot Aesthetics
ylims = [(-0.05, 0.65), (0, 0.45), (0.02, 0.17), (0.04, 0.22)]
ydists = [0.1, 0.05, 0.02, 0.02]
yticks = []
for i in range(len(ylims)):
    yticks.append(np.arange(ylims[i][0], ylims[i][1], ydists[i]))
titles = ['Panel A: Non-monotonic', 
          'Panel B: Increasing',
          'Panel C: Decreasing',
          'Panel D: Constant']
colors = [Colors.BLACK, Colors.BLUE, Colors.GREEN, Colors.RED]
linestyles = ['-', '-.', ':', '-']
xticklabs = np.arange(1, T+1, 1)
legendlabs = [f'Bin Size: {i}' for i in intervals]

# Normalize quantities
h_obs = [h_obs[i] * psi_hats[i][0][0] 
         / h_obs[i][0] for i in range(len(psiopts))]
nrmlz = False
if nrmlz:
    for i in range(len(psiopts)):
        for j in range(len(intervals)):
            psi_hats[i][:,j] = psi_hats[i][:,j] / psi_hats[i][j][0] \
                             * h_obs[i][0]

# Plot
plt.figure(figsize=(8, 8))
for i in range(len(psiopts)):
    plt.subplot(2, 2, i+1)
    custom_plot([psi_hats[i][:,j] for j in range(len(intervals))],
                subplot=True, title=titles[i], colors=colors,
                legend = False, linestyles=linestyles, ylims=ylims[i],
                xticklabs=xticklabs, ydist=ydists[i], legendlabs=legendlabs)
    plt.plot(h_obs[i], color=Colors.BLACK, linestyle=':', linewidth=2.25,
             label='Observed')
    plt.yticks(yticks[i])
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.25)
plt.legend(bbox_to_anchor=(0, -0.3), ncol=5, loc='lower center',
           alignment= 'center', 
           handlelength=3, columnspacing=3, fontsize=11)
plt.savefig('tests/quants/sim_binning.pdf', bbox_inches='tight',
            dpi = 300, format = 'pdf')

######################################################