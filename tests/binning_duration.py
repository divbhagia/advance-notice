import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utils.simdgp import dgp
from utils.dataclean import group_dur
from utils.esthelpers import model_moms, unstack, numgrad
from utils.customplot import set_plot_aes, custom_plot
from utils.config import Colors, TEST_QUANTS_DIR
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
psi_hats = [[], [], [], []]

# Estimate model for different psiopts and intervals
for i in range(len(psiopts)):
    dgpqnts = dgp(T, psin, psiopts[i], betaL, betaP)
    psiM, mu, psi = dgpqnts['psiM'], dgpqnts['mu'], dgpqnts['psi']
    nrm = dgpqnts['mu'][0]
    h_model = model_moms(psiM, mu)
    for j in range(len(intervals)):
        h_bin = bin_hazard(h_model, intervals[j])
        psiM_hat = gmm(h_bin, nrm, ffopt='np')
        psi_hats[i].append(psiM_hat @ dgpqnts['pL'])
        psi_hats[i][j] = np.repeat(psi_hats[i][j], intervals[j])/intervals[j]
np.save(f'{TEST_QUANTS_DIR}/sim_binning.npy', psi_hats)

# Plot
ylims = [(0.09, 0.325), (0.09, 0.325), (0.06, 0.17), (0.125, 0.175)]
titles = ['Panel A: Non-monotonic', 
          'Panel B: Increasing',
          'Panel C: Decreasing',
          'Panel D: Constant']
colors = [Colors.BLACK, Colors.BLUE, Colors.GREEN, Colors.RED]
linestyles = ['-', ':', '-.', '-']
linewidths = [1, 2, 1, 1]
plt.figure(figsize=(8, 8))
for i in range(len(psiopts)):
    plt.subplot(2, 2, i+1)
    custom_plot([psi_hats[i][j] for j in range(len(intervals))],
                subplot=True, title=titles[i], colors=colors,
                legend = False, linestyles=linestyles, legend=False,)
        # plt.plot(psi_hats[i][j], label=f'{intervals[j]}', 
        #          color=colors[j], linestyle=linestyles[j],
        #          linewidth=linewidths[j])
        # plt.title(titles[i])
        # plt.yticks(np.arange(0, 1, 0.05))
        # plt.ylim(ylims[i])
plt.tight_layout()
plt.legend(bbox_to_anchor=(0, -0.3), ncol=4, loc='lower center',
           title = 'Interval Size:', alignment= 'center', 
           handlelength=3, columnspacing=5)
plt.savefig('tests/sim_binning.pdf', bbox_inches='tight',
            dpi = 300, format = 'pdf')

######################################################