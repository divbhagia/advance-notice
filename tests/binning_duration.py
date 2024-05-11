import numpy as np
import multiprocessing as mp
from utils.esthelpers import model_moms, unstack
from utils.dataclean import group_dur
from utils.simdgp import dgp
from utils.simdata import sim_data
from utils.customplot import custom_plot
from utils.estgmm import gmm
import matplotlib.pyplot as plt
np.random.seed(1118)

# Fix seed so that results are reproducible

######################################################
# Functions to simulate and estimate 
######################################################

def sim_once(n, dgpqnts, interval, seed=118):
    np.random.seed(seed)
    data = sim_data(n, dgpqnts)[0]
    data['dur'] = group_dur(data['dur'], interval)
    nrm = dgpqnts['mu'][0]
    nL = data['notice'].value_counts().sort_index().values
    J = len(nL)
    T = len(data['dur'].unique())
    thta_hat = gmm(data, nrm, ffopt='np')[0]
    psiM = unstack(T-1, J, thta_hat, nrm, ffopt='np')[0]
    psi_hat = psiM @ nL/n
    return psi_hat

def sim_multi(n, dgpqnts, interval, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(sim_once, [(n, dgpqnts, interval, seed) 
                      for seed in seeds])
    pool.close()
    return r

######################################################
# Main program
######################################################

if __name__ == '__main__':

    # DGP parameters
    n = 50000
    iters = 100
    T, J = 10, 2
    interval = 2
    psin = np.array([0.15, 0.4, 0.6, 0.8])[:J]
    betaL = np.array([1, 1])
    betaP = np.array([0, 0]) 
    psiopt = 'nm'
    print('Getting DGP quantities...')
    dgpqnts = dgp(T, psin, psiopt, betaL, betaP, interval)
    psiM, mu = dgpqnts['psiM'], dgpqnts['mu']

    # Grouped true psi
    durvals = np.arange(0, T)
    grpd_durvals = group_dur(durvals, interval)
    new_durvals = np.unique(grpd_durvals)
    #psi_bind = np.zeros(len(new_durvals))
    print(f'Grouping data into {len(new_durvals)} intervals...')
    # for i, t in enumerate(new_durvals):
    #     idx_ = grpd_durvals == t
    #     psi_bind[i] = dgpqnts['psi'][idx_].sum()
    # psi_bind = psi_bind[:-1]

    # DGP moments
    custom_plot([dgpqnts['psi']*dgpqnts['mu'][0], 
                 dgpqnts['h_str'], dgpqnts['h_obs']],
                legendlabs=['$\\psi(d)\\mu_1$', '$h^{str}$', '$h^{obs}$'],
                title='DGP moments', ylims=[0.0, 0.15])
    
    # DGP moments binned
    custom_plot([dgpqnts['h_str_bin'], dgpqnts['h_str'], 
                 dgpqnts['h_obs_bin'], dgpqnts['h_obs']],
                legendlabs=['$h^{str}_{bin}$', '$h^{str}$',
                            '$h^{obs}_{bin}$', '$h^{obs}$'],
                title='DGP moments', ylims=[0.0, 0.2])
    plt.legend(loc='upper left', bbox_to_anchor=(1,1))

    # Estimate
    print('Running simulations...')
    adj = 'none'
    seeds = np.random.randint(0, 100000, iters)
    psi_ests = sim_multi(n, dgpqnts, interval, seeds)
    psi_hat = np.mean(psi_ests, axis=0)
    print(f'Estimated psi: {psi_hat}')

    # Plot
    nrmlz = True
    if nrmlz:
        psi_hat = psi_hat/psi_hat[0]
        dgpqnts['h_str_bin'] = dgpqnts['h_str_bin']/dgpqnts['h_str_bin'][0]
        dgpqnts['h_str'] = dgpqnts['h_str']/dgpqnts['h_str'][0]
    custom_plot([psi_hat, dgpqnts['h_str_bin']], 
                legendlabs=['Estimated', '$h^{str}_{bin}$'])

######################################################