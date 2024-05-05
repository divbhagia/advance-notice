##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp

# Import custom functions
from utils.simfuncs import dgp, sim_data
from utils.customplot import custom_plot
from utils.esthelpers import unstack
from utils.estgmm import gmm
from utils.dataclean import group_dur

# Seed
np.random.seed(1118)

##########################################################
# Define function to simulate data & estimate model
##########################################################

def SimOnce(n, T, J, dgpopt, nrm, ffopt, interval, seed):
    np.random.seed(seed)
    data = sim_data(n, T, J, dgpopt)
    data['dur'] = group_dur(data['dur'], interval)
    T_new = len(data['dur'].unique())
    nL = data['notice'].value_counts().sort_index().values
    thta_hat = gmm(data, nrm, ffopt)[0]
    psiM, _ = unstack(T_new-1, J, thta_hat, nrm, ffopt)
    psi_hat = psiM @ nL/n
    return psi_hat

def SimMulti(n, T, J, dgpopt, nrm, ffopt, interval, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(SimOnce, 
                     [(n, T, J, dgpopt, nrm, ffopt, interval, seed) 
                      for seed in seeds])
    pool.close()
    return r

##########################################################
# Main program
##########################################################

if __name__ == '__main__':

    # DGP
    iters = 10
    T, J = 8, 2
    n = 100000
    dgpopt = 'no_obs' 
    psiMtr, mu, nuP, betaL, betaPhi, x_means, cov_x1to3, pL = dgp(T, J, dgpopt)
    psi_true = psiMtr @ pL
    nrm = mu[0]
    interval = 2

    # Grouped true psi
    durvals = np.arange(0, T)
    grpd_durvals = group_dur(durvals, interval)
    new_durvals = np.unique(grpd_durvals)
    psi_grpd = np.zeros(len(new_durvals))
    for i, t in enumerate(new_durvals):
        idx = grpd_durvals == t
        psi_grpd[i] = psi_true[idx].sum()
    psi_grpd = psi_grpd[:-1]

    # Simulate data
    seeds = np.random.choice(100000, iters)
    r = SimMulti(n, T, J, dgpopt, nrm, 'np', interval, seeds)
    psi_avg = np.mean(r, axis=0)

    # Plot
    nrmlz = True
    if nrmlz:
        psi_grpd = psi_grpd/psi_grpd[0]
        psi_avg = psi_avg/psi_avg[0]
    custom_plot([psi_grpd, psi_avg])

##########################################################


    
