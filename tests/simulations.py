
##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import multiprocessing as mp

# Import custom functions
from utils.simdgp import dgp
from utils.simdata import sim_data
from utils.customplot import custom_plot
from utils.estgmm import estimate

# Seed
np.random.seed(1118)

##########################################################
# Functions to simulate and estimate
##########################################################

def sim_once(n, dgpqnts, seed):
    np.random.seed(seed)
    data = sim_data(n, dgpqnts)[0]
    nrm = dgpqnts['mu'][0]
    r = estimate(data, nrm, ffopt='np', adj='ipw')
    return r['psi']

def sim_multi(n, dgpqnts, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(sim_once, [(n, dgpqnts, seed) for seed in seeds])
    return r

##########################################################
# DGP and simulations
##########################################################

if __name__ == '__main__':

    # DGP parameters
    n = 50000
    T, J = 6, 2
    iters = 25
    psin = np.array([0.15, 0.4, 0.6, 0.8])[:J]
    betaL = np.array([15, 5])
    betaP = np.array([0, 0])
    psiopt = 'nm'

    # Get DGP quantities
    print('Getting DGP quantities...')
    dgpqnts = dgp(T+1, psin, psiopt, betaL, betaP)

    # Simulate data and estimate
    psi_hats = sim_multi(n, dgpqnts, seeds=range(iters))
    psi_hat = np.mean(psi_hats, axis=0)

    # Plot
    custom_plot([psi_hat, dgpqnts['psi']],
                legendlabs=['Estimated', 'True'])

##########################################################