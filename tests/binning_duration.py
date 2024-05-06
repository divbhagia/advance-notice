import numpy as np
import multiprocessing as mp
from utils.simfuncsnew import dgp, sim_data
from utils.customplot import custom_plot
from utils.estgmm import estimate, gmm
import sympy as sp
np.random.seed(1118)

######################################################
# Functions to simulate and estimate 
######################################################

def sim_once(n, dgpqnts, adj, seed):
    np.random.seed(seed)
    data = sim_data(n, dgpqnts)
    nrm = dgpqnts['mu'][0]
    ests = estimate(data, nrm, ffopt='np', adj=adj)
    return ests['psi']

def sim_multi(n, dgpqnts, adj, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(sim_once, [(n, dgpqnts, adj, seed) 
                      for seed in seeds])
    pool.close()
    return r

######################################################
# Main program
######################################################

if __name__ == '__main__':

    # DGP parameters
    n = 100000
    iters = 1
    T, J = 6, 2
    psin = np.array([0.15, 0.4, 0.6, 0.8])[:J]
    X1, X2 = sp.symbols('X1 X2')  
    betaL = np.array([0.1, 0.5])
    betaP = np.array([0.3, -0.6]) 
    psiopt = 'inc'
    dgpqnts = dgp(T, psin, psiopt, betaL, betaP)
 
    # Estimate
    print('Running simulations...')
    adj = 'ipw'
    seeds = np.random.randint(0, 100000, iters)
    psi_ests = sim_multi(n, dgpqnts, adj, seeds)
    psi_hat = np.mean(psi_ests, axis=0)

    # Plot
    custom_plot([dgpqnts['psi'][:-1], psi_hat], 
                legendlabs=['True', 'Estimated'])

######################################################