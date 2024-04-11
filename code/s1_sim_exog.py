# Housekeeping
import sys
import numpy as np
import pandas as pd
sys.path.append('code')
from utils.SimFns import DGP, SimData
from utils.DDML import DDML, IPW, ImpliedMoms
from utils.DataFns import CustomPlot
from utils.GMM import GMM
import matplotlib.pyplot as plt
import multiprocessing as mp

##########################################################
# Functions to simulate data & estimate model
##########################################################

def SimEst(n, T, J, dgpOpt, nrm, binSzs, scale, seed):
    np.random.seed(seed)
    data = SimData(n, T+1, J, dgpOpt, scale=scale)
    data_list = {}
    psiM_hats = {}
    for binSz in binSzs:
        data_list[binSz] = data.copy()
        data_list[binSz]['dur'] = BinDur(data['dur'], binSz)
        g = ImpliedMoms(data_list[binSz])[0]['raw']
        psiM_hats[binSz] = GMM(g, nrm, unstack=True)[0]
        psiM_hats[binSz] = psiM_hats[binSz].mean(axis=1)
    return psiM_hats

def SimEstMP(n, T, J, dgpOpt, nrm, binSzs, scale, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    results = pool.starmap(SimEst, [(n, T, J, dgpOpt, nrm, binSzs, scale, seed) for seed in seeds])
    pool.close()
    pool.join()
    return results

def BinDur(dur, binSz = 4):
    binned_dur = dur.copy()
    min_dur, max_dur = dur.min(), dur.max()
    bins = np.arange(min_dur, max_dur + binSz, binSz)
    for i in range(len(bins) - 1):
        binned_dur[(dur >= bins[i]) & (dur < bins[i+1])] = \
            bins[i] + binSz/2-0.5
    return binned_dur

##########################################################
# Main program
##########################################################

if __name__ == '__main__':

    # Simulation parameters
    np.random.seed(1118)
    T = 16
    J = 2   
    n, iters = 200000, 200
    dgpOpt = 'exog_inchaz'
    binSzs = [1, 2, 3, 4]
    scale = 1

    # DGP
    psiM, mu, nuP, betaL, betaPhi, _, _ = DGP(T, J, dgpOpt)
    psiM = psiM.mean(axis=1)
    nrm = 0.5*mu[0]

    # Bin psiM
    psiM_binned = {}
    for binSz in binSzs:
        bT = T // binSz
        psiM_binned[binSz] = np.zeros(bT)
        for t in range(bT):
             psiM_binned[binSz][t] = psiM[t*binSz:(t+1)*binSz].mean()
 
    # Simulate data & estimate model multiple times
    print(f'Simulation parameters: n = {n}, iterations = {iters}')
    seeds = np.random.randint(1, 10*iters, iters)
    psiM_hats = SimEstMP(n, T, J, dgpOpt, nrm, binSzs, scale, seeds)
    print(f'Finished {iters} iterations')

    # Average estimates
    psiM_hat_binned = {}
    for binSz in binSzs:
        psiM_hat_binned[binSz] = np.array([psiM_hats[i][binSz] for i in range(iters)]).mean(axis=0)
    
    # Normalize
    normalize = True
    if normalize:
        for binSz in binSzs:
            psiM_hat_binned[binSz] = psiM_hat_binned[binSz]/psiM_hat_binned[binSz][0]
            psiM_binned[binSz] = psiM_binned[binSz]/psiM_binned[binSz][0]

    # Plot estimates vs true
    labs = ['True', 'Estimate']
    for binSz in binSzs:
        CustomPlot([psiM_binned[binSz], psiM_hat_binned[binSz]], legendlabs=labs, title=f'Structural Hazard: Bin Size = {binSz}')
        plt.show()