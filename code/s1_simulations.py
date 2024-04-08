# Housekeeping
import sys
import numpy as np
import pandas as pd
sys.path.append('code')
from utils.SimFns import DGP, SimData
from utils.DDML import DDML
from utils.DataFns import CustomPlot
import matplotlib.pyplot as plt
import multiprocessing as mp

####################################################################
# Functions to simulate data & estimate model
####################################################################


def SimEst(n, T, J, opt, nrm, model_ps, model_ra, nfolds, seed):
    np.random.seed(seed)
    data = SimData(n, T, J, opt)
    fold = np.random.choice(nfolds, len(data))
    psiM_hat = DDML(data, model_ps, model_ra, nrm, fold)[0]
    return psiM_hat

def SimEstMP(n, T, J, opt, nrm, model_ps, model_ra, nfolds, seeds):
    num_cores = round(mp.cpu_count())
    pool = mp.Pool(num_cores)
    results = pool.starmap(SimEst, [(n, T, J, opt, nrm, model_ps, model_ra, nfolds, seed) for seed in seeds])
    pool.close()
    pool.join()
    return results

##########################################################
# Main program
##########################################################

if __name__ == '__main__':

    # Simulation parameters
    np.random.seed(1118)
    T = 4
    J = 2   
    n, iters = 200000, 200
    nrm = 0.5
    dgp_opt = None # Default
    #opt = 'ps_non_lin'
    pstrue = 'logit' if dgp_opt is None else 'nonlin'
    model_ps = 'lasso'
    model_ra = None
    nfolds = 1

    # DGP
    psiM, mu, nuP, beta_l, beta_phi, _, _ = DGP(T, J, dgp_opt)

    # Print
    print(f'Simulation parameters: n = {n}, iterations = {iters}')
    print(f'Model PS: {model_ps} (true: {pstrue}), Model RA: {model_ra}, nfolds: {nfolds}')

    # Simulate data & estimate model multiple times
    seeds = np.random.randint(1, 10*iters, iters)
    psiM_hat_list = SimEstMP(n, T, J, dgp_opt, nrm, model_ps, model_ra, nfolds, seeds)
    print('Simulation complete')
    psiM_hat = {}
    for key in psiM_hat_list[0].keys():
        psiM_hat[key] = np.mean([r[key] for r in psiM_hat_list], axis=0)
    
    # Normalize 
    psiM_hat_nrml = {x: psiM_hat[x][1:,0]/psiM_hat[x][1,0] for x in psiM_hat.keys()}
    psiM_nrml = psiM[1:,0]/psiM[1,0]

    # Compare estimates
    plt.figure(figsize=[8, 2.5])
    i = 0
    for x in psiM_hat.keys():
        i += 1
        plt.subplot(1, 3, i)
        plt.plot(psiM_hat_nrml[x], label='Estimate', color='black', linestyle='--')
        plt.plot(psiM_nrml, label='True', color='red')
        plt.title(x)
        plt.legend()
    plt.tight_layout()
    filename = f'n{n}_it{iters}_pstrue-{pstrue}_psest-{model_ps}_ra-{model_ra}_folds{nfolds}'
    plt.savefig(f'notice/output/simulation/{filename}.png')
    plt.show()
  
  ##########################################################