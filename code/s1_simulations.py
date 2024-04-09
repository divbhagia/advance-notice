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

####################################################################
# Functions to simulate data & estimate model
####################################################################

def RmSparseVars(data, dgp_opt):
    betaL, betaPhi = DGP(T, J, dgp_opt, print_=False)[3:5]
    not_X_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in not_X_vars]]
    X_L = X.loc[:, (betaL[:,1] != 0)]
    X_Phi = X.loc[:, (betaPhi != 0)]
    data = pd.concat([data[not_X_vars], X_L, X_Phi], axis=1)
    return data

def SimEst(n, T, J, dgp_opt, nrm, model_ps, model_ra, nfolds, seed, rmvars):
    np.random.seed(seed)
    data = SimData(n, T, J, dgp_opt)
    if rmvars:
        data = RmSparseVars(data, dgp_opt)
    folds = np.random.choice(nfolds, len(data))
    psiM_hat = DDML(data, model_ps, model_ra, folds, nrm)[0]
    return psiM_hat

def SimEstMP(n, T, J, dgp_opt, nrm, model_ps, model_ra, nfolds, seeds, rmvars):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    results = pool.starmap(SimEst, [(n, T, J, dgp_opt, nrm, model_ps, 
                                     model_ra, nfolds, seed, rmvars) for seed in seeds])
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
    dgp_opt = 'fewvars'
    pstrue = 'nonlin' if dgp_opt == 'ps_non_lin' else 'logit'
    model_ps = 'rf'
    model_ra = 'rf'
    nfolds = 2
    rmvars = True

    # DGP
    psiM, mu, nuP, betaL, betaPhi, _, _ = DGP(T, J, dgp_opt)

    # Print
    print(f'Simulation parameters: n = {n}, iterations = {iters}')
    print(f'Model PS: {model_ps} (true: {pstrue}), Model RA: {model_ra}, nfolds: {nfolds}')

    # Simulate data & estimate model multiple times
    seeds = np.random.randint(1, 10*iters, iters)
    psiM_hat_list = SimEstMP(n, T, J, dgp_opt, nrm, model_ps, model_ra, nfolds, seeds, rmvars)
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
        plt.subplot(1, len(psiM_hat.keys()), i)
        plt.plot(psiM_hat_nrml[x], label='Estimate', color='black', linestyle='--')
        plt.plot(psiM_nrml, label='True', color='red')
        plt.title(x)
        plt.legend()
    plt.tight_layout()
    filename = f'n{n}_it{iters}_pstrue-{pstrue}_psest-{model_ps}_ra-{model_ra}_folds{nfolds}'
    plt.savefig(f'notice/output/simulation/{filename}.png')
    plt.show()
  
  ##########################################################