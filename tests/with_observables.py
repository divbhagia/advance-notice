import numpy as np
from sympy.stats import E
import multiprocessing as mp
from utils.esthelpers import model_moms, unstack
from utils.dataclean import group_dur
from utils.datamoms import data_moms
from utils.simdgp import dgp, sim_data
from utils.datadesc import pred_ps
from utils.customplot import custom_plot
from utils.estgmm import gmm
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
    n = 100000
    iters = 100
    T, J = 60, 2
    interval = 12
    psin = np.array([0.15, 0.4, 0.6, 0.8])[:J]
    betaL = np.array([1, 1])
    betaP = np.array([0, 0]) 
    psiopt = 'nm'
    print('Getting DGP quantities...')
    dgpqnts = dgp(T, psin, psiopt, betaL, betaP)
    psiM = dgpqnts['psiM']
    mu = dgpqnts['mu']

    # Grouped true psi
    durvals = np.arange(0, T)
    grpd_durvals = group_dur(durvals, interval)
    new_durvals = np.unique(grpd_durvals)
    psi_bind = np.zeros(len(new_durvals))
    print(f'Grouping data into {len(new_durvals)} intervals...')
    for i, t in enumerate(new_durvals):
        idx_ = grpd_durvals == t
        psi_bind[i] = dgpqnts['psi'][idx_].sum()
    psi_bind = psi_bind[:-1]

    # Model moments
    newdurvalsid = new_durvals.astype(int)
    #newdurvalsid = np.insert(newdurvalsid, 0, 0)
    h_model, _, S_model = model_moms(psiM, mu, out='all')
    S_model_bin = S_model[newdurvalsid]
    h_str_bin = (S_model_bin[:-1] - S_model_bin[1:]) / S_model_bin[:-1] 

    # h_str binned
    data, phiX, nu, pL_X = sim_data(n, dgpqnts)
    L = data['notice'].values.astype(int)
    h_i = np.zeros((n, T))
    S_i = np.ones((n, T))
    for i in range(n):
        h_i[i, :] = psiM[:, L[i]] * nu[i]
        S_i[i, :] = np.cumprod(1-h_i[i, :])
    S_i_bin = S_i[:, newdurvalsid]
    S_i_bin = np.insert(S_i_bin, 0, 1, axis=1)
    h_i_bin = (S_i_bin[:,:-1] - S_i_bin[:,1:]) / S_i_bin[:,:-1]
    h_str_bin = h_i_bin.mean(axis=0)
    h_str = h_i.mean(axis=0)
    #h_str = np.array([h_i[L==l, :].mean(axis=0) for l in range(J)]).T
    #h_str_bin / h_model[newdurvalsid]

    # h_data, _, S_data,_ = data_moms(data, purpose='output')
    # (S_data[:-1]-S_data[1:]) / S_data[:-1]
    # data['dur'] = group_dur(data['dur'], interval)
    # h_data_bin, _, S_data_bin, _ = data_moms(data, purpose='output')
    # (S_data_bin[:-1]-S_data_bin[1:]) / S_data_bin[:-1]
    # np.row_stack((np.ones(J), S_data[newdurvalsid]))
 
    # Estimate
    print('Running simulations...')
    adj = 'none'
    seeds = np.random.randint(0, 100000, iters)
    psi_ests = sim_multi(n, dgpqnts, interval, seeds)
    psi_hat = np.mean(psi_ests, axis=0)

    # Plot
    nrmlz = True
    if nrmlz:
        psi_hat = psi_hat/psi_hat[0]
        psi_bind = psi_bind/psi_bind[0]
        h_str_bin = h_str_bin/h_str_bin[0]
        h_str = h_str/h_str[0]
    custom_plot([psi_hat, h_str_bin, psi_bind], 
                legendlabs=['Estimated', 'h str binnd', 'psi bin'])

######################################################