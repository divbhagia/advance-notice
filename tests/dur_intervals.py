##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import multiprocessing as mp

# Import custom functions
from utils.simfuncs import dgp, sim_data
from utils.customplot import custom_plot
from utils.esthelpers import unstack, model_moms
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
    iters = 1
    T, J = 6, 2
    n = 50000
    dgpopt = 'no_obs' 
    dgpopt = 'inchaz'
    dgpopt = 'dechaz'
    psiMtr, mu, nuP, betaL, betaPhi, x_means, cov_x1to3, pL = dgp(T, J, dgpopt)
    psi_true = psiMtr @ pL
    nrm = mu[0]
    interval = 1
    print(f'Simulating data for n={n} with T={T}, J={J} and DGP={dgpopt}...')

    # Simulate data
    data, nu, pL_X, phiX = sim_data(n, T, J, dgpopt, out='all')
    nu.mean()

    # Model moments
    h_model, _, S_model = model_moms(psiMtr, mu, out='all')
    L = data['notice'].values.astype(int)
    h_i = np.zeros((n, T))
    S_i = np.zeros((n, T))
    for i in range(n):
        h_i[i, :] = psiMtr[:, L[i]] * nu[i]
        S_i[i, :] = np.cumprod(1-h_i[i, :])
    h_str = np.array([h_i[L==l, :].mean(axis=0) for l in range(J)]).T
    S_str = np.array([S_i[L==l, :].mean(axis=0) for l in range(J)]).T
    #S_str = np.cumprod(1-h_str, axis=0)
    S_str_avg = S_str @ pL
    

    # Indexes after grouping
    durvals = np.arange(0, T)
    grpd_durvals = group_dur(durvals, interval)
    new_durvals = np.unique(grpd_durvals)
    newdurvalsid = new_durvals.astype(int)
    S_str_avg = S_str_avg[newdurvalsid]
    S_str_avg = np.insert(S_str_avg, 0, 1)

    S_diff = (S_str_avg[:-1]-S_str_avg[1:])/S_str_avg[:-1]
    S_diff2 = S_str_avg[:-1]-S_str_avg[1:]
    #S_diff = S_diff[1:]

    # Grouped true psi
    psi_grpd = np.zeros(len(new_durvals))
    print(f'Grouping data into {len(new_durvals)} intervals...')
    for i, t in enumerate(new_durvals):
        idx_ = grpd_durvals == t
        psi_grpd[i] = psi_true[idx_].sum()
    psi_grpd = psi_grpd[:-1]

    # Simulate data
    print('Running simulation...')
    seeds = np.random.choice(100000, iters)
    r = SimMulti(n, T, J, dgpopt, nrm, 'np', interval, seeds)
    psi_avg = np.mean(r, axis=0)

    # Plot
    nrmlz = True
    if nrmlz:
        psi_grpd = psi_grpd/psi_grpd[0]
        psi_avg = psi_avg/psi_avg[0]
        S_diff = S_diff/S_diff[0]
        S_diff2 = S_diff2/S_diff2[0]
    custom_plot([psi_avg, psi_grpd], 
                legendlabs = ['Estimated', 'True'])
    #custom_plot([psi_true])


##########################################################


    
