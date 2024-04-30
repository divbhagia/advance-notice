import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.searchmodel import parameters, avg_opt, sim_search_model
import multiprocessing as mp
from utils.datadesc import custom_plot
from utils.config import QUANTS_DIR, Colors

# np print options
np.set_printoptions(precision=4, suppress=True)

######################################################
# Function to simulate the search model using mp
######################################################

def sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, nrm, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(sim_search_model,
                        [(Tb, n, nu, p, pi, dlta0, dlta1, 
                          ffopt, nrm, seed)
                            for seed in seeds])
    pool.close()
    np.save(f'{QUANTS_DIR}/sim_search_results.npy', r)
    return r

######################################################
# Main program
######################################################

if __name__ == '__main__':

    # Parameters
    sim_again = False
    np.random.seed(1117)
    T, Tb, n = 4, 3, 3000
    othpars = parameters(T, Tb)
    nu = [1, 0.5]
    p = [0.5, 0.5]

    # Two notice types
    pi = np.array([0.5, 0.5])
    nL = n*pi
    dlta0 = np.append(1.00, 0.95*np.ones(T-1)) # short notice
    dlta1 = np.append(1.25, 0.95*np.ones(T-1)) # long notice

    # Average structural (E[h(d|nu)]) and observed hazard
    L0 = avg_opt(T, *othpars, dlta0, nu=nu, p=p)
    L1 = avg_opt(T, *othpars, dlta1, nu=nu, p=p)
    h_str = L0['h_str'] * pi[0] + L1['h_str'] * pi[1]
    piD = np.array([L0['S']/(L0['S'] + L1['S']), 
                    L1['S']/(L0['S'] + L1['S'])]).T
    h_obs = L0['h_obs'] * piD[:-1:,0] + L1['h_obs'] * piD[:-1:,1]

    # Simulate the search model
    iters = 1000
    nrm = 1
    ffopt = 'baseline'
    seeds = np.random.choice(999999999, iters)
    if sim_again:
        results = sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, 
                            ffopt, nrm, seeds)
    else:
        results = np.load(f'{QUANTS_DIR}/sim_search_results.npy',
                           allow_pickle=True)
    psi_all = np.array([results[i]['psi'] for i in range(iters)])
    psi = np.mean(psi_all, axis=0)
    psiSE = np.array([results[i]['psiSE'] for i in range(iters)])
    psiSE = np.mean(psiSE, axis=0)

    # Plot estimates
    labs =['Estimate', '$E[h(d|\\nu)]$', 'Observed']
    custom_plot([h_obs[0]*psi/psi[0], 
                 h_obs[0]*h_str/h_str[0], h_obs],
                legendlabs=labs, ylims=[0.225, 0.4], ydist=0.05,
                xlab='Time since unemployed', ylab='Hazard',)
    
    # Plot distribution of estimates
    figopts = {'bins': 30, 'color': Colors.BLACK, 'alpha': 0.1,
               'edgecolor': Colors.BLACK, 'linewidth': 0.75,
               'stat': 'density'}
    plt.figure(figsize=(6.25, 6))
    for t in range(T):
        plt.subplot(2, 2, t+1)
        plt.axvline(0, color=Colors.RED, linewidth=0.75)
        sns.histplot(psi_all[:, t]-psi[t], **figopts)
        sns.kdeplot(psi_all[:, t]-psi[t], color='black', linewidth=0.75)
        sns.despine()
        me = np.round(4 * psiSE[t], 2)
        plt.xticks(np.arange(-2, 2, 0.1))
        plt.xlim(-me, me)
        plt.ylabel('')

######################################################
