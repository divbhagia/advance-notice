import numpy as np
from utils.searchmodel import parameters, avg_opt, sim_search_model
import multiprocessing as mp
from utils.datadesc import custom_plot
from utils.config import QUANTS_DIR

######################################################
# Function to simulate the search model using mp
######################################################

def sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    r = pool.starmap(sim_search_model,
                        [(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, seed)
                            for seed in seeds])
    pool.close()
    np.save(f'{QUANTS_DIR}/sim_search_results.npy', r)
    return r


######################################################
# Main program
######################################################

if __name__ == '__main__':

    # Parameters
    sim_again = True
    np.random.seed(1117)
    T, Tb, n = 5, 3, 3000
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
    piD = np.array([L0['S']/(L0['S'] + L1['S']), L1['S']/(L0['S'] + L1['S'])])
    h_obs = L0['h_obs'] * piD[1:,0] + L1['h_obs'] * piD[1:,1]

    # Simulate the search model
    iters = 1000
    ffopt = 'baseline'
    seeds = np.random.choice(100000, iters)
    if sim_again:
        results = sim_multi(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, seeds)
    else:
        results = np.load(f'{QUANTS_DIR}/sim_search_results.npy',
                           allow_pickle=True)
    psi_all = [results[i]['psi'] for i in range(iters)]
    psi = np.mean(psi_all, axis=0)

    # Plot results
    labs =['Estimate', '$E[h(d|\\nu)]$', 'Observed']
    custom_plot([h_obs[0]*psi/psi[0], h_obs[0]*h_str[:-1]/h_str[0], h_obs],
                legendlabs=labs, ylims=[0.225, 0.4], ydist=0.05,
                xlab='Time since unemployed', ylab='Hazard',)
    
######################################################