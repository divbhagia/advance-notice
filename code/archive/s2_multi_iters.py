# Housekeeping
import sys
import numpy as np
sys.path.append('code')
from utils.SimFns import DGP, SimData
from code.utils.AddDataFns import DurDist
import multiprocessing as mp
import matplotlib.pyplot as plt

# Function to simulate data and calculate duration distribution
def DurDistSim(n, T, J, seed=1118):
    np.random.seed(seed)
    data = SimData(n, T, J)
    dur_dist = DurDist(data['obsdur'], data['cens'])
    return dur_dist

# Function loops using MP to run DurDistSim multiple times
def DurDistSimMP(n, T, J, seeds):
    num_cores = round(mp.cpu_count()/1.5)
    pool = mp.Pool(num_cores)
    results = pool.starmap(DurDistSim, [(n, T, J, seed) for seed in seeds])
    pool.close()
    pool.join()
    return results

# Main program
if __name__ == '__main__':
    
    # Parameters
    T = 8
    J = 2  
    n = 50000
    iters = 200
    
    # Set seeds
    np.random.seed(1118)
    seeds = np.random.randint(1, 10*iters, iters)
    
    # Run loop
    print(f'Running simulation {iters} times for n={n}...')
    dd_list = DurDistSimMP(n, T, J, seeds)
    print('Simulation complete.')

    # Calculate analytical & simulation standard errors
    se_h_sim = np.std([dd['h'] for dd in dd_list], axis=0)
    se_h_anl = np.mean([dd['se_h'] for dd in dd_list], axis=0)
    se_S_sim = np.std([dd['S'] for dd in dd_list], axis=0)
    se_S_anl = np.mean([dd['se_S'] for dd in dd_list], axis=0)
    se_g_sim = np.std([dd['g'] for dd in dd_list], axis=0)
    se_g_anl = np.mean([dd['se_g'] for dd in dd_list], axis=0)

    # Plot standard errors
    plt.figure(figsize=[4, 8])
    plt.subplot(3, 1, 1)
    plt.plot(se_h_sim, 'blue')
    plt.plot(se_h_anl, 'red')
    plt.legend(['Simulation', 'Analytical'])
    plt.title('Standard Errors: Hazard Rate')

    plt.subplot(3, 1, 2)
    plt.plot(se_S_sim, 'blue')
    plt.plot(se_S_anl, 'red')
    plt.legend(['Simulation', 'Analytical'])
    plt.title('Standard Errors: Survival Rate')

    plt.subplot(3, 1, 3)
    plt.plot(se_g_sim, 'blue')
    plt.plot(se_g_anl, 'red')
    plt.legend(['Simulation', 'Analytical'])
    plt.title('Standard Errors: Density')

    # # Save plot
    plt.tight_layout()
    plt.savefig('output/simulation/s2_multi_iters.png')
    plt.show()


