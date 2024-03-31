# Housekeeping
import sys
import numpy as np
sys.path.append('code')
from utils.SimFns import DGP, SimData
from utils.DataFns import DurDist
import multiprocessing as mp
import matplotlib.pyplot as plt

# Function to simulate data and calculate duration distribution
def HazardSim(n, T, J):
    data = SimData(n, T, J)
    h, se = DurDist(data['obsdur'], data['cens'])
    return h, se

# Function loops using MP to run HazardSim multiple times
def HazardSimMP(n, T, J, iters):
    num_cores = mp.cpu_count()
    pool = mp.Pool(num_cores)
    results = pool.starmap(HazardSim, [(n, T, J) for i in range(iters)])
    h_list = np.array([results[i][0] for i in range(iters)])
    se_list = np.array([results[i][1] for i in range(iters)])
    pool.close()
    pool.join()
    return h_list, se_list

# Main program
if __name__ == '__main__':
    
    # Initialise
    np.random.seed(1118)
    print('Looping multiple times using MP...')

    # Parameters
    T = 8
    J = 2  
    n = 10000
    iters = 500 
    
    # Find hazard rates and standard errors
    h_list, se_list = HazardSimMP(n, T, J, iters)
    se_sim = np.std(h_list, axis=0)
    se_anl = np.mean(se_list, axis=0)
    avg_h = np.mean(h_list, axis=0)

    # Plot standard errors
    plt.figure(figsize=[5, 4])
    plt.subplot(2, 1, 1)
    plt.plot(se_sim, 'blue')
    plt.plot(se_anl, 'red')
    plt.legend(['Simulation', 'Analytical'])
    plt.title('Standard Errors')
    
    # Plot average hazard + margin of error
    plt.subplot(2, 1, 2)
    plt.plot(avg_h + 1.96 * se_sim, 'blue')
    plt.plot(avg_h + 1.96 * se_anl, 'red')
    plt.legend(['Simulation', 'Analytical'])
    plt.title('Average Hazard + Margin of Error')

    # Save plot
    plt.savefig('output/simulation/s2_multi_iters.png')
    plt.show()


