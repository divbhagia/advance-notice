# Housekeeping
import sys
import numpy as np
sys.path.append('code')
from utils.SimFns import DGP, SimData
from utils.AddDataFns import DurDistByNotice, BalancingWeights
from code.utils.GMMFns import ModelMoments, Unstack, GMM, Distance, NumGrad
import matplotlib.pyplot as plt
import multiprocessing as mp
np.random.seed(1117)

##########################################################

def Solve(n, T, J, nrm, seed):
    np.random.seed(seed)
    data = SimData(n, T, J)
    X = data[[col for col in data.columns if col.startswith('X')]]
    data['wts'], coefs = BalancingWeights(data['notice'], X, out='all')
    durdist = DurDistByNotice(data['obsdur'], data['cens'], 
                              data['notice'], data['wts'])
    g_data = np.array([durdist[durdist['notice']==j]['g'] for j in range(J)]).T
    x_hat = GMM(g_data, nrm)
    psiM, mu = Unstack(T, x_hat, nrm)
    psi = psiM[1:, 0]
    return psi, mu

def SimMP(n, T, J, nrm, seeds):
    num_cores = round(mp.cpu_count()/1.25)
    pool = mp.Pool(num_cores)
    results = pool.starmap(Solve, [(n, T, J, nrm, seed) for seed in seeds])
    pool.close()
    pool.join()
    return results

##########################################################
# Data generating process & simulate data
##########################################################

if __name__ == '__main__':

    # Parameters
    T = 8
    J = 2   
    n = 100000
    opt = 'exog' # None for default
    opt = None
    iters = 100

    # DGP
    psiM, mu, nu_P, beta_l, beta_phi = DGP(T, J, opt)
    g_model = ModelMoments(psiM, mu)
    psi = psiM[1:, 0]
    nrm = mu[0,0]

    # Test estimator
    xH =  GMM(g_model, nrm)
    d = Distance(xH, nrm, g_model)
    gr = max(NumGrad(xH, nrm, g_model))
    print(f'GMM distance for model moments: {d:.16f}')
    print(f'Max Gradient: {gr:.16f}')

    # Set seeds
    np.random.seed(1118)
    seeds = np.random.randint(1, 10*iters, iters)
    
    # Run loop
    print(f'Running simulation {iters} times for n={n}...')
    results = SimMP(n, T, J, nrm, seeds)
    psi_list = [r[0] for r in results]
    mu_list = [r[1] for r in results]
    print('Simulation complete.')

    # Averages
    psi_avg = np.mean(psi_list, axis=0)
    mu_avg = np.mean(mu_list, axis=0)

    # Plot
    plt.figure(figsize=[5, 4])
    plt.plot(psi/psi[0], 'blue')
    plt.plot(psi_avg/psi_avg[0], 'red', linestyle='dashed')
    plt.title('Estimated Psi')
    plt.legend(['True', 'Estimated'])
    plt.show()

    show_mom = False
    if show_mom:
        plt.figure(figsize=[5, 4])
        plt.plot(mu, 'blue')
        plt.plot(mu_avg, 'red', linestyle='dashed')
        plt.title('Estimated Mu')
        plt.legend(['True', 'Estimated'])
        plt.show()



##########################################################