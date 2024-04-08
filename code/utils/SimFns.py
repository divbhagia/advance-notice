##########################################################

# Default DGP:
#   x0 is constant
#   x1, x2, x3 are multivariate normal with means [2, 0, 5] and 
#   variances (2, 1, 1) & covariances (1, -1, -0.5)
#   x4 is uniform on -3, 3
#   x5 is chi squared with 1 degrees of freedom
#   x6 is Bernoulli with p = 0.5
#   x7 is Bernoulli with p = 0.2
#   x8 is Poisson with lambda = 2
#   x9 is exponential with lambda = 1
#   X = (1, x1, x2, x3, x4, x5, x6, x7, x8, x9) (10 dimensional)
#   Only x1, x2, x4, x5, x6 determine notice (L) (coefs: betaL, model: logit)
#   Only x2, x3, x4, x6, x7 determine phi(X) (coefs: betaPhi, model: )
#   nu independent of L | X, 
#       nu ~ Beta(1, 3) if x7 = 1
#       nu ~ Beta(4, 1) if x7 = 0 & x6 = 1
#       nu ~ Beta(2, 2) if x7 = 0 & x6 = 0

# Other options:
#   'inchaz': decreasing structural hazard
#   'nu_ind_X': nu independent of X
#   'no_obs': no observables (betaL = betaPhi = 0)
#   'fewvars': only x6 & x7 affect exit and x5 & x6 affect notice
#   'ps_non_lin': non-linear propensity score 

##########################################################

import sys
sys.path.append('code')

import numpy as np
import pandas as pd
from utils.SimFnsOth import BetaMomsMixed, ComplexFunc

##########################################################
# Function spits out the data generating process
##########################################################

def DGP(T=8, J=2, opt = 'default', print_=True):
        
    #############################
    # Default parameters
    #############################

    # Which variables determine notice length and exit probs
    lvars = [0, 1, 2, 4, 5, 6]
    dvars = [0, 2, 3, 4, 6, 7]

    # Parameters for distribution of X
    x_means = np.array([1, 2, 0, 5, 0, 1, 0.5, 0.2, 2, 1]) # means for x0-x9
    cov_x1to3 = np.array([[2, 1, -1], [1, 1, -0.5], [-1, -0.5, 1]])

    # Parameters for the structural hazard (default: increasing hazard)
    psin = np.array([0.3, 0.5, 0.7])
    psiPar = [0.25, 0.5] 
    def psi_fun(y, a, b): return (b / a) * ((y / a) ** (b - 1))

    # Parameters for (Beta) distribution of nu
    nu_P = np.array([[1, 3], [4, 1], [2, 2]])
    nu_P_pr = np.array([x_means[7], (1-x_means[7])*x_means[6], 
                        (1-x_means[7])*(1-x_means[6])])

    # Beta coefficients 
    betaL = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                       [0.1, 0.8, -0.7, -0.4, 0.3, 0.2, -0.9, 0.2, 0.4, 0.5],
                       [-0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.1, 0.4, -0.9, 0.2]])
    betaL = betaL[:J, :]
    betaPhi = np.array([-0.2, 0.1, 0.5, -0.6, -0.1, 0.5, 0.4, 0.9, -0.5, 0.8])
    
    # Replace 0s in beta coefs for variables that don't enter
    K = betaL.shape[1]
    betaL = np.array([betaL[:, i] if i in lvars else np.zeros(J) for i in range(K)])
    betaPhi = np.array([betaPhi[i] if i in dvars else 0 for i in range(K)])

    #############################
    # Overwrite defaults
    #############################

    # Decreasing structural hazard
    if opt == 'inchaz':
        psiPar = [2.5, 1.25]

    # nu is independent of X
    if opt == 'nu_ind_X' or opt == 'no_obs':
        if print_:
            print('nu is independent of X.')
        nu_P_pr = np.array([0, 0, 1])

    # No observables (betaL = betaPhi = 0)
    if opt == 'no_obs':
        if print_:
            print('betaL and betaPhi are set to 0.')
        betaL = np.zeros((K, J))
        betaPhi = np.zeros(K)

    # Only x6 & x7 affect exit and x5 & x6 affect notice
    if opt == 'fewvars':
        if print_:
            print('Only x6 & x7 affect exit and x5 & x6 affect notice.')
        betaL = np.zeros((K, J))
        betaPhi = np.zeros(K)
        betaPhi[6], betaPhi[7] = 0.5, 0.2
        betaL[5, 1], betaL[6, 1] = 0.3, 0.8

    #############################
    # Implied objects
    #############################

    # Structural hazard
    psi = psi_fun(2 + np.array(range(T-1)), psiPar[0], psiPar[1])
    psiM = np.zeros((T, J))
    psiM[0, :] = psin[:J]
    psiM[1:, :] = np.repeat(psi.reshape(-1, 1), J, axis=1)

    # Implied moments of nu
    mu = BetaMomsMixed(T, nu_P, nu_P_pr)

    # betaL not used for for opt ps_non_lin
    if print_ and opt == 'ps_non_lin':
        print('For this opt betaL only identifies vars with no effect.')

    return psiM, mu, nu_P, betaL, betaPhi, x_means, cov_x1to3

##########################################################
# Function that generates data using the above DGP
##########################################################

def SimData(n=100000, T=8, J=2, opt = 'default', out='data', print_=False):

    # Get DGP parameters
    if print_:
        print(f'Simulating data for n={n} with T={T}, J={J} and DGP={opt}...')
    psiM, _, nu_P, betaL, betaPhi, x_means, cov_x1to3 = DGP(T, J, opt, print_)
    K = betaL.shape[0]

    # Generate X variables
    a4 = 3 # range for x4
    means_x1to3 = x_means[1:4]
    x_1_3 = np.random.multivariate_normal(mean=means_x1to3, cov=cov_x1to3, size=n)
    x4 = np.random.uniform(-a4, a4, n)
    x5 = np.random.chisquare(x_means[5], n)
    x6 = np.random.binomial(1, x_means[6], n)
    x7 = np.random.binomial(1, x_means[7], n)
    x8 = np.random.poisson(x_means[8], n)
    x9 = np.random.exponential(x_means[9], n)
    X = np.column_stack((np.ones(n), x_1_3, x4, x5, x6, x7, x8, x9))

    # Generate propensity scores p(L|X)
    if opt == 'ps_non_lin':
        X_L = X[:, betaL.sum(axis=1)!=0] # using variables that enter
        pL_X = ComplexFunc(X_L, J)
    else:
        pL_X = np.exp(X @ betaL) / np.exp(X @ betaL).sum(axis=1, keepdims=True)
    
    # Notice assignment mechanism & corresponding psi_l(d)
    L = np.array([0]*n)
    psi_id = np.zeros((n, T))
    for i in range(n):
        L[i] = np.random.choice(range(J), p=pL_X[i, :])
        psi_id[i, :] = psiM[:, L[i]]

    # Specify phi(X)
    phiX = 2*np.exp(X @ betaPhi)/(1 + np.exp(X @ betaPhi))
    phiX = phiX.reshape(-1, 1)

    # Unobserved heterogeneity (nu is independent of L | X)
    nu = np.zeros(n)
    nu[x7==1] = np.random.beta(nu_P[0, 0], nu_P[0, 1], np.sum(x7==1))
    nu[(x7==0) & (x6==1)] = np.random.beta(nu_P[1, 0], nu_P[1, 1], 
                                           np.sum((x7==0) & (x6==1)))
    nu[(x7==0) & (x6==0)] = np.random.beta(nu_P[2, 0], nu_P[2, 1], 
                                           np.sum((x7==0) & (x6==0)))
    if opt == 'nu_ind_X' or opt == 'no_obs':
        nu = np.random.beta(nu_P[2, 0], nu_P[2, 1], n)
    nu = nu.reshape(-1, 1)

    # Exit probabilities & unemployment duration
    exited = np.array([0]*n)
    exit_probs = psi_id * phiX * nu
    crit = np.random.random((n, T))
    exit = (exit_probs > crit).astype(int) 
    for t in range(T):
        exited += exit[:, t]
        exit[exited != 0, t+1:] = 0
    unemp_dur = np.argmax(exit, axis=1).astype(float)
    unemp_dur[exited==0] = np.nan # we only that unemp_dur > T

    # (Independent) Censoring
    censtime = np.random.choice(range(1, T), n)
    cens = ((censtime < unemp_dur) | (exited==0)).astype(int)
    obsdur = unemp_dur.copy()
    obsdur[cens==1] = censtime[cens==1]
    T_bar = round(T/2)
    cens_ind = (censtime > T_bar).astype(int)

    # Create pandas dataframe (return X, L, censored, obs_dur)
    data = np.column_stack((obsdur, cens, L, cens_ind, X))
    col_labs = ['dur', 'cens', 'notice', 'cens_ind'] +\
          ['x'+str(i) for i in range(K)]
    data = pd.DataFrame(data, columns=col_labs)
    
    # Out data or all underlying objects
    if out == 'data':
        return data
    else:
        return data, nu, pL_X, phiX

##########################################################

    
    



