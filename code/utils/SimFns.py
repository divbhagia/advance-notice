# Move these to readme file
# x0 is constant
# x1, x2, x3 are multivariate normal with means 0 and 
# variances (2, 1, 1) & covariances (1, -1, -0.5)
# x4 is uniform on -3, 3
# x5 is chi squared with 1 degrees of freedom
# x6 is Bernoulli with p = 0.5
# x7 is Bernoulli with p = 0.2
# x8 is Poisson with lambda = 2
# x9 is exponential with lambda = 1
# X = (1, x1, x2, x3, x4, x5, x6, x7, x8, x9) (10 dimensional)
# Only x1, x2, x4, x5, x6 determine notice length
# Only x1, x2, x3, x4, x7 determine transition probs (phi)
# nu independent of L | X
# Explain role of L_indep

import numpy as np
import pandas as pd
import multiprocessing as mp

##########################################################
# Function that gives moments of Beta Distribution
##########################################################

def BetaMoms(num, a, b):
    moms = np.zeros(num)
    cent_moms = np.zeros(num)
    norm_moms = np.zeros(num)
    moms[0] = a / (a + b)
    for k in range(1, num):
        moms[k] = moms[k - 1] * (a + k) / (a + b + k)
    for k in range(num):
        cent_moms[k] = moms[k] - moms[0] ** (k + 1)
        norm_moms[k] = moms[k] / moms[0] ** (k + 1)
    return moms, cent_moms, norm_moms

##########################################################
# Function spits out the data generating process
##########################################################

def DGP(T=8, J=2, opt = None):
        
    #############################
    # Default parameters
    #############################

    # Which variables determine notice length and phi(X)
    lvars = [0, 1, 2, 4, 5, 6]
    phivars = [0, 1, 2, 3, 4, 7]

    # Parameters for the structural hazard
    psin = np.array([0.2, 0.5, 1])
    psiPar = [0.25, 0.5] # decreasing hazard
    def psi_fun(y, a, b): return (b / a) * ((y / a) ** (b - 1))

    # Parameters for (Beta) distribution of nu
    nu_P1 = [2, 2]
    nu_P2 = [3, 2]
    nu_P3 = [1, 2]

    # Beta coefficients 
    beta_l1 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    beta_l2 = np.array([0.1, 0.8, -0.7, -0.4, 0.3, 0.2, -0.9, 0.2, 0.4, 0.5])
    beta_l3 = np.array([-0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.1, 0.4, -0.9, 0.2])
    beta_l = np.column_stack((beta_l1, beta_l2, beta_l3))
    beta_l = beta_l[:,:J]
    beta_phi = np.array([-0.2, 0.1, 0.5, -0.6, -0.1, 0.5, 0.4, 0.9, -0.5, 0.8])

    #############################
    # Overwrite defaults
    #############################

    if opt == 'IncHaz':
        psiPar = [0.25, 0.5]

    #############################
    # Implied objects
    #############################

    # Replace 0s in beta coefs for variables that don't enter
    K = beta_l.shape[0]
    beta_l = np.array([beta_l[i, :] if i in lvars else np.zeros(J) for i in range(K)])
    beta_phi = np.array([beta_phi[i] if i in phivars else 0 for i in range(K)])

    # Structural hazard
    psi = psi_fun(2 + np.array(range(T-1)), psiPar[0], psiPar[1])
    psiM = np.zeros((T, J))
    psiM[0, :] = psin[:J]
    psiM[1:, :] = np.repeat(psi.reshape(-1, 1), J, axis=1)

    # Moments & stacked params for (Beta) distribution for nu
    nu_P = np.column_stack((nu_P1, nu_P2, nu_P3)).T
    nu_P = nu_P[:J]
    mu = np.zeros((T, J))
    for j in range(J):
        mu[:, j] = BetaMoms(T, nu_P[j][0], nu_P[j][1])[1]

    return psiM, mu, nu_P, beta_l, beta_phi


##########################################################
# Function that generates data
##########################################################

def SimData(n, T, J, opt = None):

    # Get DGP parameters
    psiM, _, nu_P, beta_l, beta_phi = DGP(T, J)
    K = beta_l.shape[0]

    # Generate X variables
    means = [0, 0, 0]
    cov = [[2, 1, -1], [1, 1, -0.5], [-1, -0.5, 1]]
    x_1_3 = np.random.multivariate_normal(mean=means, cov=cov, size=n)
    x4 = np.random.uniform(-3, 3, n)
    x5 = np.random.chisquare(1, n)
    x6 = np.random.binomial(1, 0.5, n)
    x7 = np.random.binomial(1, 0.2, n)
    x8 = np.random.poisson(2, n)
    x9 = np.random.exponential(1, n)
    X = np.column_stack((np.ones(n), x_1_3, x4, x5, x6, x7, x8, x9))

    # Notice assignment mechanism & corresponding psi_l(d)
    den = np.exp(X @ beta_l).sum(axis=1, keepdims=True)
    p_L_X = np.exp(X @ beta_l) / den
    L = np.array([0]*n)
    L_indep = np.array(range(n))
    psi_l = np.zeros((n, T))
    for i in range(n):
        L[i] = np.random.choice(range(J), p=p_L_X[i, :])
        L_indep[i] = np.random.choice(range(J), p=p_L_X[i, :])
        L_indep[i] = np.random.choice(range(J), p=p_L_X[i, :])
        psi_l[i, :] = psiM[:, L[i]]

    # Specify phi(X)
    phi_X = np.exp(X @ beta_phi)/(1 + np.exp(X @ beta_phi))
    phi_X = phi_X.reshape(-1, 1)

    # Unobserved heterogeneity (nu is independent of L | X)
    nu = np.random.beta(nu_P[L_indep][:, 0], nu_P[L_indep][:, 1]).reshape(-1, 1)

    # Exit probabilities & unemployment duration
    exited = np.array([0]*n)
    exit_probs = psi_l * phi_X * nu
    crit = np.random.random((n, T))
    exit = (exit_probs > crit).astype(int) 
    for t in range(T):
        exited += exit[:, t]
        exit[exited != 0, t+1:] = 0
    unemp_dur = np.argmax(exit, axis=1).astype(float)
    unemp_dur[exited==0] = np.nan # we only that unemp_dur > T

   # (Independent) Censoring
    censtime = np.random.choice(range(T), n)
    cens = ((censtime < unemp_dur) | (exited==0)).astype(int)
    obsdur = unemp_dur.copy()
    obsdur[cens==1] = censtime[cens==1]
    T_bar = round(T/2)
    cns_indctr = (censtime > T_bar).astype(int)

    # Create pandas dataframe (return X, L, censored, obs_dur)
    data = np.column_stack((X, L, cens, obsdur, cns_indctr))
    data = pd.DataFrame(data, columns=['X'+str(i) for i in range(K)] + \
                         ['notice', 'cens', 'obsdur', 'cns_indctr'])
    
    return data

##########################################################

    
    



