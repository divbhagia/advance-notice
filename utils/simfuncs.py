import numpy as np
import pandas as pd
from utils.simhelpers import beta_moms_mixed, complex_ps, remove_sparse
from utils.esthelpers import psi_baseline

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
#   'fewvars': only x6 & x7 affect exit and x6 affect notice
#   'ps_non_lin': non-linear propensity score 

##########################################################
# Function spits out the data generating process
##########################################################

def dgp(T=8, J=2, dgpopt = 'default', _print=True):
        
    #############################
    # Default parameters
    #############################

    # Which variables determine notice length and exit probs
    lvars = [0, 1, 2, 4, 5, 6]
    dvars = [0, 2, 3, 4, 6, 7]

    # Parameters for distribution of X
    xmeans = np.array([1, 2, 0, 5, 0, 1, 0.5, 0.2, 2, 1]) # means for x0-x9
    cov_x1to3 = np.array([[2, 1, -1], [1, 1, -0.5], [-1, -0.5, 1]])

    # Parameters for the structural hazard (default: increasing hazard)
    psin = np.array([0.25, 0.5, 0.7])
    psiPar = [0.25, 0.5] 
    #def psi_fun(y, a, b): return (b / a) * ((y / a) ** (b - 1))

    # Parameters for (Beta) distribution of nu
    nu_P = np.array([[1, 1], [2, 1], [1, 3]])
    nu_P_pr = np.array([xmeans[7], (1-xmeans[7])*xmeans[6], 
                        (1-xmeans[7])*(1-xmeans[6])])

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
    if dgpopt == 'inchaz':
        psiPar = [2.5, 1.25]

    # nu is independent of X
    if dgpopt == 'nu_ind_X' or dgpopt == 'no_obs':
        if _print:
            print('nu is independent of X.')
        nu_P_pr = np.array([0, 0, 1])

    # No observables (betaL = betaPhi = 0)
    if dgpopt == 'no_obs':
        if _print:
            print('betaL and betaPhi are set to 0.')
        betaL = np.zeros((K, J))
        betaPhi = np.zeros(K)

    # Only x6 & x7 affect exit and x6 affects notice
    if dgpopt == 'fewvars':
        if _print:
            print('Only x6 & x7 affect exit and x6 affects notice.')
        betaL = np.zeros((K, J))
        betaPhi = np.zeros(K)
        betaPhi[6], betaPhi[7] = 0.5, 0.2
        betaL[6, 1] = 0.8
        betaL, betaPhi = remove_sparse(betaL, betaPhi)

    #############################
    # Implied objects
    #############################

    # Structural hazard
    psi = psi_baseline(psiPar, T)
    #psi = psi_fun(2 + np.array(range(T-1)), psiPar[0], psiPar[1])
    psiM = np.zeros((T, J))
    psiM[0, :] = psin[:J]
    psiM[1:, :] = np.repeat(psi.reshape(-1, 1), J, axis=1)

    # Implied moments of nu
    mu = beta_moms_mixed(T, nu_P, nu_P_pr)

    # Out pL for dgpopt = fewvars
    if dgpopt == 'fewvars':
        import sympy as sp
        from sympy.stats import Binomial, E
        x6 = Binomial('x6', 1, xmeans[6])
        x7 = Binomial('x7', 1, xmeans[7])
        Xb = betaL.T @ [x6, x7]
        expXb = np.zeros_like(Xb)
        for j in range(len(Xb)):
            expXb[j] = sp.exp(Xb[j])
        pL = np.zeros_like(Xb)
        for j in range(len(Xb)):
            pL[j] = E(expXb[j]/ expXb.sum())
        pL = np.array(pL, dtype=float)
    elif dgpopt == 'no_obs':
        pL = np.array([1/J]*J)
    else:
        pL = None

    # betaL not used for for opt ps_non_lin
    if _print and dgpopt == 'ps_non_lin':
        print('For this opt betaL only identifies vars with no effect.')

    return psiM, mu, nu_P, betaL, betaPhi, xmeans, cov_x1to3, pL

##########################################################
# Function that generates data using the above DGP
##########################################################

def sim_data(n=100000, T=8, J=2, dgpopt = 'default', out='data', _print=False):

    # Get DGP parameters
    if _print:
        print(f'Simulating data for n={n} with T={T}, J={J} and DGP={dgpopt}...')
    psiM, _, nu_P, betaL, betaPhi, xmeans, cov_x1to3, _ = dgp(T, J, dgpopt, _print)

    # Generate X variables
    if dgpopt == 'fewvars':
        x6 = np.random.binomial(1, xmeans[6], n)
        x7 = np.random.binomial(1, xmeans[7], n)
        X = np.column_stack((x6, x7))
    else:
        a4 = 3 # range for x4
        means_x1to3 = xmeans[1:4]
        x_1_3 = np.random.multivariate_normal(mean=means_x1to3, cov=cov_x1to3, size=n)
        x4 = np.random.uniform(-a4, a4, n)
        x5 = np.random.chisquare(xmeans[5], n)
        x6 = np.random.binomial(1, xmeans[6], n)
        x7 = np.random.binomial(1, xmeans[7], n)
        x8 = np.random.poisson(xmeans[8], n)
        x9 = np.random.exponential(xmeans[9], n)
        X = np.column_stack((np.ones(n), x_1_3, x4, x5, x6, x7, x8, x9))
    K = X.shape[1]

    # Generate propensity scores p(L|X)
    if dgpopt == 'ps_non_lin':
        X_L = X[:, betaL.sum(axis=1)!=0] # using variables that enter
        pL_X = complex_ps(X_L, J)
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
    if dgpopt == 'nu_ind_X' or dgpopt == 'no_obs':
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
    if dgpopt == 'fewvars':
        col_labs = ['dur', 'cens', 'notice', 'cens_ind', 'x6', 'x7']
    data = pd.DataFrame(data, columns=col_labs)
    
    # Out data or all underlying objects
    if out == 'data':
        return data
    elif out == 'all':
        return data, nu, pL_X, phiX
    else:
        raise ValueError('Invalid out option.')

##########################################################
# Function to remove sparse variables
##########################################################

def moms_fewvars(T, pull_prev = True):

    # If already saved pull from file
    if pull_prev:
        avg_moms_X = np.load('tests/quants/moms_fewvars.npy')
        avg_moms_X = avg_moms_X[:T]
        return avg_moms_X

    # Initialize
    n = 5000000
    T_bar = 18
    J = 2
    df, nu, _, phiX = sim_data(n, T_bar+1, J, 'fewvars', 'all')
    df['phiX'], df['nu'] = phiX, nu

    # Create cases for [x6, x7]
    df['grpX'] = np.where((df['x6'] == 1) & (df['x7'] == 1), 1, 0)
    df['grpX'] = np.where((df['x6'] == 1) & (df['x7'] == 0), 2, df['grpX'])
    df['grpX'] = np.where((df['x6'] == 0) & (df['x7'] == 1), 3, df['grpX'])
    df['grpX'] = np.where((df['x6'] == 0) & (df['x7'] == 0), 4, df['grpX'])
    freq_grpX = df['grpX'].value_counts(normalize=True).sort_index()

    # Create nu^2, nu^3, ... nu^T in df
    for k in range(1, T_bar+1):
        df[f'nu{k}'] = df['nu'] ** k

    # Calculate average moments phi(X)^k E[nu^k|X] for k=1,2,..T for each group
    phi_X = df.groupby('grpX')['phiX'].mean()
    avg_moms_X = np.zeros((4, T_bar))
    for k in range(1, T_bar+1):
        avg_moms_X[:, k-1] = df.groupby('grpX')[f'nu{k}'].mean() * (phi_X**k)
    avg_moms_X = avg_moms_X.T @ freq_grpX

    # Save and return
    np.save('tests/quants/moms_fewvars.npy', avg_moms_X)

    return avg_moms_X[:T]

##########################################################    



