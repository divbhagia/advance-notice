import numpy as np
import pandas as pd
import sympy as sp
from sympy.stats import Bernoulli, Beta, E, sample

##########################################################

# Default DGP:
#   x0 is constant
#   x1 is Bernoulli with p = 0.5
#   x2 is Bernoulli with p = 0.2
#   X = (1, x1, x2) (3 dimensional)
#   Notice (L) prob: exp(betaL * X) / sum(exp(betaL * X))
#   phi(X) = betaP * X
#   nu independent of L | X, 
#       nu ~ Beta(1, 1) if x7 = 1
#       nu ~ Beta(2, 1) if x7 = 0 & x6 = 1
#       nu ~ Beta(1, 3) if x7 = 0 & x6 = 0
#   psin defines first period probs by notice length
#  Note: with beta_L = (c, 0, 0), exogenous L

##########################################################
# Helper functions 
##########################################################

# Log-Logistic hazard
def loglogistic(x, a1, a2):
    return (a2/a1) * (x/a1)**(a2-1) / (1 + (x/a1)**a2) 

# Parameter a1 for turning point and a2
def llpars(tpval, a2val):
    a1, a2, x, tp = sp.symbols('a1 a2 x tp')
    dhdx = sp.diff(loglogistic(x, a1, a2), x).simplify()
    turning_point = sp.logcombine(sp.solve(dhdx, x)[0], force=True)
    pars = sp.solve(turning_point - tp, (a1))
    parfunc = sp.lambdify((tp, a2), pars, 'numpy')
    return parfunc(tpval, a2val)[0]

# Log-Logistic hazard as a function of tp and a2
def loglogistictp(x, tp, a2):
    if a2 <= 1:
        raise ValueError('No turning point for a2<=1.')
    a1 = llpars(tp, a2)
    return loglogistic(x, a1, a2)

# Weibull Hazard
def weibull(x, opt='cons'):
    if opt == 'cons':
        b, k = 1, 1
    if opt == 'inc':
        b, k = 0.25, 1.25
    if opt == 'dec':
        b, k = 1.25, 0.75
    return b * k * (x**(k-1)) 

# nu parameters
def nupars(x1, x2=None):
    if x1 == 1:
        return 1, 1
    if x1 == 0 and x2 == 1:
        return 2, 1
    if x1 == 0 and x2 == 0:
        return 1, 3

##########################################################
# Define DGP function
##########################################################

def dgp(T, psin, psiopt='nm', betaL=None, betaP=None):

    """
    psiopt options:
        'nm': non-monotonic
        'inc': increasing
        'dec': decreasing
        'cons': constant
    """
    
    # Initialize and errors and warnings
    J = len(psin)
    if sum(betaP) > 0 or sum(betaP) < -1:
        print('sum(betaP) not in (-1, 0), may lead to phi(X)>1 or <0')
    if len(betaL.shape) != J-1:
        raise ValueError('betaL must have J-1 dimensions')

    # BetaL and BetaP coefficients
    coefs = np.array([1, 0, 0])
    betaP = coefs if betaP is None else np.append(1, betaP)
    betaL = coefs if betaL is None else np.append(1, betaL)
    betaL = np.column_stack((np.ones_like(coefs), betaL))

    # Implied moments of nu: E[phi(X)^k nu^k]
    X1 = Bernoulli('X1', 0.5)
    X2 = Bernoulli('X2', 0.2)
    X = np.array([1, X1, X2])
    phiX = betaP @ X 
    pars = [nupars(x1=1), nupars(x1=0, x2=1), nupars(x1=0, x2=0)]
    nu = [Beta(f'nu{i}', pars[i][0], pars[i][1]) for i in range(3)]
    nu = nu[0] * X1 + nu[1] * (1-X1) * X2 + nu[2] * (1-X1) * (1-X2)
    mu = np.array([E(phiX**t * nu**t) for t in range(1, T+1)])

    # Probability of notice
    pLnum = [sp.exp(X @ betaL[:,j]) for j in range(J)]
    pL_X = [pLnum[j] / sum(pLnum) for j in range(J)]
    pL = np.array([E(pL_X[j]) for j in range(J)], dtype=float)

    # Specify psi and stack psin & psi into psiM
    if psiopt == 'nm':
        tvals = np.linspace(10, 20, T-1)
        psi = loglogistictp(tvals, tp=tvals[T//2], a2=5)
    else:
        psi = weibull(np.arange(1, T), psiopt)
    psiM = np.zeros((T, J))
    psiM[0, :] = psin[:J]
    psiM[1:, :] = np.repeat(psi.reshape(-1, 1), J, axis=1)
    psi = psiM @ pL

    # Collect all results in a dictionary
    quants = {'mu': mu, 'pL': pL, 'psi': psi, 'psiM': psiM, 
              'X': X, 'betaL': betaL, 'betaP': betaP, 
              'pL_X': pL_X, 'nu': nu, 'phiX': phiX}

    return quants

##########################################################
# Function to simulate data
##########################################################

def sim_data(n, dgpqnts):

    # Unpack parameters
    Xdgp, nudgp, psiM = dgpqnts['X'], dgpqnts['nu'], dgpqnts['psiM']
    phiXdgp = sp.lambdify(sp.symbols('X1 X2'), dgpqnts['phiX'], 'numpy')
    pL_Xdgp = sp.lambdify(sp.symbols('X1 X2'), dgpqnts['pL_X'], 'numpy')
    T, J = psiM.shape

    # Generate X, phi(X), nu
    X = np.zeros((n, len(Xdgp)))
    X[:, 0] = 1
    for k in range(1, len(Xdgp)):
        X[:, k] = sample(Xdgp[k], size=n).astype(int)
    phiX = np.array(phiXdgp(X[:, 1], X[:, 2])).reshape(-1, 1)
    nu = np.array(sample(nudgp, size=n)).reshape(-1, 1)
    
    # Notice assignment mechanism & corresponding psi_l(d)
    pL_X = np.array(pL_Xdgp(X[:, 1], X[:, 2])).T
    L = np.array([0]*n)
    psi_i = np.zeros((n, T))
    for i in range(n):
        L[i] = np.random.choice(range(J), p=pL_X[i, :])
        psi_i[i, :] = psiM[:, L[i]]

    # Exit probabilities & unemployment duration
    exited = np.array([0]*n)
    exit_probs = psi_i * phiX * nu
    crit = np.random.random((n, T))
    exit = (exit_probs > crit).astype(int) 
    for t in range(T):
        exited += exit[:, t]
        exit[exited != 0, t+1:] = 0
    unemp_dur = np.argmax(exit, axis=1).astype(float)
    unemp_dur[exited==0] = np.nan # we do not observe unemp_dur > T

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
          ['x'+str(i) for i in range(len(Xdgp))]
    data = pd.DataFrame(data, columns=col_labs)

    return data

##########################################################


    
        





