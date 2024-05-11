import numpy as np
import sympy as sp
from sympy.stats import Binomial, Beta, E
from utils.esthelpers import model_moms

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

def dgp(T, psin, psiopt='nm', betaL=None, betaP=None, interval=1):

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
    betaP = np.array([1, 0, 0]) if betaP is None else np.append(1, betaP)
    betaL = np.array([1, 1, 1]) if betaL is None else np.append(1, betaL)
    betaL = np.column_stack((np.array([1, 1, 1]), betaL))

    # Implied moments of nu: E[phi(X)^k nu^k]
    X1 = Binomial('X1', 1, 0.5)
    X2 = Binomial('X2', 1, 0.2)
    X = np.array([1, X1, X2])
    phiX = betaP @ X 
    pars = [nupars(x1=1), nupars(x1=0, x2=1), nupars(x1=0, x2=0)]
    nu = [Beta(f'nu{i}', pars[i][0], pars[i][1]) for i in range(3)]
    nu = nu[0] * X1 + nu[1] * (1-X1) * X2 + nu[2] * (1-X1) * (1-X2)
    mu = np.array([E(phiX**t * nu**t) for t in range(1, T+1)],
                  dtype=float)

    # Probability of notice
    pL_Xnum = sp.Array([sp.exp(X @ betaL[:,j]) for j in range(J)])
    pL_X = pL_Xnum / np.sum(pL_Xnum)
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

    # Structural moments: binned and unbinned
    h_str = np.array([E(psi[d] * phiX * nu) for d in range(T)], dtype=float)
    S_str = np.append(1, np.cumprod(1 - h_str))
    S_str_bin = S_str[1:][::interval]
    S_str_bin = np.append(1, S_str_bin)
    h_str_bin = ((S_str_bin[:-1] - S_str_bin[1:]) / S_str_bin[:-1])[:-1]

    # Observed moments: binned and unbinned
    h_obs = model_moms(psiM, mu, out='all')[0] @ pL
    S_obs = np.append(1, np.cumprod(1 - h_obs))
    S_obs_bin = S_obs[1:][::interval]
    S_obs_bin = np.append(1, S_obs_bin)
    h_obs_bin = ((S_obs_bin[:-1] - S_obs_bin[1:]) / S_obs_bin[:-1])[:-1]

    # Collect all results in a dictionary
    quants = {'mu': mu, 'pL': pL, 'psi': psi, 'psiM': psiM, 
              'X': X, 'betaL': betaL, 'betaP': betaP, 
              'pL_X': pL_X, 'nu': nu, 'phiX': phiX,
              'h_str': h_str, 'h_str_bin': h_str_bin,
              'h_obs': h_obs, 'h_obs_bin': h_obs_bin}

    return quants


##########################################################


    
        





