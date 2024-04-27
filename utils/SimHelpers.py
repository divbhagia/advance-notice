import numpy as np
from numpy import min, max
from numpy import min, max
import matplotlib.pyplot as plt
import pandas as pd

##########################################################
# Function that gives moments of Beta Distribution
##########################################################

# Gives num moments of Beta distribution given parameters
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

# Gives implied moments if nu is from different Beta distributions
def BetaMomsMixed(num, pars, probs):
    '''
    pars: M x 2 array with parameters for M different Beta distributions
    probs: M x 1 array with probabilities of each distribution
    '''
    moms_mat = np.zeros((num, len(pars)))
    for j in range(len(pars)):
        moms_mat[:, j] = BetaMoms(num, pars[j][0], pars[j][1])[0]
        for k in range(num):
            moms_mat[k, j] *= probs[j]**(k+1)
    moms = moms_mat.sum(axis=1)
    return moms

#############################################################
# Function to plot true and estimated coefficients
#############################################################

def CoefPlot(true_coef, est_coef, title='Coefficients'):
    lb = min(np.min(true_coef), np.min(est_coef))
    ub = max(np.max(true_coef), np.max(est_coef))
    lb, ub = 0.9*lb, 1.1*ub
    vars = len(true_coef)
    plt.figure(figsize=(4,3))
    plt.plot(true_coef, 'o', label='True', color='red')
    plt.plot(est_coef, 'x', label='Estimated', color='black')
    plt.plot([0,vars-1], [0,0], color='black', linestyle='--')
    ticks = [f'x{i}' for i in range(vars)]
    plt.xticks(range(vars), ticks)
    plt.ylim(lb, ub)
    plt.legend()
    plt.title(title)
    plt.show()

#############################################################
# Function to remove sparse variables
#############################################################

def RemoveSparse(betaL, betaPhi, data=None):
    betaL_ = betaL[(betaL[:,1] != 0) | (betaPhi != 0)]
    betaPhi = betaPhi[(betaL[:,1] != 0) | (betaPhi != 0)]
    betaL = betaL_
    if data is None:
        return betaL, betaPhi
    not_X_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in not_X_vars]]
    X = X.loc[:, ((betaL[:,1] != 0) | (betaPhi != 0))]
    data = pd.concat([data[not_X_vars], X], axis=1)
    return data, betaL, betaPhi
    
##########################################################
# Complex non-linear function to generate p(L|X)
##########################################################

def ComplexFunc(X, J):
    
    # Initialize
    n = X.shape[0]
    ps = np.zeros((n, J))
    x0, x1, x2 = X[:, 0], X[:, 1], X[:, 2]
    x4, x5, x6 = X[:, 3], X[:, 4], X[:, 5]

    # Non-linear transformations
    def f1(ps):
        ps = np.sin(ps) + np.exp(-np.abs(ps)) + np.power(ps, 3)
        return ps
    
    def f2(ps):
        ps[ps>np.percentile(ps, 75)] = ps**2
        ps[ps<np.percentile(ps, 25)] = ps**0.25
        return ps
    
    # Min max normalization to return p b/w 0 and 1
    def MinMaxNorm(ps):
        ps = np.array(ps)
        ps[np.isnan(ps)] = 0.5
        return (ps - min(ps))/(max(ps) - min(ps))

    # Category 1 probabilities
    ps1 = x0 + 0.5*x1 + 0.3*x2 - 0.9**np.power(x2, 2) + np.sin(x4) + \
        x1*np.power(x4, 2) + 0.5*np.exp(-np.abs(x5)) + 0.5*x6
    ps1 = f1(ps1)
    jump1 = (x2 < np.percentile(x2, 25))
    ps1[jump1] = ps1[jump1] + 2*ps1.mean()
    jump2 = (x2 > np.percentile(x2, 25)) & (x4 > np.percentile(x4, 50))
    ps1[jump2] = ps1[jump2]- 4*np.sqrt(ps1.var())
    ps1 = MinMaxNorm(ps1)

    # If J = 2, stack & return
    if J == 2:
        ps2 = 1 - ps1
        ps = np.column_stack((ps1, ps2))
        return ps
    
    # If J=3, specify category 2 probabilities
    elif J == 3:
        ps2 = x2*(0.5*x0 + 0.3*x1 + 2*(x4 * x5 * x6)) + 0.5*x5
        ps2 = np.sin(ps2)
        ps2 = f2(ps2)
        ps2 = MinMaxNorm(ps2)
        ps2[ps1 + ps2 > 1] = 1 - ps1[ps1 + ps2 > 1]
        ps3 = 1 - ps1 - ps2
        ps = np.column_stack((ps1, ps2, ps3))
        return ps
    
    # Raise error if ps> 1 or ps<0 or if sum(ps) != 1
    if np.any(ps>1) or np.any(ps<0) or np.any(ps.sum(axis=0) !=1):
        raise ValueError('Probabilities not within [0, 1] or sum not equal to 1')
   
##########################################################