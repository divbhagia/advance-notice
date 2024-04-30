import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.estgmm import estimate

######################################################
# Helpers and parameters
######################################################

def parameters(T, Tb):
    sgma = 1.75
    beta = 0.985
    rho = 1
    thta = 50*np.ones(T)
    w = np.ones(T)
    b = uiben(T, Tb, r=0.5)
    a = 0.1 * w[0]
    return sgma, beta, rho, thta, w, b, a

# UI benefits (Tb: benefit duration, r: replacement rate)
def uiben(T, Tb, r):
    b = np.zeros(T)
    b[:Tb] = r
    return b

######################################################
# Worker optimization
######################################################

def worker_opt(T, sgma, beta, rho, thta, w, b, a, dlta, nu):

    '''
    This function solves the worker's optimization problem.

    Parameters:
        T (int): Number of periods
        sgma (float): Coefficient of relative risk aversion.
        beta (float): Discount factor
        rho (float): Search cost parameter 1
        thta (array): Search cost parameter 2
        w (array): The wage profile
        b (array): The benefit profile
        a (float): Annuity payment
        dlta (array): Job arrival rate
        nu (float): Worker type

    Returns:
        s (array): Search effort
        h (array): Transition probability
        S (array): Survival function
        D (float): Unemployment duration
    '''

    # Payoffs and initialize arrays
    w = w + a * np.ones(T)
    b = b + a * np.ones(T)
    V_E, V_U, s = np.zeros(T), np.zeros(T), np.zeros(T)

    # Utility and cost of job-search functions
    def U(c): return (c**(1-sgma) - 1) / (1-sgma)    
    def C(s, thta): return thta * (s**(1+rho) / (1+rho))
    
    # Value function
    def valfun(s, thta, dlta, b, Vu, Ve):
        V = U(b) - C(s, thta) \
            + beta * ((1 - dlta * nu * s) * Vu + dlta * nu * s * Ve)
        return V
    
    # Optimal search effort function
    def se(thta, dlta, Vu, Ve):
        return (beta * dlta * nu * (Ve - Vu) / thta)**(1/rho)
    
    # Terminal value of unemployment function for optimization
    def trmunval(Vu):
        trm_s = se(thta[-1], dlta[-1], Vu, V_E[-1])
        trmVu = Vu - valfun(trm_s, thta[-1], dlta[-1], b[-1], Vu, V_E[-1]) 
        return np.abs(trmVu)
    
    # Value of employment (V_E)
    V_E[-1] = U(w[-1]) / (1 - beta)
    for d in range(T-2, -1, -1):
        V_E[d] = U(w[d]) + beta * V_E[d+1]
    
    # Value of unemployment (V_U) and search effort
    V_U[-1] = minimize(trmunval, V_E[-1], method='Nelder-Mead').x[0]  
    s[-1] = se(thta[-1], dlta[-1], V_U[-1], V_E[-1])
    for d in range(T-2, -1, -1):
        s[d] = se(thta[d], dlta[d], V_U[d+1], V_E[d+1])
        V_U[d] = valfun(s[d], thta[d], dlta[d], b[d], V_U[d+1], V_E[d+1])

    # Calculate exit rate, survival function, and unemployment duration
    h = dlta * s * nu
    S = np.zeros(T+1)
    S[0] = 1
    for d in range(1, T+1):
        S[d] = S[d-1] * (1 - h[d-1])
    D = np.sum(S[1:])
    
    return s, h, S, D

######################################################
# Average across workers
######################################################

def avg_opt(*args, nu, p=None):

    '''
    This function averages the results across different types of workers.

    Parameters:
    Same as worker_opt, except for:
        nu (array): Worker types
        p (array): Worker type probabilities (equal by default) 

    Returns:
        out (dict): A dictionary with the following keys:
            - h_str: Average exit rate
            - h_obs: Average observed exit rate
            - D: Average unemployment duration
            - s_str: Average search effort
            - h: Exit rate by worker type
            - S: Survival function by worker type
    '''
    
    # Initialization
    p = np.array([1/len(nu)]*len(nu)) if p is None else p
    T = args[0]
    J = len(p)
    s = np.zeros((T, J))
    h = np.zeros((T, J))
    S = np.zeros((T+1, J))
    D = np.zeros(J)
    
    # Solve for each type of worker
    for j in range(J):
        s[:, j], h[:, j], S[:, j], D[j] = worker_opt(*args, nu[j])
    
    # Average Quantities
    S = np.sum(p * S, axis=1)
    h_obs = np.zeros(T)
    for d in range(T):
        h_obs[d] = (S[d] - S[d+1]) / S[d] if S[d] > 0 else 0
    h_str = np.sum(h * p, axis=1)
    s_str = np.sum(s * p, axis=1)
    D = np.sum(S[1:T+1])

    # Pack and return
    out = {'h_str': h_str, 'h_obs': h_obs, 
           'D': D, 's_str': s_str, 'h': h, 'S': S}
    
    return out

######################################################
# Simulate data from the search model and estimate DTHM
######################################################

def sim_search_model(Tb, n, nu, p, pi, dlta0, dlta1, ffopt, seed=0):
    
    # Initialize
    np.random.seed(seed)
    T = len(dlta0)
    othpars = parameters(T, Tb)
    nu_i = np.random.choice(nu, size=n, p=p)
    notice = np.random.choice([0, 1], size=n, p=pi)
    dlta = notice[:, np.newaxis] * (dlta1 - dlta0) + dlta0

    # Exit probability and unemployment duration
    exit_prob = np.zeros((n, T))
    for i in range(n):
        _, exit_prob[i, :], *_ = worker_opt(T, *othpars, dlta[i], nu_i[i])
    crit = np.random.random((n, T))
    exitd = (exit_prob > crit).astype(int) 
    exited = np.zeros(n)    
    for t in range(T):
        exited += exitd[:, t]
        exitd[exited != 0, t+1:] = 0
    dur = np.argmax(exitd, axis=1).astype(float)
    
    # Create data for estimation
    cens = np.zeros(n)
    data = pd.DataFrame({'notice': notice, 'cens': cens, 'dur': dur})

    # Estimate using gmm 
    nrm = 1
    r = estimate(data, nrm, ffopt, adj='none')

    return r

######################################################

