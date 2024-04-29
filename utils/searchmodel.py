import numpy as np
from scipy.optimize import minimize

######################################################
# Worker optimization
######################################################

def worker_opt(T, sgma, beta, rho, thta, w, b, a, dlta, nu):

    '''
    This function solves the worker's optimization problem.

    Parameters:
    D (int): Number of periods
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
    h (array): Exit rate
    S (array): Survival function
    uD (float): Unemployment duration

    '''

    # Initialize arrays
    w = w + a * np.ones(T)
    b = b + a * np.ones(T)
    
    # Utility function
    def U(c): return (c**(1-sgma) - 1) / (1-sgma)
    
    # Cost of job-search function
    def C(s, thta): return thta * (s**(1+rho) / (1+rho))
    
    # Value function
    def valfun(s, thta, dlta, b, Vu, Ve):
        V = U(b) - C(s, thta) \
            + beta * ((1 - dlta * nu * s) * Vu + dlta * nu * s * Ve)
        return V
    
    # Pre-assign arrays 
    V_E, V_U, s = np.zeros(T), np.zeros(T), np.zeros(T)
    
    # Terminal value of employment
    V_E[-1] = U(w[-1]) / (1 - beta)

    # Backward induction for value of employment
    for d in range(T-2, -1, -1):
        V_E[d] = U(w[d]) + beta * V_E[d+1]
    
    # Optimal search effort function
    def se(thta, dlta, Vu, Ve):
        return (beta * dlta * nu * (Ve - Vu) / thta)**(1/rho)
    
    # Terminal value of unemployment and search effort
    def trmunval(Vu):
        trm_s = se(thta[-1], dlta[-1], Vu, V_E[-1])
        trmVu = Vu - valfun(trm_s, thta[-1], dlta[-1], b[-1], Vu, V_E[-1]) 
        return np.abs(trmVu)
    V_U[-1] = minimize(trmunval, V_E[-1], 
                       method='Nelder-Mead', 
                       options={'maxiter': 100000}).x[0]  
    s[-1] = se(thta[-1], dlta[-1], V_U[-1], V_E[-1])

    # Backward induction for value of unemployment and search effort
    for d in range(T-2, -1, -1):
        s[d] = se(thta[d], dlta[d], V_U[d+1], V_E[d+1])
        V_U[d] = valfun(s[d], thta[d], dlta[d], b[d], V_U[d+1], V_E[d+1])

    # Calculate exit rate, survival function, and unemployment duration
    h = dlta * s * nu
    S = np.zeros(T+1)
    S[0] = 1
    for d in range(1, T+1):
        S[d] = S[d-1] * (1 - h[d-1])
    
    ud = np.sum(S[1:])
    
    return s, h, S, ud

######################################################
# Average across workers
######################################################

def avg_opt(*args, nu, p):
    
    # Initialization
    T = args[0]
    J = len(p)
    s = np.zeros((T, J))
    h = np.zeros((T, J))
    S = np.zeros((T+1, J))
    ud = np.zeros(J)
    
    # Solve for each type of worker
    for j in range(J):
        s[:, j], h[:, j], S[:, j], ud[j] = worker_opt(*args, nu[j])
    
    # Average Quantities
    S = np.sum(p * S, axis=1)
    h_obs = np.zeros(T)
    for d in range(T):
        h_obs[d] = (S[d] - S[d+1]) / S[d] if S[d] > 0 else 0
    h_str = np.sum(h * p, axis=1)
    s_str = np.sum(s * p, axis=1)
    ud = np.sum(S[1:T+1])

    # Pack and return
    out = {'h_str': h_str, 'h_obs': h_obs, 
           'ud': ud, 's_str': s_str, 'h': h, 'S': S}
    
    return out

######################################################
