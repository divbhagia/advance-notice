import numpy as np
from scipy.optimize import minimize

##########################################################
# Function outputs model implied moments 
##########################################################

def ModelMoments(psiM, mu):
    T, J = psiM.shape
    if mu.shape != (T, J):
        mu = np.repeat(mu.reshape(-1, 1), J, axis=1)
    c = np.zeros((T, T, J)) 
    g = np.zeros((T, J))
    c[:, 0, :] = np.ones((T, J))   
    for t in range(1, T):
        for k in range(1, T):
            for j in range(J):
                c[t, k, j] = c[t - 1, k, j] - psiM[t - 1, j] * c[t - 1, k - 1, j]
    for j in range(J):
        g[:, j] = psiM[:, j] * (c[:, :, j] @ mu[:, j])
    return g

##########################################################
# Function unpacks model parameters from a stacked vector
##########################################################

def Unstack(T, J, x, nrm):
    psi = x[:T-1]
    mu = np.ones(T)
    mu[0] = nrm
    mu[1:] = x[T-1:2*T-2]
    psin = x[2*T-2:]
    psiM = np.array([np.append(psin[j], psi) for j in range(J)]).T
    return psiM, mu

##########################################################
# Function to calculate (absolute or weighted) distance 
# between data and model moments
##########################################################

def Distance(x, nrm, g_data, W = None):
    T, J = g_data.shape
    psiM, mu = Unstack(T, J,  x, nrm)
    g_model = ModelMoments(psiM, mu)
    mdiff = (g_model-g_data).reshape(-1)
    W = np.eye(g_data.size) if W is None else W
    dist = mdiff.T @ W @ mdiff
    return dist

##########################################################
# Function: 2-step GMM
##########################################################

def GMM(g, nrm=0.5, unstack = False):
    T, J = g.shape
    num_vars = 2 * T + J - 2
    x0 = 0.5*np.ones(num_vars)
    opts = {'disp': False, 'maxiter': 10000}
    result = minimize(Distance, x0, args=(nrm, g),
                      method='BFGS' , tol=1e-16, 
                      jac=NumGrad, options=opts)
    x_hat = result.x
    if unstack:
        psiM, mu = Unstack(T, J, x_hat, nrm)
        return psiM, mu
    else:
        return x_hat

##########################################################
# Function calculates the numerical gradient 
##########################################################

def NumGrad(x, nrm, g_data):
    epsilon = 1e-6
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        grad[i] = (Distance(x_plus, nrm, g_data) - \
                    Distance(x_minus, nrm, g_data)) / (2 * epsilon)
    return grad 

#########################################################################