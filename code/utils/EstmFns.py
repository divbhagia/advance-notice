import numpy as np
from scipy.optimize import minimize

#########################################################################

# Function outputs model implied moments (duration distribution) 
def model_moments(psiM, mu):
    T, J = psiM.shape
    c = np.zeros((T, T, J))
    c[:, 0, :] = np.ones((T, J))   
    for t in range(1, T):
        for k in range(1, T):
            for j in range(J):
                c[t, k, j] = c[t - 1, k, j] - psiM[t - 1, j] * c[t - 1, k - 1, j]
    g_model = psiM * np.tensordot(c, mu, axes=([1], [0]))
    return g_model 

#########################################################################