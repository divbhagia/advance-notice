import numpy as np

##########################################################
# Calculates the numerical gradient or Jacobian
##########################################################

def NumGrad(f, x, *args, **kwargs):

    # Initialize
    eps = 1e-6
    f_x = f(x, *args, **kwargs)
    multiOut = isinstance(f_x, tuple)
    f_x = f_x[0] if multiOut else f_x
    nEqs = len(f_x) if isinstance(f_x, np.ndarray) else 1
    nVars = len(x)

    # Calculate gradient or Jacobian
    df_dx = np.zeros([nEqs, nVars])
    for k in range(nVars):
        dx = np.zeros(nVars)
        dx[k] = eps
        f_x_plus = f(x + dx, *args, **kwargs)
        f_x_minus = f(x - dx, *args, **kwargs)
        if multiOut:
            f_x_plus, f_x_minus = f_x_plus[0], f_x_minus[0]
        if nEqs > 1:
            for j in range(nEqs):
                df_dx[j, k] = (f_x_plus[j] - f_x_minus[j]) / (2 * eps)
        else:
            df_dx[0, k] = (f_x_plus - f_x_minus) / (2 * eps)
    df_dx = df_dx if nEqs > 1 else df_dx.T
            
    return df_dx


##########################################################
# Function outputs model implied hazard  
##########################################################

def ModelMoms(psiM, mu, out='h'):

    # Initialize
    T, J = psiM.shape
    if mu.shape != (T, J):
        mu = np.repeat(mu.reshape(-1, 1), J, axis=1)

    # Density
    g = np.zeros((T, J))
    c = np.zeros((T, T, J)) 
    c[:, 0, :] = np.ones((T, J))   
    for t in range(1, T):
        for k in range(1, T):
            for j in range(J):
                c[t, k, j] = c[t - 1, k, j] - psiM[t - 1, j] * c[t - 1, k - 1, j]
    for j in range(J):
        g[:, j] = psiM[:, j] * (c[:, :, j] @ mu[:, j])

    # Hazard rate
    h, S = np.zeros((T, J)), np.zeros((T, J))
    h[0, :], S[0, :] = g[0, :], 1
    for t in range(1, T):
        S[t, :] = S[t-1, :] * (1 - h[t-1, :])
        h[t, :] = g[t, :] / S[t, :]

    # Return
    if out == 'all':
        return h, g, S
    else:
        return h
    
##########################################################
# Define psi functional form for ffopt = 'baseline'
##########################################################

def PsiBaseline(par, T):
    alph1, alph2 = par[0], par[1]
    psi = np.zeros(T-1)
    for d in range(1, T):
        psi[d-1] = (alph2/alph1) * (d/alph1)**(alph2-1) / (1 + (d/alph1)**alph2)
    return psi

##########################################################
# Function unpacks model parameters from a stacked vector
##########################################################

def Unstack(T, J, x, nrm, ffopt = 'np'):

    # Non parametric
    if ffopt == 'np':
        psin = x[:J]
        mu = np.zeros(T)
        mu[0] = nrm
        mu[1:T] = x[J:J+T-1]
        psi = x[J+T-1:]
        psiM = np.array([np.append(psin[j], psi) for j in range(J)]).T
        return psiM, mu

    if ffopt == 'baseline':
        psin = x[:J]
        mu = np.zeros(T)
        mu[0] = nrm
        mu[1:T] = x[J:J+T-1]
        par = x[J+T-1:]
        psi = PsiBaseline(par, T)
        psiM = np.array([np.append(psin[j], psi) for j in range(J)]).T
        return psiM, mu, par

# Unstack PsiM further (including SEs)
def UnstackPsiM(nL, psiM, psiSE=None):
    n, pi = nL.sum(), nL/nL.sum()
    psin = psiM[0, :]
    psi = np.sum(pi * psiM, axis=1)
    if psiSE is not None:
        piSE = np.sqrt(pi * (1-pi)/n)
        psinSE = psiSE[0, :]
        psiSE = psiSE[:, 0]
        psiSE[0] = np.sqrt(((piSE * psin)**2 + (pi * psinSE)**2).sum())
        return psi, psin, psiSE, psinSE
    else:
        return psi, psin
    
##########################################################
# Unstack standard errors
##########################################################

def UnstackAll(T, J, nL, thta, se, nrm, ffopt = 'np'):

    # Initialize
    n, pi = nL.sum(), nL/nL.sum()
    piSE = np.sqrt(pi * (1-pi)/n)
    muSE = np.zeros(T)
    psiSE = np.zeros(T)

    # Unstack parameters
    if ffopt == 'np':
        psiM, mu = Unstack(T, J, thta, nrm, ffopt)
    elif ffopt == 'baseline':
        psiM, mu, par = Unstack(T, J, thta, nrm, ffopt)
    
    # Unstack parameters further
    psin = psiM[0, :]
    psi = np.sum(pi * psiM, axis=1)

    # Standard errors for psin & mu
    psinSE = se[:J]
    muSE[1:T] = se[J:J+T-1]
    psiSE[0] = np.sqrt(((piSE * psin)**2 + (pi * psinSE)**2).sum())

    # Standard errors for psi
    if ffopt == 'np':
        psiSE[1:] = se[J+T-1:]
    elif ffopt == 'baseline':
        parSE = se[J+T-1:]
        parVar = parSE**2
        dpsi_dpar = NumGrad(PsiBaseline, par, T)
        for t in range(1, T):
            psiSE[t] = np.sqrt(dpsi_dpar[t-1]**2 @ parVar)

    # Return
    if ffopt == 'np':
        par, parSE = None, None
    
    return psin, psi, par, mu, psinSE, psiSE, parSE, muSE


##########################################################
