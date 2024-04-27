import numpy as np
from scipy.optimize import minimize
from utils.DataDesc import PredPS
from utils.DataMoms import DataMoms
from utils.Inference import OptimalWeightMat, IndvMoms
from utils.EstHelpers import Unstack, ModelMoms, NumGrad, UnstackAll
from utils.Inference import IndvMoms, IndvMomsIPW, StdErrors

##########################################################
# GMM moments & objective function at given parameters
##########################################################

def AvgMoms(thta, exit, surv, nrm, ffopt):
    T, J = exit.shape
    h_model = ModelMoms(*Unstack(T, J, thta, nrm, ffopt)[:2])
    m = np.zeros([T, J])
    for t in range(T):
        for j in range(J):
            m[t, j] = exit[t, j] - h_model[t, j] * surv[t, j]
    m = m.reshape(T*J, 1).flatten()
    return m

def ObjFunc(thta, exit, surv, nrm, ffopt, W = None):
    m = AvgMoms(thta, exit, surv, nrm, ffopt)
    W = np.eye(m.size) if W is None else W
    obj = m.T @ W @ m 
    return obj

##########################################################
# Two-step GMM
##########################################################

def GMM(data, nrm, ffopt='np', ps = None):

    # Data moments
    exit, surv, _, _ = DataMoms(data, ps)

    # Initialize
    T, J = exit.shape
    if ffopt == 'np':
        num_pars = 2 * (T-1) + J
    elif ffopt == 'baseline':
        num_pars = 2 + (T-1) + J 
    num_moms = T * J
    thta0 = 0.5*np.ones(num_pars)
    W = np.eye(num_moms) 

    # Wrapper for the numerical gradient
    def NumGradWrapper(x, *args):
        return NumGrad(ObjFunc, x, *args).flatten()
    
    # Function to minimize
    def Minimize(thta0, W):
        opts = {'disp': False, 'maxiter': 100000}
        results = minimize(ObjFunc, thta0, method='BFGS', tol=1e-32,
                           args=(exit, surv, nrm, ffopt, W), 
                           jac=NumGradWrapper,options=opts)
        return results.x
        
    # Two-step GMM
    thta_hat = Minimize(thta0, W)
    if num_moms > num_pars:
        print('Two-step GMM')
        W = OptimalWeightMat(thta_hat, data, nrm, ffopt, MomsFunc=IndvMoms)
        thta_hat = Minimize(thta_hat, W)

    return thta_hat

##########################################################
# Function to estimate and calculate standard errors
##########################################################

def Estimate(data, nrm, ffopt, adj='ipw'):
    
    nL = data['notice'].value_counts().sort_index().values
    T, J = len(data['dur'].unique())-1, len(data['notice'].unique())
    if adj == 'ipw':
        ps, coefs = PredPS(data)
        thta_hat = GMM(data, nrm, ffopt, ps)
        thta_all = np.append(thta_hat, coefs)
        se = StdErrors(thta_all, data, nrm, ffopt, MomsFunc=IndvMomsIPW)
        se = se[:len(thta_hat)]
    elif adj == 'none':
        ps, coefs = None, None
        thta_hat = GMM(data, nrm, ffopt)
        se = StdErrors(thta_hat, data, nrm, ffopt, MomsFunc=IndvMoms)
    psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
        UnstackAll(T, J, nL, thta_hat, se, nrm, ffopt)
    
    r = {'psin': psin, 'psi': psi, 'par': par, 'mu': mu,
               'psinSE': psinSE, 'psiSE': psiSE, 'parSE': parSE, 
               'muSE': muSE, 'ps': ps, 'coefs': coefs}

    return r

##########################################################