import numpy as np
from utils.DataMoms import DataMoms
from utils.EstHelpers import Unstack, ModelMoms, NumGrad
from utils.DataDesc import PredPS

##########################################################
# Individual moments incomplete
##########################################################

def IndvMoms(thta, data, nrm, ffopt='np', ps = None):
    T, J = len(data['dur'].unique())-1, len(data['notice'].unique())
    h_model = ModelMoms(*Unstack(T, J, thta, nrm, ffopt)[:2])
    _, _, exit_i, surv_i = DataMoms(data, ps)
    m_i = exit_i - h_model * surv_i
    m_i = m_i.reshape(len(data), -1)
    return m_i

##########################################################
# Individual moments complete 
##########################################################
# Only for IPW with logit model & two categories

def IndvMomsIPW(thta_all, data, nrm, ffopt='np'):

    # Unpack data
    notice = data['notice']
    notcats = np.sort(notice.unique())
    T = len(data['dur'].unique())-1
    n, J = len(data), len(notcats)
    notX_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in notX_vars]]
    X = np.array(X, dtype=float)

    # Unstack thta
    nvars = X.shape[1]
    if ffopt == 'np':
        npars = 2 * (T-1) + J
    elif ffopt == 'baseline':
        npars = 2 + (T-1) + J 
    thta = thta_all[:npars]
    coefs = thta_all[npars:]

    # All moments
    ps = PredPS(data, coefs)[0]
    psmoms = np.zeros((n, (J-1) * nvars))
    for j in range(1, J):
        psmoms[:, (j-1)*nvars:j*nvars] = \
            X * np.array((notice==notcats[j]) - ps[:, j]).reshape(-1,1)
    m_i = np.column_stack([psmoms, IndvMoms(thta, data, nrm, ffopt, ps)])
    
    return m_i

##########################################################
# Other functions for inference
##########################################################

def AvgMomsInference(*args, MomsFunc, **kwargs):
    m_i = MomsFunc(*args, **kwargs)
    m = m_i.mean(axis=0)
    return m

def OptimalWeightMat(*args, MomsFunc, **kwargs):
    m_i = MomsFunc(*args, **kwargs)
    n, nmoms = m_i.shape
    omega_i = np.zeros([n, nmoms, nmoms])
    m_i = m_i.reshape(n, nmoms, 1)
    for i in range(n):
        omega_i[i, :, :] = m_i[i, :, :] @ m_i[i, :, :].T
    omega = np.mean(omega_i, axis=0)
    W = np.linalg.inv(omega)
    return W

def StdErrors(*args, MomsFunc, **kwargs):
    W = OptimalWeightMat(*args, MomsFunc=MomsFunc, **kwargs)
    M = NumGrad(AvgMomsInference, *args, MomsFunc=MomsFunc, **kwargs)
    V = np.linalg.inv(M.T @ W @ M)
    se = np.sqrt(np.diag(V)/len(args[1]))
    return se

##########################################################