import numpy as np
from utils.ML import BestModel, ModelParams
from utils.GMM import GMM

##########################################################
# Function that gives survival & density from hazard
##########################################################

def DensityFromHazard(h):
    S = np.cumprod(1 - h, axis=0)
    if h.ndim == 1:
        S = np.vstack((1, S[:-1]))
    elif h.ndim == 2:
        ones = np.ones((1, S.shape[1]))
        S = np.vstack((ones, S[:-1, :]))
    elif h.ndim == 3:
        ones = np.ones((1, S.shape[1], S.shape[2]))
        S = np.vstack((ones, S[:-1, :, :]))
    g = h * S
    return g, S

##########################################################
# Fit prob model using X & predict probs for X_c
##########################################################

def PredProbs(Y, X, X_c, model = 'logit', coefs = False):
    if model == 'best':
        model = BestModel(X, Y, print_opt = 'quiet')
    else:
        model, _ = ModelParams(model)
        model.fit(X, Y)
    pred_c = model.predict_proba(X_c)
    if coefs and model == 'logit':
        return pred_c, model.coef_
    else:
        return pred_c

##########################################################
# Fit exit prob models by notice using X & predict for X_c
##########################################################

def PredExitProbs(dur, cens, notice, X, X_c, model='logit'):
    durvals = np.sort(dur.unique())
    notcats = np.sort(notice.unique())
    T, J = len(durvals), len(notcats)
    h_i = np.zeros((T, J, X_c.shape[0]))
    for j in range(J):
        notInd = (notice == notcats[j])
        for d in range(T):
            exitInd = (dur == durvals[d]) & (cens == 0) & (notInd)
            survInd = (dur >= durvals[d]) & (notInd)
            h_i[d, j, :] = PredProbs(exitInd[survInd], 
                                     X[survInd], X_c, model)[:,1]
    return h_i

##########################################################
# Helper function for ImpliedMoms
##########################################################

def DR_Moments(x_ipw, x_ra, x_i, ps, notice):
    notcats = np.sort(notice.unique())
    T, J = x_ipw.shape
    x_dr = np.zeros((T, J))
    for j in range(J):
        notInd = (notice==notcats[j])
        for d in range(T):
            x_dr[d, j] = x_ipw[d, j] + x_ra[d, j] \
                - np.mean((notInd/ps[:, j]) * x_i[d, j, :])
    return x_dr

##########################################################
# Double Machine Learning
##########################################################

def DDML(data, model_ps = 'logit', model_ra = 'logit', nrm = 0.5, fold=None):
        
    # Unpack data
    fold = np.zeros(len(data)) if fold is None else fold
    nfolds = np.unique(fold).shape[0]
    dur = data['dur']
    cens = data['cens']
    notice = data['notice']
    notX_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in notX_vars]]

    #################################################
    # If nfolds = 1, Double ML on full sample

    if nfolds == 1:
        psiM_hat, mu_hat = {}, {}
        ps = None if model_ps is None else PredProbs(notice, X, X, model_ps)
        h_i = None if model_ra is None else PredExitProbs(dur, cens, notice, X, X, model_ra)
        g = ImpliedMoms(data, ps, h_i)[0]
        for x in g.keys():
            psiM_hat[x], mu_hat[x] = GMM(g[x], nrm, unstack = True)
        return psiM_hat, mu_hat, ps, h_i
    
    #################################################
    # If nfolds>1, Cross-fitting (Double-Debiased ML)

    # Initialize arrays
    T, J, n = len(dur.unique()), len(notice.unique()), len(data)
    ps = np.zeros((n, J))
    h_i = np.zeros((T, J, n))
    psiM_hats, mu_hats = {}, {}
    keys = ['dr', 'ra', 'ipw']
    for x in keys:
        psiM_hats[x] = np.zeros((T, J, nfolds))
        mu_hats[x] = np.zeros((T, nfolds))

    # Implement cross-fitting
    for f in range(nfolds):

        # Estimate nuisance parameters on f complement & predict on f
        ps[fold==f, :] = PredProbs(notice[fold!=f], X[fold!=f], X[fold==f], model_ps)
        h_i[:, :, fold==f] = PredExitProbs(dur[fold!=f], cens[fold!=f], 
                                           notice[fold!=f], X[fold!=f], X[fold==f], model_ra)
        
        # Estimate hazard model on fold f
        g_f = ImpliedMoms(data[fold==f], ps[fold==f], h_i[:, :, fold==f])[0]
        for x in keys:
            psiM_hats[x][:, :, f], mu_hats[x][:, f] = GMM(g_f[x], nrm, unstack = True)
        psiM_hat = {x: psiM_hats[x].mean(axis=2) for x in keys}
        mu_hat = {x: mu_hats[x].mean(axis=1) for x in keys}
    
    psiM_hat

    return psiM_hat, mu_hat, ps, h_i

##########################################################
# Outputs Data Moments 
##########################################################

def ImpliedMoms(data, ps=None, h_i=None):

    # Unpack data
    durvals = np.sort(data['dur'].unique())
    notcats = np.sort(data['notice'].unique())
    T, J, n = len(durvals), len(notcats), len(data['dur'])
    notice = data['notice']
    dur = data['dur']
    cens = data['cens']

    # Initialize arrays
    h, g, S = {}, {}, {}

    # If ps is specified compute balancing weights
    if ps is not None:
        h['ipw'] = np.zeros((T, J))
        #ps = np.clip(ps, 1e-6, 1-1e-6)
        wts = np.zeros_like(notice)
        for j in range(J):
            wts[notice==notcats[j]] = 1/ps[notice==notcats[j], j]

    # Unadjusted and (if ps specified) IPW moments
    h['raw'] = np.zeros((T, J))
    for j in range(J):
        for d in range(T):
            notInd = (notice == notcats[j])
            exitInd = (dur == durvals[d]) & (cens == 0) & (notInd)
            survInd = (dur >= durvals[d]) & (notInd)
            h['raw'][d, j] = np.sum(exitInd)/np.sum(survInd)
            if ps is not None:
                h['ipw'][d, j] = np.sum(wts*exitInd)/np.sum(wts*survInd)
    g['raw'], S['raw'] = DensityFromHazard(h['raw'])
    if ps is not None:
        g['ipw'], S['ipw'] = DensityFromHazard(h['ipw'])

    # Regression adjusted moments 
    if h_i is not None:
        h['ra'] = h_i.mean(axis=2)
        g_i, S_i = DensityFromHazard(h_i)
        g['ra'] = g_i.mean(axis=2)
        S['ra'] = S_i.mean(axis=2)

    # Doubly robust moments
    if ps is not None and h_i is not None:
        h['dr'] = DR_Moments(h['ipw'], h['ra'], h_i, ps, notice)
        g['dr'] = DR_Moments(g['ipw'], g['ra'], g_i, ps, notice)
        S['dr'] = DR_Moments(S['ipw'], S['ra'], S_i, ps, notice)
        # g_dr, S_dr = DensityFromHazard(h_dr) # which is correct?

    return g, h, S

##########################################################


