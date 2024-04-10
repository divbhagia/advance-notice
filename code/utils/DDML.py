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
        model = BestModel(X, Y, print_opt = 'verbose')
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
    h_i = np.zeros((T-1, J, X_c.shape[0]))
    for j in range(J):
        notInd = (notice == notcats[j])
        for d in range(T-1):
            exitInd = (dur == durvals[d]) & (cens == 0) & (notInd)
            survInd = (dur >= durvals[d]) & (notInd)
            h_i[d, j, :] = PredProbs(exitInd[survInd], 
                                     X[survInd], X_c, model)[:,1]
    return h_i

##########################################################
# Inverse Probability Weighting 
##########################################################

def IPW(data, model_ps ='logit', folds = None):

    # Initialize
    folds = np.zeros(len(data)) if folds is None else folds
    nfolds = np.unique(folds).shape[0]
    notice = data['notice']
    n, J = len(data), len(notice.unique())
    notX_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in notX_vars]]

    # If nfolds = 1, IPW on full sample
    if nfolds == 1:
        ps = PredProbs(notice, X, X, model_ps)
        return ps
    
    # If nfolds>1, Cross-fitting, fit model on f complement & predict on f
    ps = np.zeros((n, J))
    for f in range(nfolds):
        ps[folds==f, :] = PredProbs(notice[folds!=f], X[folds!=f], X[folds==f], model_ps)
    
    return ps

##########################################################
# Regression Adjustment
##########################################################

def RegAdj(data, model_ra ='logit', folds=None):

    # Initialize
    folds = np.zeros(len(data)) if folds is None else folds
    nfolds = np.unique(folds).shape[0]
    dur = data['dur']
    cens = data['cens']
    notice = data['notice']
    notX_vars = ['dur', 'cens', 'notice', 'cens_ind']
    X = data[[col for col in data.columns if col not in notX_vars]]

    # If nfolds = 1, RA on full sample
    if nfolds == 1:
        h_i = PredExitProbs(dur, cens, notice, X, X, model_ra)
        return h_i
    
    # If nfolds>1, Cross-fitting, fit model on f complement & predict on f
    T, J, n = len(dur.unique()), len(notice.unique()), len(data)
    h_i = np.zeros((T-1, J, n))
    for f in range(nfolds):
        h_i[:, :, folds==f] = PredExitProbs(dur[folds!=f], cens[folds!=f], 
                                           notice[folds!=f], X[folds!=f], 
                                           X[folds==f], model_ra)
    
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
# Outputs Data Moments 
##########################################################

def ImpliedMoms(data, ps=None, h_i=None):

    # Unpack data
    durvals = np.sort(data['dur'].unique())
    notcats = np.sort(data['notice'].unique())
    T, J, n = len(durvals), len(notcats), len(data)
    notice = data['notice']
    dur = data['dur']
    cens = data['cens']

    # Initialize arrays
    h, g, S = {}, {}, {}

    # If ps is specified compute balancing weights
    if ps is not None:
        h['ipw'] = np.zeros((T-1, J))
        #ps = np.clip(ps, 1e-6, 1-1e-6)
        wts = np.zeros(n)
        for j in range(J):
            wts[notice==notcats[j]] = 1/ps[notice==notcats[j], j]

    # Unadjusted and (if ps specified) IPW moments
    h['raw'] = np.zeros((T-1, J))
    for j in range(J):
        for d in range(T-1):
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
# Double Machine Learning
##########################################################

def DDML(data, model_ps ='logit', model_ra ='logit', folds=None,  nrm=0.5):
        
    # Unpack data
    folds = np.zeros(len(data)) if folds is None else folds
    nfolds = np.unique(folds).shape[0]
    dur = data['dur']
    notice = data['notice']

    # IPW
    ps = IPW(data, model_ps, folds)

    # Regression Adjustment
    h_i = RegAdj(data, model_ra, folds)

    # If nfolds = 1, Double ML on full sample
    if nfolds == 1:
        psiM_hat, mu_hat = {}, {}
        g = ImpliedMoms(data, ps, h_i)[0]
        for x in g.keys():
            psiM_hat[x], mu_hat[x] = GMM(g[x], nrm, unstack = True)
        return psiM_hat, mu_hat, ps, h_i

    # If nfolds>1, Cross-fitting (Double-Debiased ML)
    T, J = len(dur.unique()), len(notice.unique())
    psiM_hats, mu_hats = {}, {}
    keys = ['dr', 'ra', 'ipw', 'raw']
    for x in keys:
        psiM_hats[x] = np.zeros((T-1, J, nfolds))
        mu_hats[x] = np.zeros((T-1, nfolds))
    for f in range(nfolds):
        g_f = ImpliedMoms(data[folds==f], ps[folds==f], h_i[:, :, folds==f])[0]
        for x in keys:
            psiM_hats[x][:, :, f], mu_hats[x][:, f] = GMM(g_f[x], nrm, unstack = True)
        psiM_hat = {x: psiM_hats[x].mean(axis=2) for x in keys}
        mu_hat = {x: mu_hats[x].mean(axis=1) for x in keys}

    return psiM_hat, mu_hat, ps, h_i

##########################################################


