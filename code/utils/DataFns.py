import numpy as np

##########################################################
# Function to calculate adjusted hazard rate
##########################################################

def DurDist(obsdur, cens, wts=None, adj=True):
    durvals = np.sort(obsdur.unique())
    h = np.zeros(len(durvals))
    se = np.zeros_like(h)
    for d in range(len(durvals)):
        num = np.sum((obsdur == durvals[d]) & (cens == 0))
        den = np.sum(obsdur >= durvals[d])
        if wts is not None:
            num = np.sum(wts*((obsdur == durvals[d]) & (cens == 0)))
            den = np.sum(wts*(obsdur >= durvals[d]))
        h[d] = num/den
        se[d] = np.sqrt(h[d]*(1-h[d])/den)
    return h, se

##########################################################
# Calculate Hazard Rate by Notice Length
##########################################################

def DurDistByNotice(obsdur, cens, notice, wts=None):
    T = len(obsdur.unique())
    J = len(notice.unique())
    hV = np.zeros((T, J))
    cats = np.sort(notice.unique())
    for j in range(J):
        hV[:, j], _ = DurDist(obsdur[notice==cats[j]], cens[notice==cats[j]], wts)
    return hV

##########################################################
# Function to calculate unadjusted hazard rate
##########################################################

def UnadjHazard(obsdur, wts=None):
    dur_range = obsdur.unique() 
    dur_range.sort()
    h = []
    for d in dur_range:
        num = np.sum((obsdur == d))
        den = np.sum(obsdur >= d)
        h.append(num/den)
    h = h[:-1]
    return h

##########################################################
# Function to generate balancing weights
##########################################################

def BalancingWeights(notice, X, model = 'logit', out = 'wts'):

    # Logit Model 
    if model == 'logit':
        from sklearn.linear_model import LogisticRegression
        logit = LogisticRegression(max_iter=1000)
        logit.fit(X, notice)
        coefs = logit.coef_.reshape(-1, 1)
        ps = logit.predict_proba(X)
        cat = notice.unique()
        cat.sort()
        wts = np.zeros_like(notice)
        for j in range(len(cat)):
            wts[notice==cat[j]] = 1/ps[notice==cat[j], j]

    # Return weights & coefficients       
    if out == 'all':
        return wts, coefs
    else:
        return wts
    
##########################################################