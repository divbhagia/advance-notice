import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

##########################################################
# Custom Plot
##########################################################

def CustomPlot(series, title = '', xlab = '', ylab = '',
                 legendlabs=None, xticklabs = None, ydist = 0.1, 
                     colors = None, linestyles = None, 
                     figsize = (4, 3), savepath = None, subplot_=False):
    
    # Initialize
    num_series = len(series)

    # Fonts for elements
    font_base = {'family': 'sans-serif', 'size': 10}
    font_title = font_base.copy()
    font_axis_labs = font_base.copy()
    font_legend = font_base.copy()
    font_axis_ticks = font_base.copy()

    # Custom font changes
    font_title['size'] = 12 
    font_title['weight'] = 'medium'

    # Default colors & linestyles
    if colors is None:
        red = '#FF6347'
        black = '#000000'
        blue = '#1E90FF'    
        purple = '#800080'
        colors = [red, black, blue, purple]
    if linestyles is None:
        linestyles = ['-', '-.', '--', ':']

    # Default xtick labels
    if xticklabs is None:
        xticklabs = [f'{t}' for t in range(len(series[0]))]

    # Default legend labels
    if legendlabs is None:
        legendlabs = [f'Series {j+1}' for j in range(num_series)]

    # Y-axis limits
    min_, max_ = np.min(series), np.max(series)
    lb = min_ - 0.2*(max_ - min_)
    ub = max_ + 0.2*(max_ - min_)
    yticks = np.arange(np.round(lb*10)/10, ub, ydist)

    # Plot figure
    if subplot_ == False:
        plt.figure(figsize=figsize)
    for j in range(num_series):
        plt.plot(series[j], label=legendlabs[j], color=colors[j], linestyle=linestyles[j])
    plt.xlabel(xlab, fontdict=font_axis_labs)
    plt.ylabel(ylab, fontdict=font_axis_labs)
    plt.legend(prop = font_legend)
    plt.yticks(yticks)
    plt.ylim(lb, ub)
    plt.xticks(range(len(series[0])), xticklabs)
    plt.title(title, fontdict=font_title)

    # # Customize font for tick labels
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(font_axis_ticks['size'])
        label.set_fontname(font_axis_ticks['family'])

    if savepath is not None:
        plt.savefig(savepath)
    if subplot_ == False:
        plt.show()


##########################################################
# Function to calculate se of product of 2 RVs
##########################################################

# Note: Var(XY) = X^2Var(Y) + Y^2Var(X) + Var(X)Var(Y)

def seProd(X, Y, se_X, se_Y):
    var_X, var_Y = se_X**2, se_Y**2
    var_XY = (X**2)*var_Y + (Y**2)*var_X + var_X*var_Y
    return np.sqrt(var_XY)

##########################################################
# Function to calculate adjusted hazard rate
##########################################################

def DurDist(dur, cens, wts=None):

    # Initialize
    durvals = np.sort(dur.unique())
    T = len(durvals)
    
    # Hazard rate
    h, se_h = np.zeros(T), np.zeros(T)
    for d in range(T):
        if cens is not None:
            ind_num = ((dur == durvals[d]) & (cens == 0)).astype(int)
            ind_den = (dur >= durvals[d]).astype(int)
        else:
            ind_num = (dur == durvals[d]).astype(int)
            ind_den = (dur >= durvals[d]).astype(int)
        if wts is not None:
            ind_num = wts*ind_num
            ind_den = wts*ind_den
        num = np.sum(ind_num)
        den = np.sum(ind_den)
        h[d] = num/den
        se_h[d] = np.sqrt(h[d]*(1-h[d])/den)

    # Survival rate 
    S, se_S = np.zeros(T), np.zeros(T)
    S[0] = 1
    for d in range(T-1):
        S[d+1] = S[d] * (1 - h[d])
        se_S[d+1] = seProd(1-h[d], S[d], se_h[d], se_S[d])

    # Density
    g = h * S
    se_g = np.array(seProd(h, S, se_h, se_S))

    # Return
    durdist = pd.DataFrame({'h': h, 'se_h': se_h, 
                       'S': S, 'se_S': se_S, 
                       'g': g, 'se_g': se_g})
    return durdist

##########################################################
# Calculate Hazard Rate by Notice Length
##########################################################

def DurDistByNotice(dur, cens, notice, wts=None):
    J = len(notice.unique())
    cats = np.sort(notice.unique())
    durdist = pd.DataFrame()
    for j in range(J):
        durdist_j = DurDist(dur[notice==cats[j]], 
                            cens[notice==cats[j]] if cens is not None else None,
                            wts[notice==cats[j]] if wts is not None else None)
        durdist_j['notice'] = cats[j]
        durdist = pd.concat([durdist, durdist_j])
    return durdist

##########################################################
# Function to generate balancing weights
##########################################################

def BalancingWeights(notice, X, model = 'logit', out = 'wts'):

    # Logit Model 
    if model == 'logit':
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
