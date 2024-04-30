import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import t
from sklearn.linear_model import LogisticRegression

##########################################################
# Function to predict propensity scores
##########################################################

def pred_ps(data, coefs=None):
    
    # Initialize
    notice = data['notice']
    notX_vars = ['dur', 'cens', 'notice', 'cens_ind', 'wt', 'wts']
    X = data[[col for col in data.columns if col not in notX_vars]]

    # Fit model if no coefficients are provided
    if coefs is None:
        model = LogisticRegression(max_iter=10000, solver='liblinear', 
                                   fit_intercept=False)
        model.fit(X, notice)
        coefs = model.coef_.flatten()
    
    # Predict propensity scores
    X = np.array(X, dtype=float)
    betaL = np.column_stack([np.zeros(X.shape[1]), coefs])
    ps = np.exp(X @ betaL) / np.exp(X @ betaL).sum(axis=1, keepdims=True)

    return ps, coefs

##########################################################
# Function to give asterisks for significance
##########################################################

def sig_asterisk(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ' '

##########################################################
# Mean and Standard Error for weighted & unweighted data
##########################################################

def variable_stats(var, wts):
    wts = 1 if wts is None else wts
    n = len(var)
    mean = np.average(var, weights=wts)
    variance = np.sum(wts * (var - mean)**2) / np.sum(wts)
    se = np.sqrt(variance) / np.sqrt(n)
    return mean, se, variance, n

##########################################################
# Summary statistics for a variable by category
##########################################################

def stats_by_cat(df, var, label, byvar, wts=None, se=False, stars=True):

    # Initialize
    levels = np.sort(df[byvar].unique())
    K = len(levels)
    df[wts] = 1 if wts is None else df[wts]
    row1 = np.array((2*K-1)*[''], dtype='<U16')
    row2 = np.array((2*K-1)*[''], dtype='<U16')
    
    # Subset data by category
    for k in range(K-1):

        # Subset data
        level = levels[k]
        next_level = levels[k+1]
        group1 = df[df[byvar] == level]
        group2 = df[df[byvar] == next_level]
        n1, n2 = len(group1), len(group2)
        
        # Calculate means & standard errors
        mean1, se1, variance1, n1 = variable_stats(group1[var], group1[wts])
        mean2, se2, variance2, n2 = variable_stats(group2[var], group2[wts])

        # Compare means: t-test
        S = np.sqrt((variance1/n1) + (variance2/n2))
        t_stat = (mean1 - mean2) / S
        p_val = 2 * (1 - t.cdf(np.abs(t_stat), df=(n1 + n2 - 2)))
        sig_ast = sig_asterisk(p_val)

        # Fill rows
        row1[k] = f"{mean1:.2f}"
        row1[k+1] = f"{mean2:.2f}"
        row1[K+k] = f"{mean2-mean1:.2f}{sig_ast}"
        row2[k] = f"({se1:.2f})"
        row2[k+1] = f"({se2:.2f})"
        row2[K+k] = f"({S:.2f})"
    
    # Add label
    row1 = np.insert(row1, 0, label)
    row2 = np.insert(row2, 0, '')

    return row1, row2

##########################################################
# Summary statistics table for multiple variables by category
##########################################################

def sum_tab(df, varlist, byvar, labels=None, wts=None, 
           se=False, stars=True):
    table = []
    for i in range(len(varlist)):
        row1, row2 = stats_by_cat(df, varlist[i], labels[i], byvar, wts, se, stars)
        table.append(row1)
        table.append(row2)
    nL = df[byvar].value_counts().sort_index().values
    last_row = np.array((2*len(nL))*[''], dtype='<U16')
    last_row[0] = 'Observations'
    last_row[1:-1] = nL
    table.append(last_row)
    return table

##########################################################
# Custom Plot
##########################################################

def custom_plot(series, se = None, xlab = '', ylab = '', title = '',
               legendlabs = None, xticklabs = None, figsize = [3, 2.85],
               ydist = 0.1, ylims = None, crit = 1.645, colors = None,
               linestyles = None, legendpos = 'best'):
    
    # Initialize
    #matplotlib.use('PDF')
    series = [np.array(s) for s in series]
    num_series = len(series)

    # Check if se is provided
    if se is None:
        se = [None for s in series]
    assert len(se) == num_series, 'Length of standard errors must match series'

    # Fonts for elements
    plt.rcParams['font.serif'] = 'Charter'
    font_base = {'family': 'sans-serif', 'size': 9}
    font_title = font_base.copy()
    font_axis_labs = font_base.copy()
    font_legend = font_base.copy()
    font_axis_ticks = {'family': 'sans-serif', 'size': 8}

    # Custom font changes
    font_title['size'] = 10 
    font_title['weight'] = 'medium'

    # Default colors & linestyles
    red = '#d40404'
    black = '#000000'
    blue = '#38039c'    
    green = '#115701'
    colors = [red, black, blue, green] if colors is None else colors
    linestyles = ['-', '--', '--', ':'] if linestyles is None else linestyles

    # Default xtick labels
    if xticklabs is None:
        xticklabs = [f'{t}' for t in range(len(series[0]))]

    # Default legend labels
    if legendlabs is None:
        legendlabs = [f'Series {j+1}' for j in range(num_series)]

    # Y-axis limits
    if ylims is None:
        lb, ub = np.zeros(num_series), np.zeros(num_series)
        for j in range(num_series):
            se_j = np.zeros_like(series[j]) if se[j] is None else np.array(se[j])
            lb[j] = np.min(series[j] - crit * se_j)
            ub[j] = np.max(series[j] + crit * se_j)
        min_, max_ = np.min(lb), np.max(ub)
        lb = min_ - 0.1*(max_ - min_)
        ub = max_ + 0.1*(max_ - min_)
    else:
        lb, ub = ylims
    yticks = np.arange(np.round(lb*10)/10, ub, ydist)

    # Plot figure
    plt.figure(figsize=figsize)
    for j in range(num_series):
        plt.plot(series[j], label=legendlabs[j], 
                     color=colors[j], linestyle=linestyles[j])
        if se[j] is not None:
            plt.errorbar(range(len(series[j])), series[j], 
                               yerr=crit*se[j], color=colors[j], 
                               capsize=4, 
                               alpha=0.75, fmt='o', markersize=0)
    plt.xlabel(xlab, fontdict=font_axis_labs)
    plt.ylabel(ylab, fontdict=font_axis_labs)
    plt.yticks(yticks)
    plt.ylim(lb, ub)
    plt.xlim(-0.25, len(series[0])-1 + 0.25)
    plt.xticks(range(len(series[0])), xticklabs)
    plt.title(title, fontdict=font_title)

    # Customize legend
    plt.legend(prop = font_legend, loc = legendpos,
               borderaxespad=0.5, borderpad=0.25, 
               labelspacing=0.25, handlelength=2.5, framealpha=0.5)

    # # Customize font for tick labels
    for label in plt.gca().get_xticklabels() + plt.gca().get_yticklabels():
        label.set_fontsize(font_axis_ticks['size'])
        label.set_fontname(font_axis_ticks['family'])

    # Remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Adjust layout
    plt.tight_layout(pad=1)
    plt.subplots_adjust(left=0.165)

##########################################################


