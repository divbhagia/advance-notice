
##########################################################
# Housekeeping
##########################################################

# Import external libraries
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Import custom functions and parameters
from utils.datadesc import pred_ps
from utils.config import DATA_DIR, Colors, OUTPUT_DIR

# Import custom functions
from utils.customplot import custom_plot, set_plot_aes
from utils.datamoms import data_moms
from utils.estgmm import gmm
from utils.inference import indv_moms, std_errs
from utils.esthelpers import unstack_all

##########################################################
# Initialize
##########################################################

# Load data
sample = pd.read_csv(f'{DATA_DIR}/sample_all.csv')
X = pd.read_csv(f'{DATA_DIR}/controls_all.csv')
sample['notice'] = sample['dwnotice']

# Prepare data for estimation
data_all = pd.concat([sample[['notice', 'dur', 'cens']], X], axis=1)
data_134 = data_all[data_all['notice'] != 2]
data_234 = data_all[data_all['notice'] != 1]
datasets = {'data_134': data_134, 'data_234': data_234}

# Prepare arrays to store results
Jstats, psi_hats, se_hats, h_avg = {}, {}, {}, {}

# Degrees of freedom and critical chi2 value
T = len(sample['dur'].unique()) - 1
J = 3
dof = T * J - 2 + (T-1) + J 
alpha = 0.05
critJ = chi2.ppf(1 - alpha, dof)

# Estimation options
trim_ps = False
nrm = 1
ffopt = 'baseline'
adj = 'ipw'

##########################################################
# Estimate models
##########################################################

# Loop over datasets
for dataname, data in datasets.items():

    # Propensity scores
    ps = pred_ps(data)[0]
    if trim_ps:
        trim = 0
        for j in range(1,J):
            trim += (ps[:,j]>0.1) & (ps[:,j]<0.9)
        trim = (trim==max(trim))
        ps = ps[trim]
        data = data.iloc[trim]

    # Some data quantities
    nL = data['notice'].value_counts().sort_index().values
    h = data_moms(data, ps)[0]
    h_avg[dataname] = h @ nL / nL.sum()

    # Estimate model
    thta_hat, Jstats[dataname] = gmm(data, nrm, ffopt, ps)
    se = std_errs(thta_hat, data, nrm, ffopt, MomsFunc=indv_moms)
    psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
            unstack_all(T, J, nL, thta_hat, se, nrm, ffopt)
    psi_hats[dataname] = psi
    se_hats[dataname] = psiSE

# p-values for J-statistics
pvals = {key: 1 - chi2.cdf(Jstats[key], dof) for key in Jstats.keys()}

##########################################################
# Plots and output
##########################################################

# Plot aesthetics
labs = ['Structural', 'Observed']
text = {key: f'J-Stat: {Jstats[key]:.2f} (p={pvals[key]:.2f})'
        for key in Jstats.keys()}
filenames = [f'{OUTPUT_DIR}/fig_notcats{x}.pdf' for x in ['A', 'B']]
colors = [Colors.BLUE, Colors.BLACK]
xticklabs = ['0-12', '12-24', '24-36', '36-48']
ylab = 'Hazard rate'
xlab = 'Weeks since unemployed'

# Plots
set_plot_aes()
for i, key in enumerate(psi_hats.keys()):
    series = [psi_hats[key], h_avg[key]]
    se = [se_hats[key], None]
    custom_plot(series, se, xlab=xlab, ylab=ylab, legendlabs=labs, 
                 xticklabs=xticklabs, colors=colors, 
                 ylims=[-0.175, 0.85], legendpos='lower left', 
                 ydist=0.2)
    plt.axvline(x=2, color='black', linestyle=':', linewidth=1.25, alpha=0.75)
    plt.text(0, 0.75, text[key], color=Colors.BLACK)
    plt.savefig(filenames[i], dpi=300, format='pdf')

##########################################################