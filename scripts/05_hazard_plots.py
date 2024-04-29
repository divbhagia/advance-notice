##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom functions
from utils.datadesc import custom_plot
from utils.datamoms import data_moms

# Import parameters
from utils.config import DATA_DIR, QUANTS_DIR, OUTPUT_DIR

##########################################################
# Initialize
##########################################################

# Load data
sample = pd.read_csv(f'{DATA_DIR}/sample.csv')
data_for_est = sample[['notice', 'dur', 'cens']]
J = len(sample['notice'].unique())

# Load propensity scores and coefficients
ps = np.load(f'{QUANTS_DIR}/ps.npy')

# Adjusted data moments
h, se_h, S, se_S = data_moms(data_for_est, ps, purpose='output')

# Legend labels
legendlabs = ['Short notice', 'Long notice']
xticklabs = ['0-12', '12-24', '24-36', '36-48', '>48']
xlab = 'Weeks since unemployed'
xticklabs = None

##########################################################
# Plot hazard and survival rates
##########################################################

# Plot adjusted hazard rates
series = [h[:,j] for j in range(J)]
se = [se_h[:,j] for j in range(J)]
custom_plot(series, se, xlab=xlab, ylab='Exit rate', 
           legendlabs=legendlabs) #, xticklabs=xticklabs[:-1])
plt.savefig(f'{OUTPUT_DIR}/fig_hazard_ipw.pdf', dpi=300, format='pdf')

# Plot adjusted survival rates
series = [S[:,j] for j in range(J)]
se = [se_S[:,j] for j in range(J)]
custom_plot(series, se, xlab=xlab, ylab='Proportion surviving',
              legendlabs=legendlabs, xticklabs=xticklabs, ydist=0.2)
plt.savefig(f'{OUTPUT_DIR}/fig_survival_ipw.pdf', dpi=300, format='pdf')

##########################################################