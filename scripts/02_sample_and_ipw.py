##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Import custom functions and parameters
from utils.customplot import set_plot_aes
from utils.datadesc import pred_ps, rmlinestex
from utils.config import DATA_DIR, QUANTS_DIR, Colors, OUTPUT_DIR

##########################################################
# Sample selection
##########################################################

# Load data
dws = pd.read_csv(f'{DATA_DIR}/dws.csv')

# Variables with missing values
missingval_vars = ['dwlastwrk', 'dwnotice', 'dwunion', 'dwhi', 'dwyears', 
                   'dwweekl', 'dwjobsince', 'ind', 'occ', 'obsdur']
# Note obsdur is missing when jf=1 & dwwksun is missing

# Sample selection 
sample = dws.copy()
sample = sample[(sample['year'] >= 1996) & (sample['year'] <= 2020)]
sample = sample[(sample['age'] >= 21) & (sample['age'] <= 64)]

# Conditions for sample selection
cond1 = (sample['dwrecall'] != 2) 
cond2 = (sample['dwclass'] <= 3) 
cond3 = (sample[missingval_vars].isnull().sum(axis=1) == 0)
cond4 = (sample['dwfulltime'] == 2)
cond5 = (sample['dwyears'] >= 0.5)
cond6 = (sample['dwhi'] == 2)
cond7 = (sample['dwjobsince'] <= 2)
cond8 = (sample['dwnotice'] >= 3) & (sample['dwnotice'] <= 4)
conds = np.array([eval(f'cond{j+1}') for j in range(8)]).T

# Labels for conditions
condlabs = ['DWS 1996-2020, 21-64 year old respondents',
            'No recall expectation',
            'Lost job was not self-employment', 
            'Non-missing values for variables used',
            'Worked full-time at lost job',
            'Employed for at least 6 months at lost job',
            'Had health insurance at lost job',
            'Held less than 3 jobs since lost job',
            'Got a notice of 1-2 or >2 months',
            ]

# Apply conditions and record sample size reductions in a table
sample_sizes = []
for j in range(conds.shape[1]):
    sample_sizes.append(sample.shape[0])
    sample = sample.loc[conds[:,j]]
    conds = conds[conds[:,j]]
sample_sizes.append(sample.shape[0])

# Create latex table
tab = np.array([condlabs, sample_sizes]).T
print(tabulate(tab))
tabpath = f'{OUTPUT_DIR}/tab_sample_selection.tex'
with open(tabpath, 'w') as f:
    f.write(tabulate(tab, tablefmt='latex'))
rmlinestex(tabpath)

# Are there individuals who haven't found a job and left lf?
sample[(sample['jf'] == 0)]['nilf'].value_counts()

##########################################################
# Estimate prop score and add weights to the sample
##########################################################

# Specify controls
varlist = ['age', 'female', 'married', 'black',  'educ_cat', 'dwreas', 
            'union', 'in_metro', 'dwyears', 'lnearnl', 'statefip', 
            'dyear', 'ind_cat', 'occ_cat']

# Keep only relevant variables
controls = sample[varlist].copy()

# Add quadratic terms for continuous variables
controls['age2'] = controls['age']**2
controls['dwyears2'] = controls['dwyears']**2

# Categories as numbers
controls['dwreas'] = controls['dwreas'].astype(int)
controls['statefip'] = controls['statefip'].astype(int)
controls['dyear'] = controls['dyear'].astype(int)
controls.loc[:, 'educ_cat'] = pd.Categorical(controls['educ_cat']).codes
controls.loc[:, 'ind_cat'] = pd.Categorical(controls['ind_cat']).codes
controls.loc[:, 'occ_cat'] = pd.Categorical(controls['occ_cat']).codes

# Hot encode multiple category variables
catvars = ['dwreas', 'occ_cat', 'statefip', 'dyear', 'ind_cat', 'educ_cat']
controls = pd.get_dummies(controls, columns=catvars, drop_first=True)

# Convert all boolean variables to integers
for cols in controls.columns:
    if controls[cols].dtype == bool:
        controls[cols] = controls[cols].astype(int)

# Save control variables 
controls.to_csv(f'{DATA_DIR}/control_vars.csv', index=False)

# Estimate propensity scores and save results
est_data = pd.concat([sample[['notice', 'dur', 'cens']], controls], axis=1)
ps, coefs = pred_ps(est_data)
np.save(f'{QUANTS_DIR}/ps.npy', ps)
np.save(f'{QUANTS_DIR}/coefs.npy', coefs)

# Add balancing weights to the data
notvals = np.sort(sample['notice'].unique())
for j in range(len(notvals)):
    sample.loc[(sample['notice']==notvals[j]), 'wt'] = \
        1/ps[(sample['notice']==notvals[j]),j]

# Save sample with weights
sample.to_csv(f'{DATA_DIR}/sample.csv', index=False)

##########################################################
# Check overlap in propensity scores 
##########################################################

set_plot_aes()
colors = [Colors().BLACK, Colors().RED]
plt.figure(figsize=(5.5, 3.25))
for j in range(len(notvals)):
    sns.kdeplot(ps[(sample['notice']==notvals[j]),1], 
                 color=colors[j], fill=True, alpha=0.075, 
                 edgecolor='black')
plt.annotate('Short Notice', xy=(0.4, 2), 
             xytext=(0.175, 2),
             arrowprops=dict(facecolor='black', arrowstyle='<-'))
plt.annotate('Long Notice', xy=(0.635, 2.5), 
             xytext=(0.715, 2.5),
             arrowprops=dict(facecolor='black', arrowstyle='<-'))
sns.despine()
plt.xlim(0.075, 0.99)
plt.ylabel('Density')
plt.xlabel('Propensity score')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/fig_ps_balance.pdf', dpi=300)

##########################################################

