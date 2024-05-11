##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom functions and parameters
from utils.datadesc import pred_ps
from utils.config import DATA_DIR, QUANTS_DIR, Colors, OUTPUT_DIR

##########################################################
# Sample selection
##########################################################

# Load data
dws = pd.read_csv(f'{DATA_DIR}/dws.csv')

# Sample selection 
sample = dws.copy()
sample = sample[(sample['year'] >= 1996) & (sample['year'] <= 2020)]
sample = sample[(sample['age'] >= 21) & (sample['age'] <= 64)]
sample = sample[(sample['dwfulltime'] == 2)]                    
sample = sample[(sample['dwclass'] <= 3)]                       
sample = sample[(sample['dwrecall'] != 2)]                     
sample = sample[(sample['dwnotice'] >= 3) & (sample['dwnotice'] <= 4)]
sample = sample[(sample['dwyears'] >= 0.5)]
sample = sample[(sample['dwhi'] == 2)]
sample = sample[(sample['dwjobsince'] <= 2)]

##########################################################
# Create exactly the same sample as before (REMOVE LATER)
##########################################################

old_sample = False
if old_sample:
    sample = dws.copy()
    sample = sample[(sample['year'] >= 1996) & (sample['year'] <= 2020)]
    sample = sample[(sample['age'] >= 21) & (sample['age'] <= 64)]
    sample = sample[(sample['dwfulltime'] == 2)]                    
    sample = sample[(sample['dwclass'] <= 3)]                       
    sample = sample[(sample['dwrecall'] != 2)]                     
    sample = sample[(sample['dwnotice'] >= 2) & (sample['dwnotice'] <= 4)]
    sample = sample[(sample['dwyears'] >= 0.5)]
    sample = sample[(sample['dwhi'] == 2)]
    sample = sample[(sample['dwjobsince'] <= 2)]
    sample = sample[(sample['dwlastwrk'] == 2) | (sample['dwlastwrk'] == 3)]
    sample = sample[(sample['jf'] == 1) | 
                    ((sample['jf'] == 0) & (sample['obsdur'] >= 52))]

##########################################################
# Remove missing values
##########################################################

rm_missing = True
if rm_missing:
    varlist = ['dwlastwrk', 'dwunion', 'dwhi', 'dwyears', 
               'dwweekl','dwjobsince', 'ind', 'occ', 'obsdur']
    sample = sample.dropna(subset=varlist)
    print('Missing values:')
    print(sample.isnull().sum()[sample.isnull().sum() != 0])

# Note obsdur is missing when jf=1 & dwwksun is missing

# Are there individuals who haven't found a job and left lf?
sample[(sample['jf'] == 0)]['nilf'].value_counts()
print(sample['notice'].value_counts().sort_index())

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

# set font as Charter
plt.rcParams['font.serif'] = 'Charter'
colors = [Colors().BLACK, Colors().RED]
plt.figure(figsize=(5.5, 3.25))
for j in range(len(notvals)):
    print(j, notvals[j])
    print(f'Notice value: {notvals[j]}')
    sns.kdeplot(ps[(sample['notice']==notvals[j]),j], 
                 color=colors[j], fill=True, alpha=0.075, 
                 edgecolor='black')
plt.annotate('Short Notice', xy=(0.425, 2), 
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

