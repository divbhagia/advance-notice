# Add utils to path
import sys; sys.path.append('code')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patsy import dmatrix

from utils.DDML import DDML, ImpliedMoms
from utils.DataFns import CustomPlot

##########################################################
# Prepare data
##########################################################

# Load data
data_dir = 'data'
sample = pd.read_csv(f'{data_dir}/sample.csv')
sample0 = sample.copy()

# Extra selection
sample0 = sample0[(sample0['dwyears'] >= 0.5)]
sample0 = sample0[(sample0['dwhi'] == 2)]
sample0 = sample0[(sample0['dwnotice'] >= 2) & (sample0['dwnotice'] <= 4)]
sample0 = sample0[(sample0['dwlastwrk'] == 2) | (sample0['dwlastwrk'] == 3)]
sample0 = sample0[(sample0['dwjobsince'] <= 2)]
sample0 = sample0[(sample0['jf'] == 1) | (sample0['obsdur'] >= 52)]


sample0['notice'] = (sample0['dwnotice']==4)
#sample0['notice'] = sample0['dwnotice']
sample0['union'] = np.where(sample0['union'].isnull(), 0, sample0['union'])

# Prep data
sample0['cens'] = 1-sample0['jf']
controls = ['age', 'female', 'married', 'black', 'col', 'pc', 'union', 'in_metro',
              'dwyears', 'lnearnl', 'occ_cat', 'statefip', 'dyear', 'ind_cat']
              
sample0 = sample0[controls + ['notice', 'dur', 'cens']]

# Numbers as categories
for var in ['ind_cat', 'occ_cat']:
    sample0.loc[:, f'{var}'] = pd.Categorical(sample0[var]).codes

# Create interactions vars between dyear and ind_cat
int_terms = dmatrix("C(dyear):C(ind_cat)", data=sample0, return_type='dataframe')

# Hot encode occ_cat, statefip, dyear, ind_cat
sample0 = pd.get_dummies(sample0, columns=['occ_cat', 'statefip', 'dyear', 'ind_cat'])

# Logit regression
data = pd.concat([sample0, int_terms], axis=1)

# Predict prop
from utils.DDML import DDML, ImpliedMoms, IPW, RegAdj
ps = IPW(data, model_ps='logit')
g, h, S = ImpliedMoms(data, ps)

# Regression adjustment
h_i = RegAdj(data, model_ra='logit')
g, h, S = ImpliedMoms(data, h_i = h_i)

# All moments
g, h, S = ImpliedMoms(data, ps, h_i)

# Plot h
plt.figure()
plt.plot(h['ipw'][:, 0], label='Short')
plt.plot(h['ipw'][:, 1], label='Medium')
#plt.plot(h['ipw'][:-1, 2], label='Long')
plt.legend()
plt.show()

# Plot h
plt.figure()
plt.plot(h['ra'][:, 0], label='Short')
plt.plot(h['ra'][:, 1], label='Medium')
#plt.plot(h['ra'][:-1, 2], label='Long')
plt.legend()
plt.show()

# DR
plt.figure()
plt.plot(h['dr'][:, 0], label='Short')
plt.plot(h['dr'][:, 1], label='Medium')
#plt.plot(h['dr'][:-1, 2], label='Long')
plt.legend()
plt.show()

# DDML
psiM_hat, mu_hat, ps, h_i = DDML(data, model_ps ='logit', model_ra ='logit', folds=None,  nrm=0.5)

# Plot
plt.figure()
plt.plot(psiM_hat['ipw'][:, 0], label='IPW')
plt.plot(psiM_hat['dr'][:, 0], label='DR')
plt.plot(psiM_hat['ra'][:, 0], label='RA')
plt.plot(psiM_hat['raw'][:, 0], label='Raw')
plt.legend()
plt.show()