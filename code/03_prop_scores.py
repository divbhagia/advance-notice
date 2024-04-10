# Add utils to path
import sys; sys.path.append('code')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.DDML import DDML, ImpliedMoms
from utils.DataFns import CustomPlot

##########################################################
# Prepare data
##########################################################

# Load data
data_dir = 'data'
sample = pd.read_csv(f'{data_dir}/sample.csv')

# Additional sample selection
sample = sample[sample['dwyears'] >= 0.5]
sample = sample[sample['dwhi'] == 1]
sample = sample[sample['dwjobsince'] <= 2]
sample = sample[sample['dwnotice'] != 1]

# Create additional variables
sample['cens'] = 1-sample['jf']
sample['notice'] = sample['dwnotice']

# Collect pre-notice variables
cat_vars = ['year', 'statefip', 'metro', 'sex', 'marst', 'educ_cat', 'race', 
            'dwlastwrk', 'dwunion', 'dwhi', 'ind_cat', 'occ_cat', 'dwreas']
cat_vars = ['year', 'statefip', 'metro', 'sex', 'marst', 'educ_cat', 'race', 
            'dwunion', 'ind_cat', 'occ_cat', 'dwreas']
cont_vars = ['age', 'dwyears', 'dwweekl', 'ur', 'gdp']

# Keeo only relevant variables
data = sample[cat_vars + cont_vars + ['dur'] + ['cens'] + ['notice']]

# Count missing values
data[cat_vars].isnull().sum()
data[cont_vars].isnull().sum()

# Describe variables
# f = open(f'output/var_desc.txt', 'w')
# with f:
#     f.write(f'{data[cont_vars].describe()}\n\n')
#     f.write(f'{data['notice'].value_counts().sort_index()}\n\n')
#     f.write(f'{data['cens'].value_counts().sort_index()}\n\n')
#     f.write(f'{data['dur'].value_counts().sort_index()}\n\n')
#     for var in cat_vars:
#         f.write(f'{data[var].value_counts().sort_index()}\n\n')
# f.close()

##########################################################
# Double Machine Learning
##########################################################

# Numbers as categories
for var in ['educ_cat', 'ind_cat', 'occ_cat']:
    data.loc[:, f'{var}'] = pd.Categorical(data[var]).codes
    print(data[f'{var}'].value_counts().sort_index(), 
          sample[var].value_counts().sort_index())

# Declare all categorical variables as such
data[cat_vars] = data[cat_vars].astype('category')

# Hot encode categorical variables
X_cont = data[cont_vars]
X_inds = pd.get_dummies(data[cat_vars], drop_first=True)

# Polynomial features
X2 = pd.DataFrame()
X3 = pd.DataFrame()
for var in cont_vars:
    X2[f'{var}2'] = X_cont[var]**2
    X3[f'{var}3'] = X_cont[var]**3

# # Interaction terms
# X_cont_ind = pd.concat([X_cont, X_inds], axis=1)
# X_int = pd.DataFrame()
# for var1 in X_cont_ind.columns:
#     for var2 in X_cont_ind.columns:
#         if var1 != var2:
#             X_int[f'{var1}X{var2}'] = X_cont_ind[var1] * X_cont_ind[var2]

# Combine all features
df = pd.concat([X_cont, X_inds, X2, X3], axis=1)
df = pd.concat([df, data[['dur', 'cens', 'notice']]], axis=1)
#df = pd.concat([df, X_int], axis=1)
# Create 2 folds
np.random.seed(0)
folds = np.random.choice(2, size=len(data))
model_ra='best'
model_ps='best'
# IPW
from utils.DDML import IPW
ps = IPW(df, model_ps)

# Regression Adjustment
from utils.DDML import RegAdj
h_i = RegAdj(df, model_ra)

# Raw data
g, h, S = ImpliedMoms(data)
series = [h['raw'][:-1,j] for j in range(h['raw'].shape[1])]
CustomPlot(series)

# IPW Adjusted
g, h, S = ImpliedMoms(data, ps)
series = [h['ipw'][:-1,j] for j in range(h['raw'].shape[1])]
CustomPlot(series)

# Regression Adjusted
g, h, S = ImpliedMoms(data, ps, h_i)
series = [h['ra'][:-1,j] for j in range(h['raw'].shape[1])]
CustomPlot(series)

# Double Robust
series = [h['dr'][:-1,j] for j in range(h['raw'].shape[1])]
CustomPlot(series)


from utils.GMM import GMM
psiM_hat1, mu_hat1 = GMM(g['dr'][:-1,:], unstack=True)
psiM_hat2 = DDML(df, model_ra, model_ps, folds=folds)[0]

# Plot psiM_hat
plt.plot(psiM_hat1[:,1], label='DR-GMM')
plt.plot(psiM_hat2['dr'][:-1,1], label='DR')
plt.plot(g['raw'][:-1,1]/g['raw'][0,1], label='data')
plt.plot(psiM_hat2['ipw'][:-1,1], label='IPW')
plt.plot(psiM_hat2['ra'][:-1,1], label='RA')
plt.legend()

# Plot psiM with data
#series = [(psiM_hat[1:,1]/psiM_hat[1,1]).flatten().tolist(), (g['raw'][1:-1,1]/g['raw'][1,1]).flatten().tolist()]
#CustomPlot(series)
# # Plot raw data
# from utils.GMM import GMM
# data = data[data['notice'] >=3]
# data['notice'] = np.where(data['notice'] == 4, 1, 0)
# g, h, S = ImpliedMoms(data)
# import matplotlib.pyplot as plt
# series = [h['raw'][:-1,j] for j in range(2)]
# CustomPlot(series)
# psiM_hat, mu_hat = GMM(g['raw'][:-1,:], unstack=True)
#psiM_hat = DDML(df, model_ra='logit', model_ps='logit')[0]

# # Plot psiM_hat
#series = [psiM_hat['dr'][:-1,j] for j in range(2)]
#CustomPlot(series)