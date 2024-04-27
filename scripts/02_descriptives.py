##########################################################
# Housekeeping
##########################################################

# Import external libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from patsy import dmatrix

# Import custom functions
from utils.DataDesc import SumTab, CustomPlot, PredPS
from utils.DataMoms import DataMoms
from utils.EstGMM import GMM
from utils.EstHelpers import UnstackAll
from utils.Inference import StdErrors, IndvMoms, IndvMomsIPW

# Set np print options
np.set_printoptions(precision=4, suppress=True)

##########################################################
# Create old sample
##########################################################

# Load data
data_dir = 'data'
sample = pd.read_csv(f'{data_dir}/sample.csv')

# Sample previous
sample = sample[(sample['dwyears'] >= 0.5)]
sample = sample[(sample['dwhi'] == 2)]
sample = sample[(sample['dwnotice'] >= 3) & (sample['dwnotice'] <= 4)]
sample = sample[(sample['dwlastwrk'] >= 1) & (sample['dwlastwrk'] <= 3)]
sample = sample[(sample['dwjobsince'] <= 2)]
#sample = sample[(sample['jf'] == 1) | (sample['obsdur'] >= 52)]

# Remove missing?
rm_missing = True
if rm_missing:
    varlist = ['union', 'dwyears', 'dwweekl', 'ind', 'occ', 'obsdur', 'hi']
    sample = sample.dropna(subset=varlist)
    print(sample.isnull().sum()[sample.isnull().sum() != 0])
else:
    sample['union'] = np.where(sample['union'].isnull(), 0, sample['union'])

# Censoring & notice indicator
sample['cens'] = 1-sample['jf']
#sample['dur'] = sample['dur_9week']
sample['notice'] = sample['dwnotice']

# Print some descriptives
print(sample['notice'].value_counts().sort_index())
print(sample.groupby('notice')['cens'].mean())
print(sample.groupby('notice')['j2j'].mean())

##########################################################
# Prepare data for logit regression
##########################################################

# Specify controls
controls = ['age', 'female', 'married', 'black', 'col', 'dwreas', 
            'union', 'in_metro', 'dwyears', 'lnearnl', 'occ_cat', 
            'statefip', 'dyear', 'ind_cat', 'educ_cat']

# Keep only relevant variables
data = sample[controls + ['notice', 'dur', 'cens']].copy()

# Add quadratic terms for continuous variables
data['age2'] = data['age']**2
data['dwyears2'] = data['dwyears']**2

# Categories as numbers
data.loc[:, 'ind_cat'] = pd.Categorical(data['ind_cat']).codes
data.loc[:, 'occ_cat'] = pd.Categorical(data['occ_cat']).codes

# Create interactions vars between dyear and ind_cat
# int_terms = dmatrix("C(dyear):C(ind_cat)", data=data, return_type='dataframe')
# int_terms = int_terms.loc[:, int_terms.columns.str.contains('dyear')]
# int_terms = int_terms.iloc[:, 1:]
# for name in int_terms.columns:
#     print(name)

# Hot encode occ_cat, statefip, dyear, ind_cat
data = pd.get_dummies(data, columns=['dwreas', 'occ_cat', 'statefip',
                                     'dyear', 'ind_cat', 'educ_cat'], 
                                     drop_first=True)

# Final data for regression
#data = pd.concat([data, int_terms], axis=1)
#data = pd.concat([data], axis=1)

##########################################################
# Propensity scores and add balancing weights
##########################################################

# Estimate propensity scores
ps, coefs = PredPS(data)
h, se_h, S, se_S = DataMoms(data, ps, purpose='output')
h_unadj, se_h_unadj, S_unadj, se_S_unadj = DataMoms(data, purpose='output')

# Add balancing weights to the data
notvals = np.sort(sample['notice'].unique())
for j in range(len(notvals)):
    sample.loc[(sample['notice']==notvals[j]), 'wt'] = \
        1/ps[(sample['notice']==notvals[j]),j]

##########################################################
# Check overlap in propensity scores 
##########################################################

# Check overlap in propensity scores
plt.figure(figsize=(3, 2.75))
colors = ['blue', 'red', 'green', 'orange', 'purple']
for j in range(len(notvals)):
    # distribution of ps
    plt.hist(ps[data['notice']==notvals[j],1], bins=30, alpha=0.15, 
             label=f'Group {j+1}', color=colors[j])

##########################################################
# Summary statistics
##########################################################

# Variable list
varlist = ['age', 'female', 'married', 'black', 'col', 'pc', 
            'union', 'in_metro', 'dwyears', 'lnearnl']

# Variable labels
labels = ['Age', 'Female', 'Married', 'Black',
            'College Degree', 'Plant Closure', 'Union Membership',
            'In Metro Area', 'Years of Tenure', 'Log Earnings']

# Unbalanced
table1 = SumTab(sample, varlist, 'notice', labels, se=False, stars=True)
table1 = np.array(table1)

# Balanced
table2 = SumTab(sample, varlist, 'notice', labels, wts='wt', se=False, stars=True)
table2 = np.array(table2)[:,1:]

# Merge table 1 & 2 by column
extracol = np.array([['']*2*len(varlist)]).T
table = np.concatenate((table1, extracol, table2), axis=1)

# Print table to latex
with open('output/tab_sum_stats.tex', 'w') as f:
    f.write(tabulate(table, tablefmt='latex'))

# Remove unnecessary lines from the file
remove_lines = False
if remove_lines:
    with open('output/tab_sum_stats.tex', 'r') as f:
        lines = f.readlines()
    with open('output/tab_sum_stats.tex', 'w') as f:
        for line in lines:
            if not line.startswith('\\hline') and \
               not line.startswith('\\begin{tabular}') and \
               not line.startswith('\\end{tabular}'):
                f.write(line)


##########################################################
# Plot hazard rates
##########################################################

T, J = h.shape

# Raw Hazard rates
CustomPlot([h_unadj[:,j] for j in range(J)], 
           #legendlabs=['Short', 'Long'],
           #xticklabs=['0-12', '12-24', '24-36', '36-48'],
           ydist=0.1, savepath='output/hazard_raw.pdf')

# IPW: Hazard rates
CustomPlot([h[:,j] for j in range(J)], 
           #legendlabs=['Short', 'Long'],
           #xticklabs=['0-12', '12-24', '24-36', '36-48'],
           ydist=0.1, savepath='output/hazard_ipw.pdf')

##########################################################
# Estimates from the Structural Model
##########################################################

# Data hazard
nL = data['notice'].value_counts().sort_index().values
h_avg_ipw = h @ nL / nL.sum()
h_avg_raw = h_unadj @ nL / nL.sum()
nrm = 0.5
ffopt = 'baseline'

# Raw Data
thta_hat = GMM(data, nrm, ffopt)
se = StdErrors(thta_hat, data, nrm, ffopt, MomsFunc=IndvMoms)
psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
    UnstackAll(T, J, nL, thta_hat, se, nrm, ffopt)

# Plot raw with confidence intervals
h_avg_raw = psi[0] * h_avg_raw / h_avg_raw[0]
plt.figure(figsize=(3, 2.75))
plt.errorbar(np.arange(0, T), psi, yerr=1.96*psiSE,
              color='red', capsize=2, label='Structural', alpha=0.75)
plt.plot(h_avg_raw, label='Observed', color='black', linestyle='--')
plt.legend()
plt.savefig('output/est_raw.pdf')
plt.show()

# IPW
thta_hat = GMM(data, nrm, ffopt, ps)
thta_all = np.append(thta_hat, coefs)
se_adj = False
if se_adj:
    se = StdErrors(thta_all, data, nrm, ffopt, MomsFunc=IndvMomsIPW)
    se = se[:len(thta_hat)]
else:
    se = StdErrors(thta_hat, data, nrm, ffopt, MomsFunc=IndvMoms)
psin, psi, par, mu, psinSE, psiSE, parSE, muSE = \
    UnstackAll(T, J, nL, thta_hat, se, nrm, ffopt)

# Plot raw with confidence intervals
h_avg_ipw = psi[0] * h_avg_ipw / h_avg_ipw[0]
plt.figure(figsize=(3, 2.75))
plt.errorbar(np.arange(0, T), psi, yerr=1.96*psiSE,
              color='red', capsize=2, label='Structural', alpha=0.75)
plt.plot(h_avg_ipw, label='Observed', color='black', linestyle='--')
plt.legend()
plt.savefig('output/est_ipw.pdf')
plt.show()

