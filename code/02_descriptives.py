import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load data
data_dir = 'data'
sample = pd.read_csv(f'{data_dir}/sample.csv')

# Keep if dwreas == 1 or 3 (closure or position abolished)
sample = sample[sample['dwreas'].isin([1, 3])]
sample = sample[sample['age'] >= 25]

################################################################
# Descritve statistics
################################################################

sample.groupby('dwnotice').size()
sample.groupby('j2j').size()
sample.groupby('jf').size()

# Plot dwreas props by notice length
props = sample.groupby('dwnotice')['dwreas'].value_counts(normalize=True).unstack()
plt.figure(figsize=(10, 5))
props.plot(kind='bar', stacked=True)
plt.legend(['Closure', 'Insufficient Work', 'Position Abolished'],
            title='Reason', loc='upper center', bbox_to_anchor=(0.5, -0.2),
            ncol=3)

# Variables by notice length
var = 'j2j'
props = sample.groupby('dwnotice')[var].mean()
props_lim = sample.groupby('dwnotice')[var].mean()
plt.figure(figsize=(8, 5))
plt.bar(props.index - 0.2, props, width=0.4, label='Full Sample')
plt.bar(props_lim.index + 0.2, props_lim, width=0.4, label='Closures & Layoffs Only')
plt.ylabel(var)
plt.xticks(props.index)
plt.legend()
plt.show()

# Density of vars of by notice length
var = 'dwyears'
plt.figure(figsize=(10, 5))
for l in sample['dwnotice'].unique():
    sample_i = sample[(sample['dwnotice'] == l) & (sample['age']>25) & (sample['age']<50)]
    sample_i[var].plot(kind='kde', label=l)
plt.ylabel('Density')
plt.xlabel(var)
plt.legend()
plt.show()

# Table for average notice by statefip
tmp = sample.groupby('state')['notice'].mean()

# group year into intervals
sample['year'].value_counts().sort_index()
sample['grpd_year'] = np.nan
sample['grpd_year'] = np.where(sample['year'].isin([1996, 1998]), '1996-1998', sample['grpd_year'])
sample['grpd_year'] = np.where(sample['year'].isin([2000, 2002]), '2000-2002', sample['grpd_year'])
sample['grpd_year'] = np.where(sample['year'].isin([2004, 2006]), '2004-2006', sample['grpd_year'])
sample['grpd_year'] = np.where(sample['year'].isin([2008, 2010]), '2008-2010', sample['grpd_year'])
sample['grpd_year'] = np.where(sample['year'].isin([2012, 2014]), '2012-2014', sample['grpd_year'])
sample['grpd_year'] = np.where(sample['year'].isin([2016, 2018, 2020]), '2016-2020', sample['grpd_year'])
sample['grpd_year'].value_counts().sort_index()




################################################################
# Plot hazard rates
################################################################

# Function to calculate adjusted hazard rate
def calc_hazard_adj(obsdur, cens):
    dur_range = obsdur.unique() 
    dur_range.sort()
    h = []
    for d in dur_range:
        num = np.sum((obsdur == d) & (cens == 0))
        den = np.sum(obsdur >= d)
        h.append(num/den)
    h = h[:-1]
    return h

# Function to calculate hazard rate
def calc_hazard(obsdur):
    dur_range = obsdur.unique() 
    dur_range.sort()
    h = []
    for d in dur_range:
        num = np.sum((obsdur == d))
        den = np.sum(obsdur >= d)
        h.append(num/den)
    h = h[:-1]
    return h

# Plot hazard rate for each notice length
obsdur = sample['dur']
cens = 1-sample['jf']
notice = sample['notice']
cats = notice.unique()
plt.figure(figsize=(10, 5))
for l in notice.unique():
    obsdur_i = obsdur[notice==l]
    cens_i = cens[notice==l]
    h_adj = calc_hazard_adj(obsdur_i, cens_i)
    h = calc_hazard(obsdur_i)
    plt.subplot(1, 2, 1)
    plt.title('Adjusted')
    plt.plot(h_adj, label=l)
    plt.subplot(1, 2, 2)
    plt.title('Unadjusted')
    plt.plot(h, label=l)
    plt.legend()
plt.show()

################################################################