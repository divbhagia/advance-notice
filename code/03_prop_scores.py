import numpy as np
import pandas as pd

################################################################
# Prepare data
################################################################

# Load data
data_dir = 'data'
sample = pd.read_csv(f'{data_dir}/sample.csv')

# Keep if dwyears>=1
sample = sample[sample['dwyears'] >= 1]

# Generate education categories
sample['educ_cat'] = np.where(sample['educ'] <= 72, 1,0) # Less than high school
sample['educ_cat'] = np.where(sample['educ'] == 73, 2, sample['educ_cat']) # High school
sample['educ_cat'] = np.where((80 <= sample['educ']) & (sample['educ'] <= 110), 3, sample['educ_cat']) # Some college
sample['educ_cat'] = np.where(sample['educ']==111, 4, sample['educ_cat']) # College
sample['educ_cat'] = np.where(sample['educ']>111, 5, sample['educ_cat']) # Advanced degree
sample['educ_cat'] = np.where(sample['educ']==999, np.nan, sample['educ_cat']) # Unknown
sample['educ'].value_counts().sort_index()
sample['educ_cat'].value_counts().sort_index()

# Collect pre-notice variables
sample.columns
cat_vars = ['year', 'statefip', 'metro', 'sex', 'marst', 'educ_cat', 'race', 
            'dwlastwrk', 'dwunion', 'dwhi', 'ind_cat', 'occ_cat', 'dwreas']
cont_vars = ['age', 'dwyears', 'dwweekl', 'ur', 'gdp']

# Count missing values
sample[cat_vars].isnull().sum()
sample[cont_vars].isnull().sum()

# Remove observations with missing values
df = sample.dropna(subset=cat_vars + cont_vars)

# Remove more missing values
df = df[df['dwunion']<96]
df = df[df['dwlastwrk']<97]
df = df[df['dwhi']<96]

# make race 1 digit from 3 digits, eg, 200, 201 = 2
df['race'] = df['race'].apply(lambda x: int(str(x)[0]))

# Remove if industry or occupation category is 'Unknown'
df = df[df['ind_cat']!='Unknown']
df = df[df['occ_cat']!='Unknown']

# Tabulate all categorical variables and present table properly
# for var in cat_vars:
#     print(df[var].value_counts())
#     print('\n')

# Summarize continuous variables
# df[cont_vars].describe()
# df['dwnotice'].value_counts()
# df['dur'].describe()
# df['censdur'].describe()


# Create fresh copy of data
df['cens'] = 1-df['jf']
df['notice'] = df['dwnotice']
data = df[cont_vars + cat_vars + ['dur'] + ['cens'] + ['notice']]

# Hot encode categorical variables
data = pd.get_dummies(data, columns=cat_vars, drop_first=True)

from utils.DDML import DDML, ImpliedMoms
import matplotlib.pyplot as plt
from utils.DataFns import CustomPlot


# Plot raw data
from utils.GMM import GMM
data = data[data['notice'] >=3]
data['notice'] = np.where(data['notice'] == 4, 1, 0)
g, h, S = ImpliedMoms(data)
import matplotlib.pyplot as plt
series = [h['raw'][:-1,j] for j in range(2)]
CustomPlot(series)
psiM_hat, mu_hat = GMM(g['raw'][:-1,:], unstack=True)
psiM_hat = DDML(data, model_ra='rf', model_ps='rf')[0]

# Plot psiM_hat
series = [psiM_hat['dr'][:-1,j] for j in range(2)]
CustomPlot(series)