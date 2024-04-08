import pandas as pd
from ipumspy import readers
import numpy as np
from OccInd import get_occ, get_broad_occ
from OccInd import get_ind, get_broad_ind

################################################################
# Initialise 
################################################################

# Specify directories & parameters
ipums_dir = 'data/raw/IPUMS-Extract'
cpi_file = 'data/raw/cpi99.txt'
gdp_file = 'data/raw/gdp_1977_2022_long.csv'
unemp_file = 'data/raw/yearly_state_ur.csv'
data_dir = 'data'
extract_ipums = False

# Extract IPUMS data or load previously extracted data
if extract_ipums:
    ddi = readers.read_ipums_ddi(f'{ipums_dir}/cps_00039.xml')
    df = readers.read_microdata(ddi, f'{ipums_dir}/cps_00039.dat')
    df.columns = df.columns.str.lower()
    df.to_csv(f'{ipums_dir}/cps_raw.csv')
else:
    df = pd.read_csv(f'{ipums_dir}/cps_raw.csv')

################################################################
# Remove armed forces and keep select variables 
################################################################
    
# Filter armed forces & children under 14/15
df = df[df['popstat']==1]

# Indicator for dws
df['dws'] = np.where((df['dwstat']==1) & (df['dwresp']==2), 1, 0)

# Select variables from cps
cps_oth_vars = ['serial', 'cpsid', 'mish', 'pernum', 'hwtfinl', 'wtfinl']
cps_cat_vars = ['year', 'month', 'statefip', 'metro', 'metarea', 'county', 
                'faminc', 'sex', 'marst', 'empstat', 'labforce', 'occ1990', 
                'ind1990', 'classwkr', 'numjob', 'educ', 'race', 'durunem2']
cps_cont_vars = ['age', 'uhrsworkt', 'uhrswork1', 'durunemp']

# Select variables from dws
dws_oth_vars = ['dwstat', 'dwresp', 'dwrecall', 'dwfulltime', 'dwclass', 
                'dwsuppwt', 'dws']
dws_cat_vars = ['dwreas', 'dwnotice', 'dwlastwrk', 'dwunion', 'dwben', 
                'dwexben', 'dwhi', 'dwind1990', 'dwocc1990', 
                'dwmove', 'dwhinow']
dws_cont_vars = ['dwyears', 'dwweekl', 'dwweekc', 'dwjobsince', 
                 'dwhrswkc', 'dwwksun']

# Keep selected variables
df = df[cps_oth_vars + cps_cat_vars + cps_cont_vars + \
        dws_oth_vars + dws_cat_vars + dws_cont_vars]

# Declare all variables as intergers
df = df.astype(int, errors='ignore')

################################################################
# Merge state-year level unemployment rate & GDP data 
################################################################

unemp = pd.read_csv(unemp_file)
gdp = pd.read_csv(gdp_file)
df = pd.merge(df, unemp, on=['year', 'statefip'], how='left')
df = pd.merge(df, gdp, on=['year', 'statefip', 'state'], how='left')

# Tabulate missing values
df[df['ur'].isnull()]['year'].value_counts()
df[df['gdp'].isnull()]['year'].value_counts()

################################################################
# Missing values & top codes for continous variables
################################################################

# CPS variables
df['uhrsworkt'] = np.where(df['uhrsworkt'] >= 997, np.nan, df['uhrsworkt'])
df['uhrswork1'] = np.where(df['uhrswork1'] == 0, np.nan, df['uhrswork1'])
df['uhrswork1'] = np.where(df['uhrswork1'] >= 997, np.nan, df['uhrswork1'])
df['faminc'] = np.where(df['faminc'] >= 995, 995, df['faminc'])
df['durunemp'] = np.where(df['durunemp'] == 999, np.nan, df['durunemp'])
df['durunem2'] = np.where(df['durunem2'] == 99, np.nan, df['durunem2'])
df['numjob'] = np.where(df['numjob'] == 0, np.nan, df['numjob'])

# Tenure at lost job
df['dwyears'] = np.where(df['dwyears'] > 99, np.nan, df['dwyears'])
df['dwyears'] = np.where(df['dwyears'] > 24, 24, df['dwyears'])

# Earnings at lost and current job
df['dwweekl'] = np.where(df['dwweekl'] > 9999, np.nan, df['dwweekl'])
df['dwweekc'] = np.where(df['dwweekc'] > 9999, np.nan, df['dwweekc'])
df['dwweekl'] = np.where(df['dwweekl'] == 0, np.nan, df['dwweekl'])
df['dwweekc'] = np.where(df['dwweekc'] == 0, np.nan, df['dwweekc'])

# Other DWS variables
df['dwwksun'] = np.where(df['dwwksun'] >= 996, np.nan, df['dwwksun'])
df['dwjobsince'] = np.where(df['dwjobsince'] >= 95, np.nan, df['dwjobsince'])
df['dwhrswkc'] = np.where(df['dwhrswkc'] >= 96, np.nan, df['dwhrswkc'])

################################################################
# CPI adjustment
################################################################

# Load CPI data
cpi99 = pd.read_csv(cpi_file, sep="\t", header=None, 
                    comment="#", usecols=[0, 3])
cpi99.columns = ['year', 'cpi99']

# Merge CPI data with CPS data
df = pd.merge(df, cpi99, on='year')

# Adjust dollar values for inflation
df['dwweekl'] = df['dwweekl'] * df['cpi99']
df['dwweekc'] = df['dwweekc'] * df['cpi99']
df['faminc'] = df['faminc'] * df['cpi99']

################################################################
# Generate additional variables
################################################################

df = df.assign(
    female = np.where(df['sex'] == 2, 1, 0),
    black = np.where(df['race'] == 200, 1, 0),
    married = np.where((df['marst'] == 1) | (df['marst'] == 2), 1, 0),
    col = np.where(df['educ'] >= 110, 1, 0),
    lnearnl = np.log(df['dwweekl']),
    lnearnc = np.log(df['dwweekc']),
    pc = np.where(df['dwreas'] == 1, 1, 0),
    union = np.where(df['dwunion'] == 2, 1, 0),
    hi = np.where(df['dwhi'] == 2, 1, 0),
    jf = np.where(df['dwjobsince'] != 0, 1, 0),
    notice = np.where(df['dwnotice'] == 4, 1, 0),
    j2j = np.where((df['dwwksun'] == 0) & (df['dwjobsince'] > 0), 1, 0),
    dyear = df['year'] - df['dwlastwrk'],
    in_metro = np.where((df['metro'] >= 2) & (df['metro'] <= 4), 1, 0),
    emp = np.where((df['empstat'] >= 10) & (df['empstat'] <= 12), 1, 0),
    unemp = np.where((df['empstat'] >= 20) & (df['empstat'] <= 22), 1, 0),
    nilf = np.where((df['empstat'] >= 30) & (df['empstat'] <= 36), 1, 0),
)
df = df.assign(ext = np.where((df['dyear'] >= 2001) & (df['dyear'] <= 2004) | 
                 (df['dyear'] >= 2008) & (df['dyear'] <= 2013), 1, 0))

################################################################
# Unemployment duration
################################################################

# Generate observed duration variable
df['censdur'] = np.where(df['jf'] == 1, df['dwwksun'], df['durunemp'])

# Function to group duration
def group_dur(dur_var, interval):
    dur = 0
    for i in range(0, 53, interval):
        dur = np.where((dur_var >= i) & (dur_var < i + interval),
                        i + 0.5 * interval, dur)
    dur = np.where(dur_var > i, i + 0.5 * interval, dur)
    return dur

# Generate duration 4, 9, and 12 week intervals
df['dur'] = group_dur(df['censdur'], 12)
df['dur_4week'] = group_dur(df['censdur'], 4)
df['dur_9week'] = group_dur(df['censdur'], 9)

# Leaving in first interval indicator
df['h0'] = (df['censdur'] == 0).astype(int)
df['h0to12'] = (df['dur'] == 6).astype(int)

################################################################
# Occupation & Industry Categories
################################################################

df['occ'] = df['dwocc1990'].apply(get_occ)
df['occ_cat'] = df['occ'].apply(get_broad_occ)
df['ind'] = df['dwind1990'].apply(get_ind)
df['ind_cat'] = df['ind'].apply(get_broad_ind)

################################################################
# Sample selection
################################################################

# Sample selection 
dws = df[df['dws'] == 1]
sample = dws
sample = sample[(sample['year'] >= 1996) & (sample['year'] <= 2020)]
sample = sample[(sample['age'] >= 21) & (sample['age'] <= 65)]
sample = sample[(sample['dwfulltime'] == 2)]                    
sample = sample[(sample['dwclass'] <= 3)]                       
sample = sample[(sample['dwrecall'] != 2)]                     
sample = sample[(sample['nilf'] == 0) | (sample['jf'] == 1)]    
sample = sample[sample['dwjobsince'].isnull()==False]           
sample = sample[(sample['jf'] == 0) | (sample['dwwksun'].isnull()==False)] 
sample = sample[(sample['dwnotice'] >= 1) & (sample['dwnotice'] <= 4)]

# Remove missing values
#sample = sample.dropna(subset=['lnearnl', 'dwyears'])
#sample = sample[sample['ind_cat'] != "Unknown"]
#sample = sample[sample['occ_cat'] != "Unknown"]

################################################################
# Save data
################################################################

sample.to_csv(f'{data_dir}/sample.csv', index=False)
dws.to_csv(f'{data_dir}/dws.csv', index=False)
df.to_csv(f'{data_dir}/cps.csv', index=False)

################################################################
