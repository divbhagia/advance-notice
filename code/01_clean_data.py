##########################################################
# Housekeeping
##########################################################

# Add utils to path
import sys; sys.path.append('code')

# Import external libraries
import pandas as pd
import numpy as np

# Import custom functions
from utils.OccInd import get_occ, get_broad_occ
from utils.OccInd import get_ind, get_broad_ind
from utils.DataFns import Indicator

# Specify directories & parameters
ipums_dir = 'data/raw/IPUMS-Extract'
cpi_file = 'data/raw/cpi99.txt'
gdp_file = 'data/raw/gdp_1977_2022_long.csv'
unemp_file = 'data/raw/yearly_state_ur.csv'
data_dir = 'data'
extract_ipums = False

##########################################################
# Extract IPUMS data or load previously extracted data
##########################################################

# Extract IPUMS data or load previously extracted data
if extract_ipums:
    from ipumspy import readers # type: ignore
    ddi = readers.read_ipums_ddi(f'{ipums_dir}/cps_00039.xml')
    df = readers.read_microdata(ddi, f'{ipums_dir}/cps_00039.dat')
    df.columns = df.columns.str.lower()
    df.to_csv(f'{ipums_dir}/cps_raw.csv')
else:
    df = pd.read_csv(f'{ipums_dir}/cps_raw.csv')

##########################################################
# Remove armed forces and keep select variables 
##########################################################
    
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

##########################################################
# Merge state-year level unemployment rate & GDP data 
##########################################################

unemp = pd.read_csv(unemp_file)
gdp = pd.read_csv(gdp_file)
df = pd.merge(df, unemp, on=['year', 'statefip'], how='left')
df = pd.merge(df, gdp, on=['year', 'statefip', 'state'], how='left')

# Tabulate missing values
df[df['ur'].isnull()]['year'].value_counts()
df[df['gdp'].isnull()]['year'].value_counts()

##########################################################
# Missing values & top codes for continous variables
##########################################################

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
varlist = ['dwjobsince', 'dwhrswkc', 'dwunion', 'dwlastwrk', 'dwhi']
for var in varlist:
    df[var] = np.where(df[var] >= 96, np.nan, df[var])
df['dwreas'] = np.where(df['dwreas'] >= 95, np.nan, df['dwreas'])

##########################################################
# CPI adjustment
##########################################################

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

##########################################################
# Generate additional variables
##########################################################

# Create new binary variables (preserves missing values)
df['female'] = Indicator(df['sex'], 2)
df['black'] = Indicator(df['race'], 200)
df['married'] = Indicator(df['marst'], [1, 2])
df['col'] = Indicator(df['educ'], 110, 'greater')
df['pc'] = Indicator(df['dwreas'], 1)
df['union'] = Indicator(df['dwunion'], 2)
df['hi'] = Indicator(df['dwhi'], 2)
df['jf'] = Indicator(df['dwjobsince'], 1, 'greater')
df['in_metro'] = Indicator(df['metro'], [2, 3, 4])
df['emp'] = Indicator(df['empstat'], [10, 11, 12])
df['unemp'] = Indicator(df['empstat'], [20, 21, 22])
df['nilf'] = Indicator(df['empstat'], [30, 36], 'range')

# Log earnings
df['lnearnl'] = np.log(df['dwweekl'])
df['lnearnc'] = np.log(df['dwweekc'])

# Displacement year, j2j, and notice
df['dyear'] = df['year'] - df['dwlastwrk']
df['dyear'] = np.where(df['dwlastwrk'].isna(), np.nan, df['dyear'])
df['j2j'] = np.where((df['dwwksun'] == 0) & (df['dwjobsince'] > 0), 1, 0)
df['j2j'] = np.where(df['dwjobsince'].isnull(), np.nan, df['j2j'])
df['notice'] = Indicator(df['dwnotice'], 4)

# Indicator for extended benefits
df = df.assign(ext = np.where((df['dyear'] >= 2001) & (df['dyear'] <= 2004) | 
                 (df['dyear'] >= 2008) & (df['dyear'] <= 2013), 1, 0))

# Convert race to 1 digit
df['race'] = df['race'].apply(lambda x: int(str(x)[0]))

# Generate education categories
df['educ_cat'] = np.where(df['educ'] <= 72, 'Less than HS', np.nan) 
df['educ_cat'] = np.where(df['educ'] == 73, 'HS Degree', df['educ_cat'])
df['educ_cat'] = np.where((80 <= df['educ']) & (df['educ'] <= 110), 'Some College', df['educ_cat']) 
df['educ_cat'] = np.where(df['educ']==111, 'College', df['educ_cat']) 
df['educ_cat'] = np.where(df['educ']>111, 'Graduate Degree', df['educ_cat']) 
df['educ_cat'] = np.where(df['educ']==999, np.nan, df['educ_cat']) 

##########################################################
# Unemployment duration
##########################################################

# Generate observed duration variable
df['obsdur'] = np.where(df['jf'] == 1, df['dwwksun'], df['durunemp'])

# Function to group duration
def group_dur(dur_var, interval):
    dur = 0
    for i in range(0, 53, interval):
        dur = np.where((dur_var >= i) & (dur_var < i + interval),
                        i + 0.5 * interval, dur)
    dur = np.where(dur_var > i, i + 0.5 * interval, dur)
    return dur

# Generate duration 4, 9, and 12 week intervals
df['dur'] = group_dur(df['obsdur'], 12)
df['dur_4week'] = group_dur(df['obsdur'], 4)
df['dur_9week'] = group_dur(df['obsdur'], 9)

# Leaving in first interval indicator
df['h0'] = (df['obsdur'] == 0).astype(int)
df['h0to12'] = (df['dur'] == 6).astype(int)

##########################################################
# Occupation & Industry Categories
##########################################################

df['occ'] = df['dwocc1990'].apply(get_occ)
df['occ_cat'] = df['occ'].apply(get_broad_occ)
df['ind'] = df['dwind1990'].apply(get_ind)
df['ind_cat'] = df['ind'].apply(get_broad_ind)

##########################################################
# Sample selection
##########################################################

# Sample selection 
dws = df[df['dws'] == 1]
sample = dws
sample = sample[(sample['year'] >= 1996) & (sample['year'] <= 2020)]
sample = sample[(sample['age'] >= 21) & (sample['age'] <= 65)]
sample = sample[(sample['dwfulltime'] == 2)]                    
sample = sample[(sample['dwclass'] <= 3)]                       
sample = sample[(sample['dwrecall'] != 2)]                     
#tmp = sample[(sample['nilf'] == 0) | (sample['jf'] == 1)]    
sample = sample[(sample['dwnotice'] >= 1) & (sample['dwnotice'] <= 4)]

# Remove missing values
print(sample.isnull().sum()[sample.isnull().sum() != 0])
varlist = ['dwlastwrk', 'dwunion', 'dwhi', 'dwyears', 'dwweekl',
           'dwjobsince', 'ind', 'occ', 'obsdur']
sample = sample.dropna(subset=varlist)
print(sample.isnull().sum()[sample.isnull().sum() != 0])

# Note obsdur is missing when jf=1 & dwwksun is missing

# Are there individuals who haven't found a job and left lf?
sample[(sample['jf'] == 0)]['nilf'].value_counts()

##########################################################
# Save data
##########################################################

sample.to_csv(f'{data_dir}/sample.csv', index=False)
dws.to_csv(f'{data_dir}/dws.csv', index=False)
df.to_csv(f'{data_dir}/cps.csv', index=False)

##########################################################
