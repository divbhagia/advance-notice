import numpy as np
import pandas as pd
from utils.datadesc import sum_tab
from tabulate import tabulate

# Data directory
from utils.config import DATA_DIR, OUTPUT_DIR

##########################################################
# Summary statistics
##########################################################

# Load data
sample = pd.read_csv(f'{DATA_DIR}/sample.csv')

# Variable list
varlist = ['age', 'female', 'married', 'black', 'col', 'pc', 
            'union', 'in_metro', 'dwyears', 'lnearnl']

# Variable labels
labels = ['Age', 'Female', 'Married', 'Black',
            'College Degree', 'Plant Closure', 'Union Membership',
            'In Metro Area', 'Years of Tenure', 'Log Earnings']

# Unbalanced
table1 = sum_tab(sample, varlist, 'notice', labels, se=False, stars=True)
table1 = np.array(table1)

# Balanced
table2 = sum_tab(sample, varlist, 'notice', labels, wts='wt', se=False, stars=True)
table2 = np.array(table2)[:,1:]

# Merge table 1 & 2 by column
extracol = np.array([['']*(2*len(varlist)+1)]).T
table = np.concatenate((table1, extracol, table2), axis=1)

# Print table to latex
with open(f'{OUTPUT_DIR}/tab_sum_stats.tex', 'w') as f:
    f.write(tabulate(table, tablefmt='latex'))

# Remove unnecessary lines from the file
remove_lines = True
if remove_lines:
    with open(f'{OUTPUT_DIR}/tab_sum_stats.tex', 'r') as f:
        lines = f.readlines()
    with open(f'{OUTPUT_DIR}/tab_sum_stats.tex', 'w') as f:
        for line in lines:
            if not line.startswith('\\hline') and \
               not line.startswith('\\begin{tabular}') and \
               not line.startswith('\\end{tabular}'):
                f.write(line)

##########################################################