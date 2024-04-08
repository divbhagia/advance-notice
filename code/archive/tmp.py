# Housekeeping
import sys
import numpy as np
import pandas as pd
sys.path.append('code')
from utils.SimFns import DGP, SimData, CoefPlot
from utils.MLFns import BestModel, ModelParams

##########################################################

# Parameters
T = 4
J = 3   
n, iters = 10000, 1
opt = 'fewvars' # None for default
opt = None
nrm = 0.5

# DGP
psiM, mu, nuP, beta_l, beta_phi, _, _ = DGP(T, J, opt)

# Simulate data
data, _ = SimData(n, T, J, opt)
X = data[[col for col in data.columns if col.startswith('X')]]

# Remove sparse variables to quicken estimation (beta_phi=0)
#X = X.loc[:, beta_l[:,1]!=0]



##########################################################
# Fit propensity score model 
##########################################################

# Divide data into training and testing
p_X, model = BestModel(X, data['notice'])


##########################################################
# Fit exit models
##########################################################



    