import numpy as np
from utils.searchmodel import avg_opt
from utils.datadesc import custom_plot

######################################################
# Tests
######################################################

# Parameters
T = 20
sgma = 1.75
beta = 0.985
rho = 1
thta = 50*np.ones(T)

# Payoffs
w0 = 1
a = 0.1 * w0
Tb = int(round(0.5 * T))
Tb = 10
r = 0.5
w = w0 * np.ones(T)
b1 = np.zeros(T)
b1[:Tb] = r * w0
b2 = np.zeros(T)
b2[:Tb+6] = r * w0  

# Duration dependence
dlta1 = 0.5 * np.ones(T)
for d in range(1, T):
    dlta1[d] = dlta1[d-1]**1  
dlta2 = dlta1.copy()

# Heterogeneity
nu1 = 1
p1 = 1
nu2 = np.array([1, 2])
p2 = np.array([0.9, 0.1])

# Solve
r1 = avg_opt(T, sgma, beta, rho, thta, w, b1, a, dlta1, nu=nu2, p=p2)
r2 = avg_opt(T, sgma, beta, rho, thta, w, b2, a, dlta2, nu=nu2, p=p2)

# Plots
custom_plot([r1['s_str'], r2['s_str']])
custom_plot([r1['h_obs'], r2['h_obs']])

######################################################
# Calibration
######################################################