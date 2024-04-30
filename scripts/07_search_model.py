import numpy as np
from scipy.optimize import minimize
from utils.datadesc import custom_plot
from utils.searchmodel import parameters, avg_opt
from utils.config import QUANTS_DIR

######################################################
# Initialization
######################################################

# Parameters
T = 4 
sgma, beta, rho, thta, w, b, a = parameters(T, Tb=3)
dlta0 = 1

# Load observed h and estimated psiH
h_data = np.load(f'{QUANTS_DIR}/h_avg_ipw.npy')
est = np.load(f'{QUANTS_DIR}/baseline_est_out.npy', allow_pickle=True)
est = est.item()
psiH = est['psi']

# Distance to find optimal dlta & nu 
def distance(x, h_opt, h_data, types):
    dlta = np.append(dlta0, x[:T-1])
    if types == 1:
        if len(x) != T: raise ValueError('len(x) must be equal to T')
        nu = np.array([x[-1]])
        p = np.array([1])
    elif types == 2:
        if len(x) != T+2: raise ValueError('len(x) must be equal to T+2')
        nu = x[T-1:T+1]
        p = np.array([x[-1], 1-x[-1]])
    r = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta, nu=nu, p=p)
    vec_dist = np.append(r['h_str'] - h_opt, r['h_obs'] - h_data)
    dist = np.linalg.norm(vec_dist)
    if dlta.min()<0 or nu.min()<0: 
        dist = 1e+10
    return dist

######################################################
# Calibrations
######################################################

# Optimization options
opt = {'disp': False, 'maxiter': 100000, 'ftol': 1e-12, 'xtol': 1e-2}
mthd = 'Powell'

# Calibration 1: h_str = h_obs, no heterogeneity
x0 = 0.5*np.ones(T)
x = minimize(distance, x0, args=(h_data, h_data, 1), 
             method=mthd, options=opt).x
dlta1 = np.concatenate([np.array([dlta0]), x[:T-1]])
nu1 = np.array([x[-1]])
r1 = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta1, nu=nu1)

# Calibration 2: h_str = h_obs, no heterogeneity
x0 = 0.5 * np.ones(T+2)
x = minimize(distance, x0, args=(psiH, h_data, 2), 
             method=mthd, options=opt).x
dlta2 = np.concatenate([np.array([dlta0]), x[:T-1]])
nu2 = x[T-1:T+1]
p2 = np.array([x[-1], 1-x[-1]])
print(nu2, p2)
r2 = avg_opt(T, sgma, beta, rho, thta, w, b, a, dlta2, nu=nu2, p=p2)

######################################################
# Assess model fit
######################################################

custom_plot([h_data, r1['h_obs'], r2['h_obs']], 
            legendlabs=['Observed', 'Calibration 1', 'Calibration 2'])


######################################################
# Plots
######################################################

labs = ['2-types of workers', 'No heterogeneity']

custom_plot([dlta2, dlta1], legendlabs=labs)
custom_plot([r2['s_str']/r2['s_str'][0], r1['s_str']/r1['s_str'][0]],
             legendlabs=labs)

######################################################

