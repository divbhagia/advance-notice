# Housekeeping
import numpy as np
from utils.SimFns import DGP, SimData
from utils.DataFns import DurDist, DurDistByNotice, UnadjHazard, BalancingWeights
import matplotlib.pyplot as plt
np.random.seed(1117)

##########################################################
# Data generating process & simulate data
##########################################################

# Parameters
T = 8
J = 2   
n = 100000

# DGP
psiM, mu, nu_P, beta_l, beta_phi = DGP(T, J)

# Simulate data for very large n
n_large = 100000
data = SimData(n, T, J)

##########################################################
# Balancing weights & coefficients
##########################################################

# Generate balancing weights
X = data[[col for col in data.columns if col.startswith('X')]]
data['wts'], coefs = BalancingWeights(data['notice'], X, out='all')

# Verify coeficients match up to true coefficients
plt.figure(figsize=[5, 4])
plt.plot(beta_l[:,1].flatten(), 'blue')
plt.plot(coefs.flatten(), 'red', linestyle='dashed')
plt.title('Coefficients for Pr(L=1|X)')
plt.legend(['True', 'Estimated'])
plt.show()

##########################################################
# Hazard rate & survival rate
##########################################################

# Verify adjusted hazard = unadjusted hazard for censtime<T_bar
h, se_h = DurDist(data['obsdur'], data['cens'])
T_bar = round(T/2) # censind = 1 if censtime > T_bar 
h_unadj = UnadjHazard(data[data['cns_indctr']==1]['obsdur'])
plt.figure(figsize=[5, 4])
plt.plot(h[:T_bar], 'blue')
plt.plot(h_unadj[:T_bar], 'red')
plt.ylim([min(h)-0.1, max(h)+0.1])
plt.title('Adjusting for censoring two ways')
plt.show()

# Plot balanced & unbalanced hazards for different notice lengths
hV = DurDistByNotice(data['obsdur'], data['cens'], data['notice'])
hV_bal = DurDistByNotice(data['obsdur'], data['cens'], data['notice'], data['wts'])
color = ['blue', 'red']
# Unbalanced
plt.figure(figsize=[8, 4])
plt.subplot(1, 2, 1) 
for j in range(J):
    plt.plot(hV[:, j], color[j], linestyle='dashed')
plt.title('Unbalanced')
plt.legend(['Short', 'Long'])
plt.ylim([np.min(hV)-0.1, np.max(hV)+0.1])
# Balanced
plt.subplot(1, 2, 2) 
for j in range(J):
    plt.plot(hV_bal[:, j], color[j])
plt.title('Balanced')
plt.legend(['Short', 'Long'])
plt.ylim([np.min(hV)-0.1, np.max(hV)+0.1])
plt.show()


##########################################################
# Model Implied Moments
##########################################################

# Notes: h_adj = 

obsdur = data['obsdur']
cens = data['cens']

def DurDist(obsdur, cens, wts=None, adj=True):

    # Initialize
    durvals = np.sort(obsdur.unique())
    h = np.zeros(len(durvals))
    g = np.zeros_like(h)
    S = np.zeros_like(h)
    exit = np.zeros_like(h)
    surv = np.zeros_like(h)
    censd = np.zeros_like(h)

    ##########################################################
    # Not adjusted for censoring & unweighted        
    ##########################################################

    # Hazard rate
    for d in range(len(durvals)):
        exit[d] = np.sum(obsdur == durvals[d])
        surv[d] = np.sum(obsdur >= durvals[d])
    h = exit/surv

    # Survival rate & density
    S = np.cumprod(1 - h)
    S = np.append(1, S)
    g = h * S[:-1]

    # Verify hazard & survival (delete later)
    n = surv[0]
    g_verify = exit/n
    S_verify = surv/n

    # Standard errors
    se_h = np.sqrt(h*(1-h)/surv)
    se_S = np.sqrt(S*(1-S)/surv[0])
    se_g = np.sqrt(g*(1-g)/surv[0])

    # Covarian between h & S
    #cov_h1_S2 = 

    # Verify standard errors (delete later) using delta method
    # S[2] = (1-h(1)), SE_S[2] = sqrt(h(1)*(1-h(1))/n)
    se_S_chk = np.zeros_like(se_S)
    se_S_chk[1] = se_h[0]
    var = np.zeros_like(se_S)
    var[1] = se_h[0]**2
    for d in range(2, len(durvals)):
        var[d] = se_S_chk[d-1]**2 * S[d-1]**2 + \
                               se_h[d-1]**2 * (1-h[d-1])**2 + \
                                  2 * cov_hS[d-1] * (1-h[d-1]) * S[d-1]
        se_S_chk[d] = np.sqrt((se_S_chk[d-1]**2) * S[d-1]**2 + \
                               (se_h[d-1]**2) * (1-h[d-1])**2)
    
    ##########################################################
    # Adjusted for censoring & unweighted        
    ##########################################################

    # Hazard rate
    for d in range(len(durvals)):
        exit[d] = np.sum((obsdur == durvals[d]) & (cens == 0))
        censd[d] = np.sum((obsdur == durvals[d]) & (cens == 1))
        surv[d] = np.sum(obsdur >= durvals[d])
    h = exit/surv

    # Survival rate & density
    S = np.cumprod(1 - h)
    S = np.append(1, S[:-1])
    g = h * S

    # Stack columns
    dur_dist = np.column_stack((exit, censd, surv, h, S, g))
    # column names
    colnames = ['exit', 'censd', 'surv', 'h', 'S', 'g']
    dur_dist = pd.DataFrame(dur_dist, columns=colnames)

    # Verify hazard & survival (delete later)
    S_verify = np.zeros_like(S)
    S_verify[0] = 1
    for d in range(1, len(durvals)):
        S_verify[d] = (surv[d-1]-surv[d]-exit[d]) / surv[d-1]

    # Standard errors
    se_h = np.sqrt(h*(1-h)/surv)
    se_S = np.sqrt(S*(1-S)/surv[0])
    se_g = np.sqrt(g*(1-g)/surv[0])

    # Verify standard errors (delete later) using delta method
    # S[2] = (1-h(1)), SE_S[2] = sqrt(h(1)*(1-h(1))/n)

    se_S_3 = np.sqrt((se_h[1]**2) * (1-h[1])**2 + (se_h[2]**2) * (1-h[2])**2)
        
        

        
    h = h[:-1]
    return h

