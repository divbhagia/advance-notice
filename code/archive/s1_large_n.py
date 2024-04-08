# Housekeeping
import sys
import numpy as np
sys.path.append('code')
from utils.SimFns import DGP, SimData
from code.utils.SimFnsOth import CoefPlot
from utils.AddDataFns import DurDistByNotice, BalancingWeights
from code.utils.GMMFns import ModelMoments, Unstack, GMM
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
np.random.seed(1117)

##########################################################
# Data generating process & simulate data
##########################################################

# Parameters
T = 6
J = 2   
n = 500000
opt = None # Default
colors = ['blue', 'red']

# DGP
psiM, mu, nuP, betaL, betaPhi, _, _ = DGP(T, J, opt)

# Simulate data for very large n
data, nu, pL_X, phiX = SimData(n, T, J, opt)
X = data[[col for col in data.columns if col.startswith('X')]]

# Few checks
notice = data['notice']
exit0 = (data['dur']==0).astype(int)
X = data[[col for col in data.columns if col.startswith('X')]]
nu[notice==0].mean(), nu[notice==1].mean()

# True mu_1, mu_2 | X
from utils.SimFns import BetaMoms
muMat = np.zeros((T, 3))
for k in range(3):
    muMat[:,k], _, _ = BetaMoms(T, nuP[k,0], nuP[k,1])
mu1 = np.zeros(n)
mu2 = np.zeros(n)
mu1[data['X7']==1] = muMat[0, 0]
mu2[data['X7']==1] = muMat[1, 0]
mu1[(data['X7']==0) & (data['X6']==1)] = muMat[0, 1]
mu2[(data['X7']==0) & (data['X6']==1)] = muMat[1, 1]
mu1[(data['X7']==0) & (data['X6']==0)] = muMat[0, 2]
mu2[(data['X7']==0) & (data['X6']==0)] = muMat[1, 2]

##########################################################
# Balancing weights & coefficients
##########################################################

# Generate balancing weights
data['wts'], coefs_l = BalancingWeights(data['notice'], X, out='all')

# Verify coeficients match up to true coefficients
CoefPlot(betaL[:,1].flatten(), coefs_l.flatten(), 
         'Coefficients for Pr(L=1|X)')

##########################################################
# Estimating E[E[I(D=1)|l, X]/E[I(D=1)|l', X]]
##########################################################

# Pick model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20, max_depth=10)

# Remove sparse variables to quicken estimation (beta_phi=0)
X_ = X.loc[:, betaPhi!=0]

# Fit model for exit0
preds = np.zeros((n, J))
for j in range(J):
    model.fit(X_ [notice==j], exit0[notice==j])
    preds[:,j] = model.predict_proba(X_)[:,1]
print(f'Estimate: {(preds[:,1]/preds[:,0]).mean():.2f}')
print(f'Data: {(exit0[notice==1].mean()/exit0[notice==0].mean()):.2f}')
print(f'True: {psiM[0,1]/psiM[0,0]:.2f}')

# Fit model for exit1
exit1 = ((data['dur']==1) & (data['cens']==0)).astype(int)
preds = np.zeros((n, J))
for j in range(J):
    model.fit(X_ [notice==j], exit1[notice==j])
    preds[:,j] = model.predict_proba(X_)[:,1]
print(f'Estimate: {(preds[:,1]/preds[:,0]).mean():.2f}')
print(f'Data: {(exit1[notice==1].mean()/exit1[notice==0].mean()):.2f}')
num = mu1 - psiM[0,1] * mu2
den = mu1 - psiM[0,0] * mu2
print(f'True: {(num/den).mean():.2f}')

##########################################################
# Duration distribution
##########################################################

# Duration distribution
durdist_ub = DurDistByNotice(data['dur'], data['cens'], 
                             data['notice'])
durdist = DurDistByNotice(data['dur'], data['cens'], 
                          data['notice'], data['wts'])

# To verify censoring
T_bar = round(T/2)
data_cens = data[data['cns_indctr']==1]
durdist_unadj = DurDistByNotice(data_cens['dur'], data_cens['cens'], 
                         data['notice'], data_cens['wts'], adj=False)

# Plot to verify censoring adjustment
plt.figure(figsize=[4, 3])
for j in range(J):
    plt.plot(durdist[durdist['notice']==j]['h'][:T_bar], colors[j])
    plt.plot(durdist_unadj[durdist_unadj['notice']==j]['h'][:T_bar], 
             colors[j], linestyle='dashed')
plt.title('Adjusting for censoring two ways')
plt.show()

# Plot balanced & unbalanced hazards for different notice lengths
plt.figure(figsize=[8, 3])
plt.subplot(1, 2, 1) 
for j in range(J):
    plt.plot(durdist_ub[durdist_ub['notice']==j]['h'], colors[j])
plt.title('Unbalanced')
plt.legend(['Short', 'Long'])
plt.subplot(1, 2, 2) 
for j in range(J):
    plt.plot(durdist[durdist['notice']==j]['h'], colors[j])
plt.title('Balanced')
plt.legend(['Short', 'Long'])
plt.show()

##########################################################
# Model implied vs data moments
##########################################################

# Model implied moments
g_model = ModelMoments(psiM, mu)
g_data = np.array([durdist[durdist['notice']==j]['g'] for j in range(J)]).T

# Plot g_model vs g_data
plt.figure(figsize=[4, 3])
for j in range(J):
    plt.plot(g_model[:,j], color=colors[j])
    plt.plot(g_data[:,j], color=colors[j], linestyle='dashed')
plt.title('Model implied vs data moments')

# GMM
if opt is None:
    nrm = mu[0,0]
else:
    nrm = mu[0]
x_hat = GMM(g_data, nrm)
psiM_hat, mu_hat = Unstack(T, x_hat, nrm)

# Plot estimated vs true psiM
plt.figure(figsize=[4, 3])
plt.plot(psiM[1:,0]/psiM[0,0])
plt.plot(psiM_hat[1:,0]/psiM_hat[0,0])

# # Plot estimated vs true mu
# plt.figure(figsize=[4, 3])
# plt.plot(mu_hat)
# plt.plot(mu)


##########################################################
# Outcome model
##########################################################

data['h0'] = (data['dur']==0).astype(int)
h0 = data['h0']
obsdur = data['dur']
notice = data['notice']
X = data[[col for col in data.columns if col.startswith('X')]]

h0_1 = h0[notice==1]
h0_0 = h0[notice==0]
X_1 = X[notice==1]
X_0 = X[notice==0]

model = LogisticRegression(max_iter=1000)
model.fit(X_1, h0_1)
coefs_1 = model.coef_.flatten()
h0_1_hat = model.predict_proba(X)[:,1]
model.fit(X_0, h0_0)
coefs_0 = model.coef_.flatten()
h0_0_hat = model.predict_proba(X)[:,1]

(h0_1_hat/h0_0_hat).mean()

h0_1_hat.mean()/h0_0_hat.mean()
h0_1.mean()/h0_0.mean()

psiM[0, 1]/psiM[0, 0]


