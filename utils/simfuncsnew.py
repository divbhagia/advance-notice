import numpy as np
import pandas as pd
from utils.simhelpers import beta_moms_mixed, complex_ps, remove_sparse
from utils.esthelpers import psi_baseline

##########################################################

# Default DGP:
#   x0 is constant
#   x1 is Bernoulli with p = 0.5
#   x2 is Bernoulli with p = 0.2
#   X = (1, x1, x2) (3 dimensional)
#   L (notice) assignment depends on p_L = betaL * X
#   phi(X) = betaPhi * X
#   nu independent of L | X, 
#       nu ~ Beta(1, 3) if x7 = 1
#       nu ~ Beta(4, 1) if x7 = 0 & x6 = 1
#       nu ~ Beta(2, 2) if x7 = 0 & x6 = 0
#   psiPar define the parameters of the psi function
#   psin defines first period probs by notice length
# Note: with beta_L = (c, 0, 0), exogenous L

##########################################################
# Helper functions
##########################################################

# Log-Logistic PDF
def llfunc(x, a1, a2):
    return (a2/a1) * (x/a1)**(a2-1) / (1 + (x/a1)**a2) ** 2

# Log-Logistic CDF
def llhaz(par, T, c=0.2):
    a1, a2 = par[0], par[1]
    psi = np.zeros(T-1)
    for d in range(T-1):
        psi[d] = llfunc(c * (d-1), a1, a2)
    return psi

##########################################################
# Define DGP function
##########################################################

from utils.customplot import custom_plot

T = 10
psin = np.array([0.2, 0.5])
betaL = np.array([0.5, 0, 0])
betaPhi = np.array([0.5, 0, 0])
psiPar = np.array([1, 3])
psi = llhaz(psiPar, T)
custom_plot([psi])

import numpy as np
import matplotlib.pyplot as plt


# Define the range of values for a1 and a2
a1_values = np.linspace(1, 1, 1)
a2_values = np.linspace(0.5, 8, 10)  
a2_values = [0.5, 1, 2, 4, 8]

# Set T
T = 10

# Plot for different values of a1 and a2
plt.figure(figsize=(10, 6))
for a1 in a1_values:
    for a2 in a2_values:
        par = [a1, a2]
        psi = llhaz(par, T)
        x = np.linspace(0.1, 2, 20)
        psi = llfunc(x, a1, a2)
        plt.plot(x, psi, label=f'a1={a1}, a2={a2}')
plt.legend()
plt.grid(True)
plt.show()



def dgp(T=8, psin=[0.5, ], psiPar, betaL=None, betaPhi=None, dgpopt=None):

    # Initialize
    if betaL is None:
        betaL = np.array([0.5, 0, 0])
    if betaPhi is None:
        betaPhi = np.array([0.2, 0, 0])





