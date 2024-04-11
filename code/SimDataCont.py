# Housekeeping
import sys
sys.path.append('code')

import numpy as np
import pandas as pd
from scipy.integrate import quad as integrate
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter 

# Function for structural hazard
def f_h(t): 
    lmbda, gamma = 0.3, -0.05
    h_t = lmbda * np.exp(gamma * t)
    return h_t

# Implied survival function
def f_S(t):
    H_t = integrate(f_h, 0, t)[0] 
    S_t = np.exp(-H_t)
    return S_t

# Given hazard function generate duration data
def GenDur(n, censtime=100):
    crit = np.random.uniform(size=n)
    dur = np.zeros(n)
    cens = np.zeros(n)
    for i in range(n):
        while True:
            if 1-f_S(dur[i]) > crit[i]:
                cens[i] = 0
                break
            dur[i] += 0.01
            if dur[i] >= censtime:
                cens[i] = 1
                break
    return dur, cens  


# Generate unemployment durations
n = 10000
dur, cens = GenDur(n)
cens.mean()

# Plot durations
plt.figure(figsize=(3, 2))
plt.hist(dur[cens==0], bins=10)
plt.title('Unemployment Durations')

# Empirical survival function
def EMP_S(dur, step):
    t_vals = np.arange(0, dur.max(), step)
    S = np.zeros(len(t_vals))
    for i, t in enumerate(t_vals):
        S[i] = (dur >= t).mean()
    return t_vals, S

# Empirical Hazard
def EMP_h(dur, step):
    t_vals, S = EMP_S(dur, step)
    h = np.zeros(len(t_vals)+1)
    for t in range(len(t_vals)-1):
        h[t] = (S[t]-S[t+1])/(S[t])
    h = h[1:]
    return t_vals, h

# Generate h & S for a range of values
Tp = 40
t_vals = np.arange(0, Tp, 0.1)
h = [f_h(t) for t in t_vals]
S = [f_S(t) for t in t_vals]
h, S = np.array(h), np.array(S)

# Empirical hazard & survival functions
step=3
tvals_h, emp_h = EMP_h(dur, step)
tvals_S, emp_S = EMP_S(dur, step)
kmf = KaplanMeierFitter()
kmf.fit(dur, event_observed=1-cens)

plt.figure(figsize=(3, 2))
plt.plot(t_vals, S)
plt.plot(tvals_S[:Tp], emp_S[:Tp], linestyle='--')
kmf.plot_survival_function(color='black', linestyle=':')
plt.title('Survival Function')
plt.show()

plt.figure(figsize=(3, 2))
plt.plot(t_vals, h)
plt.plot(tvals_h[:int(Tp//step)], emp_h[:int(Tp//step)], linestyle='--')
plt.title('Hazard Function')
plt.show()

def f_h_bin(a, b):
    h_t = integrate(f_h, a, b)[0]/(b-a)
    return h_t
    
f_h_bin(0, 3)
f_h(0)
emp_h[:10]
h[:10]
