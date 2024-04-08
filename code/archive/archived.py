####################################################################
# Simulate data for very large n
data, nu = SimData(n, T, J, opt)
durdist = DurDistByNotice(data['obsdur'], data['cens'], data['notice'])

# Prepare data
X = data[[col for col in data.columns if col.startswith('X')]]
X_ra = X.loc[:, beta_phi!=0] # remove sparse vars for now
X_ps = X.loc[:, beta_l!=0] # remove sparse vars for now
X_both = pd.concat([X_ra, X_ps], axis=1)

g, h, S = DataMoments(data['obsdur'], data['cens'], data['notice'], 
                      X_ps, X_ra, model_ps = 'rf', model_ra = 'logit')
keys = ['raw', 'ipw', 'ra', 'dr']
labels = ['Unadjusted', 'IPW', 'Reg. Adjusted', 'Doubly Robust']

# Plot hazard rates
plt.figure(figsize = (8, 5))
for i in range(len(keys)):
    plt.subplot(2, 2, i+1)
    plt.plot(h[keys[i]])
    plt.title(labels[i])
    plt.ylim(0, 0.5)
plt.tight_layout()
plt.show()

# Estimate psi & plot for each method
nrm = 0.5
plt.figure(figsize = (8, 5))
for i in range(len(keys)):
    psiM_hat, _ = Unstack(T, GMM(g[keys[i]], nrm), nrm)
    plt.subplot(2, 2, i+1)
    plt.plot(psiM_hat[:, 0]/psiM_hat[0, 0], label='Estimate', color='black', linestyle='--')
    plt.plot(psiM[:, 0]/psiM[0, 0], label='True', color='red')
    plt.title(labels[i])
    plt.legend()
    plt.ylim(0.5, 1.2*max(psiM[:, 0]/psiM[0, 0]))
plt.tight_layout()
plt.show()
