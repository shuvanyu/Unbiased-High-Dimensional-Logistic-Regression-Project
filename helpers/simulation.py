import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.model_selection import train_test_split

## Data Generating Process #################################################
# The data generating process is parameterized by gamma, ratio, n
# Step-1 - Draw n examples of covariate vectors X from a given distribution
# Step-2 -  calculate the mu
# Step-3 - Calculate beta 
# Step 4 - Finally draw Y ~ Bernoulli(proba)
############################################################################
def simulate_gaussian(ratio, n, p, gamma):
    mu = torch.zeros(p)
    x = torch.randn(p,n)*0.009
    cov = torch.cov(x)

    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X = x_dist.sample([n,])

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test