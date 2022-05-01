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

def normalizing_flow(X_train, X_test):
    X_train_flow = torch.zeros(X_train.shape[0], X_train.shape[1])
    X_test_flow = torch.zeros(X_test.shape[0], X_test.shape[1])

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    p = X_train.shape[1]

    u = 5
    w = torch.rand(p)*30
    b = 0.01
    # h is sigmoid function

    for row in range(n_train):
        X_train_flow[row,:] = X_train[row,:] + u * torch.sigmoid(w.dot(X_train[row,:]) + b)
    for row in range(n_test):
        X_test_flow[row,:] = X_test[row,:] + u * torch.sigmoid(w.dot(X_test[row,:]) + b)

    return X_train_flow, X_test_flow