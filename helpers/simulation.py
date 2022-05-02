import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    # Normalize data
    scaler = StandardScaler()
    X_train_flow = torch.from_numpy(scaler.fit_transform(X_train_flow.numpy()))
    X_test_flow = torch.from_numpy(scaler.fit_transform(X_test_flow.numpy()))

    return X_train_flow, X_test_flow

def simulate_latent_gaussian(ratio, n, p, gamma, latent_ratio):
    num_latent = int(np.rint(latent_ratio * p))
    num_not_latent = p - num_latent

    # Generate the latent variable as a Gaussian
    z_dist = Normal(loc = 0, scale = 1)
    Z = z_dist.sample([n,])
    
    # Generate covariates as a noisy linear function of Z
    X_latent = torch.zeros(n, num_latent)
    for col in range(num_latent):
        a = torch.rand(1)
        b = torch.rand(n)
        X_latent[:, col] = a * Z + b

    # Generate remaining covariates as jointly Gaussian
    mu = torch.zeros(num_not_latent)
    x = torch.randn(num_not_latent,n)*0.009
    cov = torch.cov(x)

    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X_not_latent = x_dist.sample([n,])

    # Concatenate
    X = torch.cat((X_latent, X_not_latent), 1)
    print('Correlation Matrix for X:')
    print(torch.corrcoef(X))
    
    # Normalize X
    scaler = StandardScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test