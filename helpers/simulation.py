import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.studentT import StudentT
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from helpers.models_test import test_baseline, test_sloe
from helpers.plotting import plot_conf_ints, plot_p_vals
plt.rcParams['text.usetex'] = True
np.random.seed(1)
torch.manual_seed(1)


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

    # Normalize data
    scaler = RobustScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test

def simulate_non_gaussian(ratio, n, p, gamma):
    # Simulate Gaussian data
    mu = torch.zeros(p)
    x = torch.randn(p,n)*0.009
    cov = torch.cov(x)

    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X = x_dist.sample([n,])
    
    # Transform it using normalizing flow
    X_flow = torch.zeros(n, p)
    u = 5
    w = torch.rand(p)*30
    b = 0.01
    # h is sigmoid function

    for row in range(n):
        X_flow[row,:] = X[row,:] + u * torch.sigmoid(w.dot(X[row,:]) + b)

    # Normalize data
    scaler = RobustScaler()
    X_flow = torch.from_numpy(scaler.fit_transform(X_flow.numpy()))

    # Generate y
    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X_flow,beta))
    y_flow = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X_flow, y_flow, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test

def simulate_latent_gaussian(ratio, n, p, gamma, latent_ratio=0.8):
    num_latent = int(np.rint(latent_ratio * p))
    num_not_latent = p - num_latent

    # Generate the latent variable as a Gaussian
    z_dist = Normal(loc = 0, scale = 1)
    #Z = z_dist.sample([n,])
    Z = z_dist.sample()
    
    # Generate covariates as a noisy linear function of Z
    X_latent = torch.zeros(n, num_latent)
    for col in range(num_latent):
        #a = torch.rand(1)
        #b = torch.rand(n)
        #X_latent[:, col] = a * (Z**3) + b
        X_latent[:,col] = Normal(loc = Z, scale = 1).sample([n,])

    # Generate remaining covariates as jointly Gaussian
    mu = torch.zeros(num_not_latent)
    x = torch.randn(num_not_latent,n)*0.009
    cov = torch.cov(x)

    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X_not_latent = x_dist.sample([n,])

    # Concatenate
    X = torch.cat((X_latent, X_not_latent), 1)

    # Normalize data
    scaler = RobustScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    # Shuffle columns
    shuffle_idxs = np.random.permutation(range(0,X.shape[1]))
    X = X[:,shuffle_idxs]

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test

def simulate_latent_non_gaussian(ratio, n, p, gamma, latent_ratio=0.8):
    num_latent = int(np.rint(latent_ratio * p))
    num_not_latent = p - num_latent

    # Generate the latent variable as a Gaussian
    z_dist = Normal(loc = 0, scale = 1)
    #Z = z_dist.sample([n,])
    Z = z_dist.sample()

    # Transform it using normalizing flow
    u = 5
    w = torch.rand(1)*30
    b = 0.01
    #h is sigmoid function
    Z = Z + u * torch.sigmoid(w*Z + b)

    X_latent = torch.zeros(n, num_latent)
    for col in range(num_latent):
        #a = torch.rand(1)
        #b = torch.rand(n)
        #X_latent[:, col] = a * (Z ** 3) + b
        X_latent[:, col] = Normal(loc = Z, scale = 1).sample([n,])[:,0]

    # Transform it using normalizing flow
    X_flow = torch.zeros(n, num_latent)
    u = 5
    w = torch.rand(num_latent)*30
    b = 0.01
    # h is sigmoid function

    for row in range(n):
        X_flow[row,:] = X_latent[row,:] + u * torch.sigmoid(w.dot(X_latent[row,:]) + b)

    X_latent = X_flow

    # Generate remaining covariates using normalizing flow
    mu = torch.zeros(num_not_latent)
    x = torch.randn(num_not_latent,n)*0.009
    cov = torch.cov(x)

    # Generate Gaussian first
    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X_not_latent = x_dist.sample([n,])

    # Transform it using normalizing flow
    X_flow = torch.zeros(n, num_not_latent)
    u = 5
    w = torch.rand(num_not_latent)*30
    b = 0.01
    # h is sigmoid function

    for row in range(n):
        X_flow[row,:] = X_not_latent[row,:] + u * torch.sigmoid(w.dot(X_not_latent[row,:]) + b)

    X_not_latent = X_flow
    
    # Concatenate
    X = torch.cat((X_latent, X_not_latent), 1)

    # Normalize data
    scaler = RobustScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    # Shuffle columns
    shuffle_idxs = np.random.permutation(range(0,X.shape[1]))
    X = X[:,shuffle_idxs]

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test

def simulate_cauchy(ratio, n, p, gamma):
    x_dist = StudentT(df=3)
    X = x_dist.sample([n,p])

    # Normalize data
    scaler = RobustScaler()
    X = torch.from_numpy(scaler.fit_transform(X.numpy()))

    beta = torch.zeros(p)
    beta[:p//8] = 2*gamma/np.sqrt(p)
    beta[p//8:p//4] = -2*gamma/np.sqrt(p)

    proba = torch.sigmoid(torch.matmul(X,beta))
    y = torch.bernoulli(proba)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    return X_train, X_test, y_train, y_test

def run_test(ratio, n, p, ci, title, type, gamma=np.sqrt(5), X=None, y=None, is_max=False, latent_ratio=None):
    if type == 'gaussian':
        func = simulate_gaussian
    elif type == 'non_gaussian':
        func = simulate_non_gaussian
    elif type == 'latent_gaussian':
        func = simulate_latent_gaussian
    elif type == 'latent_non_gaussian':
        func = simulate_latent_non_gaussian
    elif type == 'heavy':
        func = simulate_cauchy
    
    if type != 'heart':
        # Simulate data
        X_train, X_test, y_train, y_test = func(ratio, n, p, gamma)
    elif type == 'latent_gaussian' or type == 'latent_non_gaussian':
        X_train, X_test, y_train, y_test = func(ratio, n, p, gamma, latent_ratio=latent_ratio)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.8, random_state=0)

    # Test with baseline model
    p_vals_baseline, pred_ints_baseline, score_baseline, performance_baseline = test_baseline(X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(), ci=ci)

    # Test with SLOE Model
    p_vals_sloe, alpha, pred_ints_sloe, score_sloe, performance_sloe = test_sloe(X_train.numpy(), y_train.numpy(), X_test.numpy(), y_test.numpy(), ci=ci)

    # Generate plots
    if (p_vals_baseline is not None) and (pred_ints_baseline is not None) and (p_vals_sloe is not None) and (pred_ints_sloe is not None):
        if is_max and type != 'heart':
            plot_p_vals(p_vals_baseline, p_vals_sloe, title, type, n, ratio, latent_ratio)
        plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test.numpy(), y_test.numpy(), title, type, n, ratio, latent_ratio, ci=ci)

    # Output score dict
    return {
        'type': type,
        'n': n,
        'ratio': ratio,
        'latent_ratio': latent_ratio,
        'alpha': alpha,
        'baseline F1 score': score_baseline,
        'SLOE F1 score': score_sloe,
        'baseline time (s)': performance_baseline,
        'SLOE time (s)': performance_sloe
    }

def test_normalizing_flow(ratio, n, p, gamma):
    # Simulate Gaussian data
    mu = torch.zeros(p)
    x = torch.randn(p,n)*0.009
    cov = torch.cov(x)

    x_dist = MultivariateNormal(loc = mu, covariance_matrix = cov)
    X = x_dist.sample([n,])
    
    # Transform it using normalizing flow
    X_flow = torch.zeros(n, p)
    u = 5
    w = torch.rand(p)*30
    b = 0.01
    # h is sigmoid function

    for row in range(n):
        X_flow[row,:] = X[row,:] + u * torch.sigmoid(w.dot(X[row,:]) + b)

    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].hist(X[:,0].numpy(), bins=50)
    ax[0].set_title(r'Distribution of first column from $X$')
    ax[1].hist(X_flow[:,0].numpy(), bins=50)
    ax[1].set_title(r'Distribution of first column from $X_{pf}$')
    fig.tight_layout()
    plt.savefig('results/plots/normalizing_flow_test.pdf')
    plt.savefig('results/plots/normalizing_flow_test.jpg')
    plt.clf()