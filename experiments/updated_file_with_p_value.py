import torch
import numpy as np
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
from torch.distributions.multivariate_normal import MultivariateNormal
from sklearn.model_selection import train_test_split
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
np.random.seed(1)
torch.manual_seed(1)

## Data Generating Process #################################################
# The data generating process is parameterized by gamma, ratio, n
# Step-1 - Draw n examples of covariate vectors X from a given distribution
# Step-2 -  calculate the mu
# Step-3 - Calculate beta 
# Step 4 - Finally draw Y ~ Bernoulli(proba)
############################################################################

ratio = 0.2    # p/n
# p  = Number of features
# m = Number of samples
n = 4000
p = int(ratio*n)
gamma = np.sqrt(5)

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

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)


def test_baseline(X_train, X_test, y_train, y_test):
    lr = sm.Logit(y_train.numpy(), X_train.numpy()).fit()
    p_vals = lr.pvalues

    return p_vals


def test_sloe(X_train, X_test, y_train, y_test):
    ur = UnbiasedLogisticRegression()
    model = ur.fit(X_train.numpy(), y_train.numpy())
    p_vals = model.p_values()

    return p_vals


# Test with statsmodel model
p_vals_baseline = test_baseline(X_train, X_test, y_train, y_test)
plt.hist(p_vals_baseline, bins=20)
plt.show()


# Test with SLOE Model
p_vals_sloe = test_sloe(X_train, X_test, y_train, y_test)
plt.hist(p_vals_sloe, bins=20)
plt.show()