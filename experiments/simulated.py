import torch
import numpy as np
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
import statsmodels.api as sm

from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils import gen_y

import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

def probs_to_labels(y_probs):
    labels = []
    for probs in y_probs:
        labels.append(np.argmax(probs))

    return np.array(labels)

# RESULTS
# Ratio between (0.05, 0.24) exclusive

# Set dimensions of simulation
# r - ratio
r = 0.2
# n - number of rows
n = 100
# p - number of columns
p = int(np.ceil(r * n))

def test_baseline(X_train, X_test, y_train, y_test):
    lr = sm.Logit(y_train, X_train).fit()
    p_vals = lr.pvalues

    y_pred_proba = lr.predict(X_test)
    y_pred = list(map(round, y_pred_proba))
    baseline_score = f1_score(y_test, y_pred)
    return baseline_score, p_vals

def test_sloe(X_train, X_test, y_train, y_test):
    ur = UnbiasedLogisticRegression()
    model = ur.fit(X_train, y_train)
    p_vals = model.p_values()

    y_pred_proba = ur.predict_proba(X_test)
    y_pred = probs_to_labels(y_pred_proba)
    sloe_score = f1_score(y_test, y_pred)
    return sloe_score, p_vals

# TEST 1: Gaussian data
print("TEST 1: Gaussian Data")
# Generate multivariate normal data
X = Normal(torch.zeros(p), torch.ones(p)).sample((n,)).numpy()
#y = gen_y(X)
y = np.rint(np.random.rand(n))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)

# Test with statsmodel model
# score, p_vals = test_baseline(X_train, X_test, y_train, y_test)
# print("Baseline F-1 score: {}".format(score))
# plt.hist(p_vals, bins=20)
# plt.show()

# # Test with SLOE Model
# score, p_vals = test_sloe(X_train, X_test, y_train, y_test)
# print("SLOE F-1 score: {}".format(score))
# plt.hist(p_vals, bins=20)
# plt.show()

# TEST 2: Normalizing flow: Gaussian -> Non-Gaussian
def normalizing_flow(x):
    d = x.shape[0]
    u = torch.ones(d)*10
    b = 1
    w = torch.rand(d)*100
    # h is sigmoid function
    f = torch.from_numpy(x) + u*torch.sigmoid(w.dot(torch.from_numpy(x)) + b)

    return f.numpy()

X_train_flow = np.apply_along_axis(lambda x: normalizing_flow(x), 1, X_train)
X_test_flow = np.apply_along_axis(lambda x: normalizing_flow(x), 1, X_test)

# Test with statsmodel model
score, p_vals = test_baseline(X_train, X_test, y_train, y_test)
print("Baseline F-1 score: {}".format(score))
plt.hist(p_vals, bins=20)
plt.show()

# Test with SLOE Model
score, p_vals = test_sloe(X_train, X_test, y_train, y_test)
print("SLOE F-1 score: {}".format(score))
plt.hist(p_vals, bins=20)
plt.show()