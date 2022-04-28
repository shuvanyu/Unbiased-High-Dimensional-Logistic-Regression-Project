import torch
import numpy as np
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
from sklearn.linear_model import LogisticRegression

from torch.distributions import Normal
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from utils import gen_y

np.random.seed(1)
torch.manual_seed(1)

# RESULTS
# Ratio between (0.05, 0.24) exclusive

# Set dimensions of simulation
# r - ratio
r = 0.2
# n - number of rows
n = 200
# p - number of columns
p = int(np.ceil(r * n))

# Models
lr = LogisticRegression()
ur = UnbiasedLogisticRegression()

def probs_to_labels(y_probs):
    labels = []
    for probs in y_probs:
        labels.append(np.argmax(probs))

    return np.array(labels)

# TEST 1: Gaussian data
print("TEST 1: Gaussian Data")
# Generate multivariate normal data
X = Normal(torch.zeros(p), torch.ones(p)).sample((n,)).numpy()
y = gen_y(X)

# Predict with baseline sklearn model
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=0)
lr.fit(X_train, y_train)
y_pred_proba = lr.predict_proba(X_test)
y_pred = probs_to_labels(y_pred_proba)
baseline_score = f1_score(y_test, y_pred)
print("Baseline Model F1 Score: {}".format(baseline_score))


# Predict with SLOE Model
ur.fit(X_train, y_train)
y_pred_proba = ur.predict_proba(X_test)
y_pred = probs_to_labels(y_pred_proba)
sloe_score = f1_score(y_test, y_pred)
print("SLOE Model F1 Score: {}".format(sloe_score))