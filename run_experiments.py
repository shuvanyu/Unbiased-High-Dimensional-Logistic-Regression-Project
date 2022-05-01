import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.simulation import simulate_gaussian, normalizing_flow
from helpers.plotting import plot_p_vals, plot_conf_ints, test_normalizing_flow
from helpers.models_test import test_baseline, test_sloe
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)
torch.manual_seed(1)

ratio = 0.2    # p/n
# p  = Number of features
# n = Number of samples
n = 3000
p = int(ratio*n)
gamma = np.sqrt(5)

ci = 90

# TEST 1: Simulated Gaussian data
print("TEST 1: Simulated Gaussian data")

# Simulate Gaussian data
X_train, X_test, y_train, y_test = simulate_gaussian(ratio, n, p, gamma)
#y_train = torch.round(torch.rand(y_train.shape[0]))
#y_test = torch.round(torch.rand(y_test.shape[0]))

# Test with baseline model
p_vals_baseline, pred_ints_baseline = test_baseline(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# Test with SLOE Model
p_vals_sloe, pred_ints_sloe = test_sloe(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# Generate plots
plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Gaussian', 'gaussian')
plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test.numpy(), y_test.numpy(), 'Simulated Gaussian', 'gaussian', ci=ci)

# TEST 2: Simulated Non-Gaussian data
print('TEST 2: Simulated Non-Gaussian data')

# Simulate Non-Gaussian data with normalizing flow
X_train_flow, X_test_flow = normalizing_flow(X_train, X_test)

# Test with baseline model
p_vals_baseline, pred_ints_baseline = test_baseline(X_train_flow.numpy(), y_train.numpy(), X_test_flow.numpy(), ci=ci)

# Test with SLOE Model
p_vals_sloe, pred_ints_sloe = test_sloe(X_train_flow.numpy(), y_train.numpy(), X_test_flow.numpy(), ci=ci)

# Generate plots
plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Non-Gaussian', 'non_gaussian')
plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test_flow.numpy(), y_test.numpy(), 'Simulated Non-Gaussian', 'non_gaussian', ci=ci)