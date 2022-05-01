import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.simulation import simulate_gaussian, normalizing_flow, simulate_latent_gaussian
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

# # TEST 1: Simulated Gaussian data
# print("TEST 1: Simulated Gaussian data")

# # Simulate Gaussian data
# X_train, X_test, y_train, y_test = simulate_gaussian(ratio, n, p, gamma)

# # Test with baseline model
# p_vals_baseline, pred_ints_baseline = test_baseline(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# # Test with SLOE Model
# p_vals_sloe, pred_ints_sloe = test_sloe(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# # Generate plots
# plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Gaussian', 'gaussian')
# plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test.numpy(), y_test.numpy(), 'Simulated Gaussian', 'gaussian', ci=ci)

# # TEST 2: Simulated Non-Gaussian data
# print('TEST 2: Simulated Non-Gaussian data')

# # Simulate Non-Gaussian data with normalizing flow
# X_train_flow, X_test_flow = normalizing_flow(X_train, X_test)

# # Test with baseline model
# p_vals_baseline, pred_ints_baseline = test_baseline(X_train_flow.numpy(), y_train.numpy(), X_test_flow.numpy(), ci=ci)

# # Test with SLOE Model
# p_vals_sloe, pred_ints_sloe = test_sloe(X_train_flow.numpy(), y_train.numpy(), X_test_flow.numpy(), ci=ci)

# # Generate plots
# plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Non-Gaussian', 'non_gaussian')
# plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test_flow.numpy(), y_test.numpy(), 'Simulated Non-Gaussian', 'non_gaussian', ci=ci)

# TEST 3: Latent Variables
print('TEST 3: Latent Variables')
latent_ratio = 0.8

# We will calculate the covariates as linear functions of 3 Gaussian latent variables
X_train_latent, X_test_latent, y_train_latent, y_test_latent = simulate_latent_gaussian(ratio, n, p, gamma, latent_ratio)

# Test with baseline model
p_vals_baseline, pred_ints_baseline = test_baseline(X_train_latent.numpy(), y_train_latent.numpy(), X_test_latent.numpy(), ci=ci)

# Test with SLOE Model
p_vals_sloe, pred_ints_sloe = test_sloe(X_train_latent.numpy(), y_train_latent.numpy(), X_test_latent.numpy(), ci=ci)

# Generate plots
plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Latent', 'latent')
plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test_latent.numpy(), y_test_latent.numpy(), 'Simulated Latent', 'latent', ci=ci)