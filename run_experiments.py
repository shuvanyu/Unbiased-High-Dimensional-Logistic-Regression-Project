import torch
import numpy as np
import matplotlib.pyplot as plt
from helpers.simulation import simulate_gaussian
from helpers.plotting import plot_p_vals, plot_conf_ints
from helpers.models_test import test_baseline, test_sloe
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)
torch.manual_seed(1)

ratio = 0.2    # p/n
# p  = Number of features
# n = Number of samples
n = 1000
p = int(ratio*n)
gamma = np.sqrt(5)

ci = 90

X_train, X_test, y_train, y_test = simulate_gaussian(ratio, n, p, gamma)

# TEST 1: Gaussian data
print("TEST 1: Gaussian data")

# Test with baseline model
p_vals_baseline, pred_ints_baseline = test_baseline(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# Test with SLOE Model
p_vals_sloe, pred_ints_sloe = test_sloe(X_train.numpy(), y_train.numpy(), X_test.numpy(), ci=ci)

# Generate plots
plot_p_vals(p_vals_baseline, p_vals_sloe, 'Simulated Gaussian', 'gaussian')
plot_conf_ints(pred_ints_baseline, pred_ints_sloe, X_test.numpy(), y_test.numpy(), 'Simulated Gaussian', 'gaussian', ci=ci)