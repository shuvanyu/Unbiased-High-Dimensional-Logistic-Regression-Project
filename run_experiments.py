import torch
import numpy as np
import pandas as pd
from helpers.simulation import run_test
from helpers.heart_data import retrieve_heart_data
import warnings
warnings.filterwarnings('ignore')
np.random.seed(1)
torch.manual_seed(1)

# SIMULATED DATA TESTS
print('SIMULATED DATA TESTS')
ratios = [0.1, 0.2]
n_vals = [500, 1000, 5000]
gamma = np.sqrt(5)
ci = 90

# To store F1 scores
scores = []

for ratio in ratios:
    for n in n_vals:
        print('Trials for: n={}, ratio={}'.format(n, ratio))
        p = int(ratio*n)
        # TEST 1: Simulated Gaussian data
        print("TEST 1: Simulated Gaussian data")

        test_res = run_test(ratio, n, p, ci, 'Simulated Gaussian', 'gaussian', gamma=gamma)
        scores.append(test_res)

        # TEST 2: Simulated Non-Gaussian data
        print('TEST 2: Simulated Non-Gaussian data')

        test_res = run_test(ratio, n, p, ci, 'Simulated Non-Gaussian', 'non_gaussian', gamma=gamma)
        scores.append(test_res)

        # TEST 3: Latent Variables
        print('TEST 3: Latent Variables')

        test_rest = run_test(ratio, n, p, ci, 'Simulated Latent', 'latent', gamma=gamma)
        scores.append(test_res)

        # TEST 4: Heavy tailed Gaussian (Cauchy)
        print('TEST 4: Heavy Tailed Gaussian (Cauchy)')

        test_res = run_test(ratio, n, p, ci, 'Simulated Heavy-tailed distribution','heavy', gamma=gamma)
        scores.append(test_res)

# Write scores to file
scores_df = pd.DataFrame(scores)
print('Final Scores')
print(scores_df)
scores_df.to_csv('results/simulated_scores.csv', index=False)

# HEART DISEASE DATA
print('HEART DISEASE DATA TESTS')
scores = []
sampling_ratios = [0.33, 0.4, 0.58]

for sampling_ratio in sampling_ratios:
    heart_df = retrieve_heart_data(sampling_ratio)
    n = heart_df.shape[0]
    p = heart_df.shape[1]-6
    ratio = np.around(p / float(n), decimals=2)
    print('Trials for: n={}, ratio={}'.format(n, ratio))

    y = torch.tensor(heart_df.pop('prediction').astype(float).to_numpy())
    X = torch.tensor(heart_df.astype(float).to_numpy())[:,:15]
    test_res = run_test(ratio, n, p, ci, 'Heart Disease Data', 'heart', X=X, y=y)
    scores.append(test_res)

# Write scores to file
scores_df = pd.DataFrame(scores)
print('Final Scores')
print(scores_df)
scores_df.to_csv('results/heart_disease_scores.csv', index=False)