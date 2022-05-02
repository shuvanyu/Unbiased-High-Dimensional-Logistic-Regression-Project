import torch
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
import statsmodels.api as sm
import scipy

def test_baseline(X_train, y_train, X_test, ci=95):
    lr = sm.Logit(y_train, X_train).fit()
    p_vals = lr.pvalues
    
    logits = X_test.dot(lr.params.T).reshape(-1)
    variances = (X_test.dot(lr.cov_params()) *
            X_test).sum(axis=-1).reshape(-1)
    z = scipy.stats.norm.ppf(0.5 + (ci / 100.0) / 2.0)

    lower_ci = logits - z * np.sqrt(variances)
    upper_ci = logits + z * np.sqrt(variances)

    results = np.zeros((X_test.shape[0], 2))
    results[:, 0] = lower_ci
    results[:, 1] = upper_ci

    pred_ints = 1.0 / (1.0 + np.exp(-results))

    return p_vals, pred_ints


def test_sloe(X_train, y_train, X_test, ci=95):
    ur = UnbiasedLogisticRegression(ci=ci)
    model = ur.fit(X_train, y_train)
    # Get p-values for coefficients
    p_vals = model.p_values()

    # Get prediction CI for the sample
    pred_ints = model.prediction_intervals(X_test)
    pred_ints = pred_ints[:,[0,2]]

    return p_vals, pred_ints
