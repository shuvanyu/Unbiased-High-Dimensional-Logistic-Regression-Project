import torch
import numpy as np
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
import statsmodels.api as sm
import scipy
from sklearn.metrics import f1_score
import time
np.random.seed(1)
torch.manual_seed(1)

def probs_to_labels(y_probs):
    labels = []
    for probs in y_probs:
        labels.append(np.argmax(probs))

    return np.array(labels)

def test_baseline(X_train, y_train, X_test, y_test, ci=95):
    try:
        start = time.time()
        lr = sm.Logit(y_train, X_train).fit(maxiter=60)
        end = time.time()
    except np.linalg.LinAlgError:
        print('Baseline test failed: Singular matrix')
        return None, None, None, None
    # Get p values for coefficents
    p_vals = lr.pvalues[X_train.shape[1]//4:]
    
    # Calculate the prediction intervals
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

    # Calculate the F1 score on the test set
    y_pred_proba = lr.predict(X_test)
    y_pred = list(map(round, y_pred_proba))
    score = f1_score(y_test, y_pred)

    # Get time taken to fit
    performance = end - start

    return p_vals, pred_ints, score, performance


def test_sloe(X_train, y_train, X_test, y_test, ci=95):
    ur = UnbiasedLogisticRegression(ci=ci)
    try:
        start = time.time()
        model = ur.fit(X_train, y_train)
        end = time.time()
    except ValueError:
        print('SLOE Test failed.')
        return None, None, None, None, None
    except np.linalg.LinAlgError:
        print('SLOE Test failed: Singular matrix')
        return None, None, None, None, None
    # Get p-values for coefficients
    p_vals = model.p_values()[X_train.shape[1]//4:]

    # Get logit inflation alpha
    alpha = model.alpha

    # Get prediction CI for the sample
    pred_ints = model.prediction_intervals(X_test)
    pred_ints = pred_ints[:,[0,2]]

    # Get F1 score
    y_pred_proba = ur.predict_proba(X_test)
    y_pred = probs_to_labels(y_pred_proba)
    score = f1_score(y_test, y_pred)

    # Get time taken to fit
    performance = end - start

    return p_vals, alpha, pred_ints, score, performance
