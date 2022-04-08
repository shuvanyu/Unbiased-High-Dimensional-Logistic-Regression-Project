import numpy as np
from sloe_logistic.unbiased_logistic_regression import UnbiasedLogisticRegression
from sklearn.linear_model import LogisticRegression

# 4000 x 10 normally disributed data
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from sklearn.model_selection import train_test_split

# Normally distributed N(0,1) features, sigmoid(nonlinear function) for y's
#X = MN(torch.zeros(4000, 5), torch.eye(5)).sample()
#y = torch.round(torch.sigmoid(X[:,0] + X[:,1] + X[:,2] + X[:,3] + X[:,4]))

# Baseline model: sklearn's Logistic Regression
#X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), train_size=0.8, shuffle=True)

#lr = LogisticRegression()
#lr.fit(X_train, y_train)
#lr.predict_proba(X_test)
#score = lr.score(X_test, y_test)

#print("Baseline Score (sklearn): {}\n".format(score))

# SLOE model testing
np.random.seed(1)
n = 1000
d = 5
features = np.random.randn(n, d)
beta = np.sqrt(5 * 2.0 / d) * np.ones(d)
beta[(d // 2):] = 0

outcome = (np.random.rand(n) <= 1.0 /
            (1.0 + np.exp(-features.dot(beta)))).astype(float)

ur = UnbiasedLogisticRegression()
ur.fit(features, outcome)
print(ur.coef_)