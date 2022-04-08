import torch
#from sloe_logistic import unbiased_logistic_regression_test
from sklearn.linear_model import LogisticRegression

# 4000 x 10 normally disributed data
from torch.distributions.multivariate_normal import MultivariateNormal as MN
from sklearn.model_selection import train_test_split

# Normally distributed N(0,1) features, sigmoid(nonlinear function) for y's
X = MN(torch.zeros(4000, 5), torch.eye(5)).sample()
y = torch.round(torch.sigmoid(X[:,0] ** 2 + X[:,1] ** 3 + X[:,2] ** 2 + X[:,3] ** 3 + X[:,4] ** 2))

# Baseline model: sklearn's Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), train_size=0.8, shuffle=True)

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr.predict_proba(X_test)
score = lr.score(X_test, y_test)

print("Baseline Score (sklearn): {}\n".format(score))

# SLOE model testing
