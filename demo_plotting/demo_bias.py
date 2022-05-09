import torch
from torch.distributions.normal import Normal
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

p = 800 # Number of features
n = 4000
ratio = p/n
gamma = np.sqrt(5)
mu = torch.zeros(p)
X = torch.randn(p,n)*0.5
cov = torch.cov(X)
x_dist = Normal(loc = 0, scale = 1.0/n)
x_samples = x_dist.sample([n,p])*63.29
beta = torch.zeros(p)
beta[:p//8] = 10
beta[p//8:p//4] = -10

print(torch.matmul(x_samples,beta).var())

proba = torch.sigmoid(torch.matmul(x_samples,beta))
y = torch.bernoulli(proba)
lr = sm.Logit(y.numpy(), x_samples.numpy()).fit()
coeff = lr.params
plt.scatter(np.arange(0,p),coeff, s=3, color='red')
plt.plot((0,p//8),(10,10),'-',color='black', linewidth=3)
plt.plot((p//8,p//4),(-10,-10),'-',color='black', linewidth=3)
plt.plot((p//4,p),(0,0),'-',color='black', linewidth=3)
plt.title('The parameters of logistic regression for $p=800, n=4000$')
plt.xlabel('Index')
plt.ylabel('Coefficients (true and MLE fitted)')
plt.legend(['Coefficients using MLE estimation','True coefficients'])

plt.grid()
plt.savefig('demo_plotting/plots/bias.pdf')
plt.savefig('demo_plotting/plots/bias.jpg')
plt.show()




