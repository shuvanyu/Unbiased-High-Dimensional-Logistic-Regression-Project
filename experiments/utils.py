import numpy as np

def gen_y(X):
    n = X.shape[0]
    p = X.shape[1]
    beta = np.sqrt(5 * 2.0 / p) * np.ones(p)
    beta[(p // 2):] = 0
    y = (np.random.rand(n) <= 1.0 / (1.0 + np.exp(-X.dot(beta)))).astype(float)

    return y