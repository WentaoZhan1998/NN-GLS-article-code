#### This file produces an example of a fixed Gaussian process surface

import utils
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

sigma = 1
phi = 1
tau = 0.01
theta = [sigma, phi / np.sqrt(2), tau]

np.random.seed(2022)
n = 100
b = 10
coord0 = np.random.uniform(low=0, high=b, size=(n, 2))
theta0 = theta.copy()

I_B, F_diag, rank, cov = utils.bf_from_theta(theta, coord0, 20, method='0', nu=1.5, sparse=False)

corerr0 = utils.rmvn(1, np.zeros(n), cov, I_B, F_diag, sparse = False)

def corerr_gen(pos):
    df0 = pd.DataFrame(coord0, columns=['x', 'y'])
    df1 = pd.DataFrame(pos, columns=['x', 'y'])
    mask = np.concatenate([np.repeat(True, df0.shape[0]), np.repeat(False, df1.shape[0])])
    df = pd.concat([df0, df1])
    dist = distance_matrix(df.values, df.values)
    cov = utils.make_cov(theta0, dist)
    theta0[2] = 0
    C = utils.make_cov(theta0, dist)
    del df
    del dist
    corerr = np.matmul(C[np.invert(mask), :][:, mask], np.linalg.solve(cov[mask, :][:, mask], corerr0))
    del cov
    del C
    return (corerr)

plt.clf()
x, y = np.mgrid[0:10:.1, 0:10:.1]
pos = np.dstack((x, y))
fig, ax = plt.subplots()
im = ax.pcolormesh(x, y, corerr_gen(pos.reshape(-1,2)).reshape(100, 100), cmap = 'plasma')
fig.colorbar(im)

ax.axis('tight')
plt.show()
plt.savefig(".//temp_figure//shape.png")