#### This file produces simulation results for parameter estimation in section S3.2

import torch
import sys
sys.path.append("/Users/zhanwentao/Documents/Abhi/NN")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import utils

def order(X, Y, coord):
    s_sum = coord[:, 0] + coord[:, 1]
    order = s_sum.argsort()
    X_new = X[order, :]
    Y_new = Y[order]
    coord_new = coord[order, :]
    return X_new, Y_new, coord_new

sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
p = 5
funXY = utils.f5
b = 10

n = 1000

torch.manual_seed(2023)
X, Y, I_B, F_diag, rank, coord, cov, corerr = utils.Simulate(n, p, funXY, 10, theta, method=method, a=0, b=b)
X, Y, coord = order(X, Y, coord)

det = np.empty(0)
eigen_max = np.empty(0)
eigen_min = np.empty(0)
for nn in np.arange(1, 250):
    print(nn)
    I_B, F_diag, rank, cov = utils.bf_from_theta(theta, coord, nn, sparse=False)
    I_B = I_B.detach().numpy()
    F_diag = F_diag.detach().numpy()

    cov_sqrt = np.linalg.cholesky(cov)
    E_sqrt = np.matmul((np.sqrt(np.reciprocal(F_diag)) * I_B.T).T, cov_sqrt)
    E = np.matmul(E_sqrt.T, E_sqrt)

    det = np.append(det, np.linalg.det(E))
    eigen_max = np.append(eigen_max, np.linalg.norm(E, 2))
    eigen_min = np.append(eigen_min, np.linalg.norm(E, -2))

df = pd.DataFrame(
    {'Det': det, 'Eigen_max': eigen_max, 'Eigen_min': eigen_min})
name = "p%i"%p + 'phi%i'%phi + "sig%i"%(theta[0]) + "tau%i"%(int(100*tau)) + "n%i"%int(n)
df.to_csv(".//simulation//E//" + name + "_py.csv")

