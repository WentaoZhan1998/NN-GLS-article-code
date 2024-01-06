#### This file produces simulation results for comparing GAM-GLS and NN-GLS in section S4.4

import os
#os.environ['R_HOME'] = {the R lib path where necessary packages are installed}
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import utils
import utils_pygam

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor
import lhsmdu
import copy

def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

def order(X, Y, coord):
    s_sum = coord[:, 0] + coord[:, 1]
    order = s_sum.argsort()
    X_new = X[order, :]
    Y_new = Y[order]
    coord_new = coord[order, :]
    return X_new, Y_new, coord_new

def int_score(x, u, l, coverage=0.95):
    alpha = 1 - coverage
    score = u - l + 2 * ((l - x) * (l > x) + (x - u) * (x > u)) / alpha
    return (np.mean(score))

def int_coverage(x, u, l):
    score = np.logical_and(x>=l, x<=u)
    return (np.mean(score))

p = 5
Netp = utils.Netp_sig

sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
k = 50
q = 1
lr = 0.01
b = 10

n = 1000
n_train = 500
nn = 20
batch_size = int(n/20)
ADDRFGLS = True
Sparse = False

N = 1000
n_small = int(N / 100)
np.random.seed(2021)
X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
for i in range(99):
    temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
    X_MISE = np.concatenate((X_MISE, temp))

X_MISE = torch.from_numpy(X_MISE).float()

name = "Friedman-comparison"

rho_vec = np.empty(0)
MISE_GAM = np.empty(0)
MISE_GAMGLS = np.empty(0)
MISE_NN = np.empty(0)
MISE_NNGLS = np.empty(0)

for rho in list(np.array(range(21))/20):
    def f5(X): return rho * (10 * np.sin(np.pi * X[:, 0] * X[:, 1]))/3 + \
                      (1-rho)*(20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:,4]) / 3

    funXY = f5
    Y_MISE = funXY(X_MISE.detach().numpy())

    Y_MISE_np = Y_MISE
    Y_MISE = torch.from_numpy(Y_MISE).float()
    Y_MISE = torch.reshape(Y_MISE, (N, 1))
    for rand in range(10):
        torch.manual_seed(2023+rand)
        np.random.seed(2023+rand)
        X, Y, I_B, F_diag, rank, coord, cov, corerr = utils.Simulate(n, p, funXY, nn, theta, method=method, a=0,
                                                                     b=b, sparse = Sparse)

        neigh = NearestNeighbors(n_neighbors=nn)
        neigh.fit(coord)

        A = neigh.kneighbors_graph(coord)
        A.toarray()
        edge_index = torch.from_numpy(np.stack(A.nonzero()))

        torch.manual_seed(2023+rand)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord = coord)
        transform = T.RandomNodeSplit(num_train_per_class=int(0.3*n_train), num_val=int(0.2*n_train), num_test=int(n-n_train))
        data = transform(data)
        data.n = data.x.shape[0]

        Y_test = Y[data.test_mask]

        torch.manual_seed(2023+rand)
        data = utils.batch_gen(data, batch_size)
        ####################################################################################################################
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        gam = utils_pygam.my_LinearGAM()
        gam.fit(X[~data.test_mask,], Y[~data.test_mask])
        Xspline = gam._modelmat(X[~data.test_mask, :])
        gam.my_fit(X[~data.test_mask,], Xspline, Y[~data.test_mask])
        PI_GAM = gam.confidence_intervals(X[data.test_mask])
        def model_GAM(X, edge_index=0):
            if torch.is_tensor(X):
                X = X.detach().numpy()
            return (torch.from_numpy(gam.predict(X)).reshape(-1))


        Est_MISE_GAM = model_GAM(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_GAM = np.append(MISE_GAM, RMSE(Est_MISE_GAM, Y_MISE_np.reshape(-1)))

        Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = data.y - torch.from_numpy(Y_hat)
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        _, theta_hat_GAM = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        ####################################################################################################################
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        gam = utils_pygam.my_LinearGAM()
        gam.fit(X[~data.test_mask,], Y[~data.test_mask])
        Xspline = gam._modelmat(X[~data.test_mask, :])
        I_B_GAM, F_GAM, _, _ = utils.bf_from_theta(theta_hat_GAM, coord[~data.test_mask, :], nn, sparse=Sparse,
                                                   version='sparseB')
        F_GAM = F_GAM.detach().numpy()
        if Sparse:
            FI_B_GAM = I_B_GAM.Fmul(np.sqrt(np.reciprocal(F_GAM))).to_tensor()
            gam.my_fit(X[~data.test_mask,],
                       sparse.csr_matrix(np.array(utils.decor_sparse_np(Xspline.todense(), FI_B_GAM))),
                       np.array(utils.decor_sparse_np(Y[~data.test_mask], FI_B_GAM)))
        else:
            I_B_GAM = I_B_GAM.detach().numpy();
            FI_B_GAM = (I_B_GAM.T * np.sqrt(np.reciprocal(F_GAM))).T
            gam.my_fit(X[~data.test_mask,],
                       sparse.csr_matrix(np.array(utils.decor_dense_np(Xspline.todense(), FI_B_GAM))),
                       np.array(utils.decor_dense_np(Y[~data.test_mask], FI_B_GAM)))
        del I_B_GAM, F_GAM, FI_B_GAM
        PI_GAMGLS = gam.confidence_intervals(X[data.test_mask])

        def model_GAMGLS(X, edge_index=0):
            if torch.is_tensor(X):
                X = X.detach().numpy()
            return (torch.from_numpy(gam.predict(X)).reshape(-1))


        Est_MISE_GAMGLS = model_GAMGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_GAMGLS = np.append(MISE_GAMGLS, RMSE(Est_MISE_GAMGLS, Y_MISE_np.reshape(-1)))
        ######################################################################################################################
        torch.manual_seed(2023)
        model_NN = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
        patience_half = 10
        patience = 20

        _, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                              patience = patience, patience_half = patience_half)
        Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NN = np.append(MISE_NN, RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
        ####################################################################################################################
        Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = data.y - torch.from_numpy(Y_hat)
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        theta_hat0 = theta_hat
        ###################################################################################
        torch.manual_seed(2023)
        model_NNGLS = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
        patience_half = 10
        patience = 20

        _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse = Sparse,
                                                Update=True, patience=patience, patience_half=patience_half)
        Y_hat_NNGLS = model_NNGLS(data.x, data.edge_index).reshape(-1).detach().numpy()

        Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
        ####################################################################################################################
        rho_vec = np.append(rho_vec, rho)

    df_MISE = pd.DataFrame(
        {'GAM': MISE_GAM, 'GAMGLS': MISE_GAMGLS,
         'NN': MISE_NN, 'NNGLS': MISE_NNGLS, 'rho': rho_vec})
    df_MISE.to_csv(".//simulation//friedman//" + name + '_MISE.csv')