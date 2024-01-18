#### This file produces simulation results for comparing GAM-GLS and NN-GLS in section S3.1, Figure S5.

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
import lhsmdu

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

fun = 'sin'

if fun == 'sin':
    p = 1; funXY = utils.fx; Netp = utils.Netp_sig
    update_init = 50; update_step = 50
elif fun == 'friedman':
    p = 5; funXY = utils.f5; Netp = utils.Netp_sig
    update_init = 20; update_step = 20
k = 50
q = 1
lr = 0.01
b = 10
method = '0'

n = 1000
n_train = int(n/2)
nn = 20
batch_size = int(n/20)
Sparse = False
if n > 10000: Sparse = True

N = 1000
n_small = int(N / 100)
np.random.seed(2021)
X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
for i in range(99):
    temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
    X_MISE = np.concatenate((X_MISE, temp))

Y_MISE = funXY(X_MISE)
X_MISE = torch.from_numpy(X_MISE).float()

Y_MISE_np = Y_MISE
Y_MISE = torch.from_numpy(Y_MISE).float()
Y_MISE = torch.reshape(Y_MISE, (N, 1))

name = 'par'
MISE_NN = np.empty(0)
MISE_NNGLS = np.empty(0)
MISE_NNGLS2 = np.empty(0)
MISE_NNGLS3 = np.empty(0)
RMSE_NN = np.empty(0)
RMSE_NN_krig = np.empty(0)
RMSE_NNGLS = np.empty(0)
RMSE_NNGLS2 = np.empty(0)
RMSE_NNGLS3 = np.empty(0)

df_theta = pd.DataFrame(columns=['sigma', 'phi', 'tau', 'method', 'ind', 'sigma0', 'phi0', 'tau0'])

theta_mat = np.array([[1,1,1,1,1,1,5,5,5,5,5,5],
                      [1,1,3,3,6,6,1,1,3,3,6,6],
                      [0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1,0.01,0.1]])

theta_mat[1,:] = theta_mat[1,:]/np.sqrt(2)

for rand in range(100):
    for t in range(12):
        theta0 = theta_mat[:,t]
        theta = theta0.copy()
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
        Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
        Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = torch.from_numpy(Y_hat) - data.y
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        theta_hat0 = theta_hat

        Pred_NN = utils.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                     coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
        RMSE_NN_krig = np.append(RMSE_NN_krig, RMSE(Pred_NN[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ###################################################################################
        torch.manual_seed(2023)
        model_NNGLS = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
        patience_half = 10
        patience = 20

        theta_hat, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse = Sparse,
                                                Update=False, patience=patience, patience_half=patience_half)
        Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
        Pred_NNGLS = utils.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat)
        RMSE_NNGLS= np.append(RMSE_NNGLS, RMSE(Pred_NNGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        df_theta = df_theta.append({'sigma': theta_hat0[0], 'phi': theta_hat0[1], 'tau': theta_hat0[2],
                                    'method': 'NN_BRISC', 'ind': rand,
                                    'sigma0': theta[0], 'phi0': theta[1], 'tau0': theta[2]}, ignore_index=True)
        ###################################################################################
        torch.manual_seed(2023)
        model_NNGLS2 = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NNGLS2.parameters(), lr=0.1)
        patience_half = 10
        patience = 20

        theta_hat2, _, model_NNGLS2 = utils.train_decor_new(model_NNGLS2, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                                      Update=True, patience=patience, patience_half=patience_half,
                                                      Update_bound=100, Update_method='optimization',
                                                      Update_init=20, Update_step=20)
        Est_MISE_NNGLS2 = model_NNGLS2(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NNGLS2 = np.append(MISE_NNGLS2, RMSE(Est_MISE_NNGLS2, Y_MISE_np.reshape(-1)))
        Pred_NNGLS2 = utils.krig_pred(model_NNGLS2, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat2)
        RMSE_NNGLS2 = np.append(RMSE_NNGLS2, RMSE(Pred_NNGLS2[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        df_theta = df_theta.append({'sigma': theta_hat2[0], 'phi': theta_hat2[1], 'tau': theta_hat2[2],
                                    'method': 'NNGLS_update1', 'ind': rand,
                                    'sigma0': theta[0], 'phi0': theta[1], 'tau0': theta[2]}, ignore_index=True)
        ###################################################################################

    df_MISE = pd.DataFrame({'NN': MISE_NN,
                            'NNGLS': MISE_NNGLS, 'NNGLS_update1': MISE_NNGLS2})
    df_RMSE = pd.DataFrame({'NN': RMSE_NN, 'NN_krig': RMSE_NN_krig,
                            'NNGLS': RMSE_NNGLS, 'NNGLS_update1': RMSE_NNGLS2})
    df_theta.to_csv(".//simulation//par//dim" + str(p) + name + '_theta.csv')
    df_MISE.to_csv(".//simulation//par//dim" + str(p) + name + '_MISE.csv')
    df_RMSE.to_csv(".//simulation//par//dim" + str(p) + name + '_RMSE.csv')
