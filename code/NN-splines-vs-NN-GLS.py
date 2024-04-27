#### This file produces simulation results for comparing NN and NN-GLS in section S4.9

import utils

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
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

p = 5; funXY = utils.f5; Netp = utils.Netp_sig

sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
k = 50
q = 1
lr = 0.1
if method == '1':
    lr = 1 if p == 1 else 0.1
elif method == '2':
    lr = 0.5 if p == 1 else 0.05
#lr = 0.1
#### method 1: 1 for p=1, 0.1 for p=5; method2: 0.5 for p=1, 0.05 for p=5
ordered = False

N = 1000
n_small = int(N / 100)
np.random.seed(2021)
X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
for i in range(99):
    temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
    X_MISE = np.concatenate((X_MISE, temp))

Y_MISE = funXY(X_MISE).reshape(-1)
X_MISE_int = np.concatenate((X_MISE, np.repeat(1, N).reshape(N,1)), axis = 1)
X_MISE = torch.from_numpy(X_MISE).float()
X_MISE_int = torch.from_numpy(X_MISE_int).float()

Y_MISE_np = Y_MISE
Y_MISE = torch.from_numpy(Y_MISE).float()
Y_MISE = torch.reshape(Y_MISE, (N, 1))

name = 'NN-vs-NN-GLS'

RMSE_NN = np.empty(0)
RMSE_NNlatlong = np.empty(0)
RMSE_NNDK = np.empty(0)
RMSE_NNGLS_krig = np.empty(0)
n_vec = np.empty(0)

df_CI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
df_CI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

for rand in range(1, 10+1):
    for n in [500, 1000, 2000, 5000, 10000]:
        n_vec = np.append(n_vec, n)
        b = 10
        n_train = int(n / 2)
        nn = 20
        n_batch = 20
        batch_size = int(n / n_batch)
        ADDRFGLS = False
        # if n_train <= 1000: ADDRFGLS = True
        Sparse = False
        if n > 10000: Sparse = True

        torch.manual_seed(2023+rand)
        np.random.seed(2023+rand)
        X, Y, I_B, F_diag, rank, coord, cov, corerr = utils.Simulate(n, p, funXY, nn, theta, method=method, a=0,
                                                                          b=b, sparse = Sparse, meanshift= False)

        X_int = np.concatenate((X, np.repeat(1, n).reshape(n, 1)), axis=1)

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
        torch.manual_seed(2023)
        model_NN = Netp(p, k)
        optimizer = torch.optim.Adam(model_NN.parameters(), lr=lr)
        patience_half = 10
        patience = 20

        _, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                              patience = patience, patience_half = patience_half)
        Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################
        torch.manual_seed(2023)
        model_NNlatlong = Netp(p + 2, k, q)
        optimizer = torch.optim.Adam(model_NNlatlong.parameters(), lr=lr)
        patience_half = 10
        patience = 20
        data_latlong = copy.copy(data)

        data_latlong.x = torch.concatenate((data.x, torch.from_numpy(data.coord)), axis=1).float()

        _, _, model_NNlatlong = utils.train_gen_new(model_NNlatlong, optimizer, data_latlong, epoch_num=1000,
                                                       patience=patience, patience_half=patience_half)
        Pred_NNlatlong = model_NNlatlong(data_latlong.x[data_latlong.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_NNlatlong = np.append(RMSE_NNlatlong, RMSE(Pred_NNlatlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################
        num_basis = [2 ** 2, 4 ** 2, 6 ** 2]
        knots_1d = [np.linspace(0, 1, int(np.sqrt(i))) for i in num_basis]
        ##Wendland kernel
        K = 0
        phi_temp = np.zeros((n, sum(num_basis)))
        for res in range(len(num_basis)):
            theta_temp = 1 / np.sqrt(num_basis[res]) * 2.5
            knots_s1, knots_s2 = np.meshgrid(knots_1d[res], knots_1d[res])
            knots = np.column_stack((knots_s1.flatten(), knots_s2.flatten()))
            for i in range(num_basis[res]):
                d = np.linalg.norm(data.coord / b - knots[i, :], axis=1) / theta_temp
                for j in range(len(d)):
                    if d[j] >= 0 and d[j] <= 1:
                        phi_temp[j, i + K] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3
                    else:
                        phi_temp[j, i + K] = 0
            K = K + num_basis[res]

        torch.manual_seed(2023)
        model_NNDK = Netp(p + K, 100, q)
        optimizer = torch.optim.Adam(model_NNDK.parameters(), lr=lr)
        patience_half = 10
        patience = 20
        data_DK = copy.copy(data)
        data_DK.x = torch.concatenate((data.x, torch.from_numpy(phi_temp)), axis=1).float()
        _, _, model_NNDK = utils.train_gen_new(model_NNDK, optimizer, data_DK, epoch_num=1000,
                                                  patience=patience, patience_half=patience_half)
        Pred_NNDK = model_NNDK(data_DK.x[data_DK.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_NNDK = np.append(RMSE_NNDK, RMSE(Pred_NNDK, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
        Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = data.y - torch.from_numpy(Y_hat)
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        theta_hat0 = theta_hat.copy()

        ###################################################################################
        torch.manual_seed(2023)
        model_NNGLS = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=lr)
        patience_half = 10
        patience = 20

        _, _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse = Sparse,
                                                Update=True, patience=patience, patience_half=patience_half)
        Y_hat_NNGLS = model_NNGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual_NNGLS = data.y - torch.from_numpy(Y_hat_NNGLS)
        residual_NNGLS_np = residual_NNGLS.detach().numpy()
        beta_hat_NNGLS, theta_hat_NNGLS = utils.BRISC_estimation(residual_NNGLS_np[~data.test_mask,], X[~data.test_mask,],
                                                                    coord[~data.test_mask,])

        Pred_NNGLS = utils.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat_NNGLS)
        RMSE_NNGLS_krig = np.append(RMSE_NNGLS_krig,
                                    RMSE(Pred_NNGLS[0], Y_test)/RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
        df_RMSE = pd.DataFrame(
            {'NN_latlong': RMSE_NNlatlong, 'NNDK': RMSE_NNDK,
             'NN': RMSE_NN, 'NNGLS_krig': RMSE_NNGLS_krig,
             'n': n_vec})
        df_RMSE.to_csv(".//simulation//NN//" + name + '_RMSE.csv')

