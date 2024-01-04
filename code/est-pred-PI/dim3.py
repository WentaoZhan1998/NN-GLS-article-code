import os
os.environ['R_HOME'] = '/users/wzhan/anaconda3/envs/torch_geom/lib/R'
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import utils_NN
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.logging import log
import sys
sys.path.append("/Users/zhanwentao/Documents/Abhi/NN")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance_matrix
from scipy import sparse
from pygam import LinearGAM, s, f
from sklearn.ensemble import RandomForestRegressor
import lhsmdu
import copy
import time

import mygam

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

class Netp_sig(torch.nn.Module):
    def __init__(self, p, k = 50, q = 1):
        super(Netp_sig, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)

    def forward(self, x, edge_index = 0):
        x = torch.sigmoid(self.l1(x))
        return self.l2(x)

class Netp_tanh(torch.nn.Module):
    def __init__(self, p, k = 50, q = 1):
        super(Netp_tanh, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)

    def forward(self, x, edge_index = 0):
        x = torch.tanh(self.l1(x))
        return self.l2(x)

class Netp2(torch.nn.Module):
    def __init__(self, p, k1 = 100, k2 = 20, q = 1):
        super(Netp2, self).__init__()
        self.l1 = torch.nn.Linear(p, k1)
        self.l2 = torch.nn.Linear(k1, k2)
        self.l3 = torch.nn.Linear(k2, 1)

    def forward(self, x, edge_index = 0):
        x = torch.sigmoid(self.l1(x))
        x = torch.sigmoid(self.l2(x))
        return self.l3(x)

x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
var1 = multivariate_normal(mean=[0.25, 0.25], cov=[[0.01, 0], [0, 0.01]])
var2 = multivariate_normal(mean=[0.6, 0.9], cov=[[0.01, 0], [0, 0.01]])
mean = np.mean(var1.pdf(pos) + var2.pdf(pos))
var = np.var(var1.pdf(pos) + var2.pdf(pos))
def corerr_gen(pos): # must be designed for unit square
    n = pos.shape[0]
    return ((var1.pdf(pos) + var2.pdf(pos) - mean) * np.sqrt(sigma) / np.sqrt(var) + np.sqrt(sigma*tau)*np.random.randn(n))

def partition (list_in, n):
    idx = torch.randperm(list_in.shape[0])
    list_in = list_in[idx]
    return [torch.sort(list_in[i::n])[0] for i in range(n)]

def decor(y, I_B_local, F_diag_local):
    y_decor = (np.array(np.matmul(I_B_local, y)).T*np.sqrt(np.reciprocal(F_diag_local))).T
    return(y_decor)

def batch_gen (data, k):
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        data[mask + '_batch'] = partition(torch.tensor(range(data.n))[data[mask]],
                                          int(torch.sum(data[mask])/k))
    return(data)

def int_score(x, u, l, coverage=0.95):
    alpha = 1 - coverage
    score = u - l + 2 * ((l - x) * (l > x) + (x - u) * (x > u)) / alpha
    return (np.mean(score))

def int_coverage(x, u, l):
    score = np.logical_and(x>=l, x<=u)
    return (np.mean(score))


def decor_dense(y, FI_B_local, idx=None):
    if idx is None: idx = range(y.shape[0])
    y_decor = np.matmul(FI_B_local[idx, :], y)
    return (y_decor)

def decor_sparse(y, FI_B_local, idx=None):
    if idx is None: idx = range(y.shape[0])
    n = len(idx)
    if np.ndim(y) == 2:
        p = y.shape[1]
        y_decor = np.zeros((n, p))
        for i in range(n):
            y_decor[i, :] = np.dot(FI_B_local.B[idx[i], :], y[FI_B_local.Ind_list[idx[i], :], :])
    elif np.ndim(y) == 1:
        y_decor = np.zeros(n)
        for i in range(n):
            y_decor[i] = np.dot(FI_B_local.B[idx[i], :], y[FI_B_local.Ind_list[idx[i], :]])
    return (y_decor)

for sigma in [1, 5]:
    for phi in [1, 3, 6]:
        for tau in [0.01, 0.1, 0.25]:
            method = '0'
            theta = [sigma, phi / np.sqrt(2), tau]
            '''
            method = '0'
            theta = [1, 4, 0.01]
            '''
            fun = 'sin'
            if fun == 'sin':
                p = 1; funXY = utils_NN.fx; Netp = Netp_tanh
            elif fun == 'p3':
                p = 3; funXY = utils_NN.f3; Netp = Netp_sig
            elif fun == 'friedman':
                p = 5; funXY = utils_NN.f5; Netp = Netp_sig
            elif fun == 'p15':
                p = 15; funXY = utils_NN.f15; Netp = Netp_sig
            k = 50
            q = 1
            if method == '1':
                lr = 1 if p == 1 else 0.1
            elif method == '2':
                lr = 0.5 if p == 1 else 0.05
            #lr = 0.1
            #### method 1: 1 for p=1, 0.1 for p=5; method2: 0.5 for p=1, 0.05 for p=5
            b = 10
            ordered = False

            if method == '2':
                np.random.seed(2022)
                #np.random.seed(2023)
                n = 100
                coord0 = np.random.uniform(low=0, high=b, size=(n, 2))
                theta0 = theta.copy()

                I_B, F_diag, rank, cov = utils_NN.bf_from_theta(theta, coord0, 20, method='0', nu=1.5, sparse=False)

                X = np.random.uniform(size=(n, p))
                corerr0 = utils_NN.rmvn(1, np.zeros(n), cov, I_B, F_diag, sparse = False)

                def corerr_gen(pos):
                    df0 = pd.DataFrame(coord0, columns=['x', 'y'])
                    df1 = pd.DataFrame(pos, columns=['x', 'y'])
                    mask = np.concatenate([np.repeat(True, df0.shape[0]), np.repeat(False, df1.shape[0])])
                    df = pd.concat([df0, df1])
                    dist = distance_matrix(df.values, df.values)
                    cov = utils_NN.make_cov(theta0, dist)
                    theta0[2] = 0
                    C = utils_NN.make_cov(theta0, dist)
                    del df
                    del dist
                    corerr = np.matmul(C[np.invert(mask), :][:, mask], np.linalg.solve(cov[mask, :][:, mask], corerr0))
                    del cov
                    del C
                    return (corerr)

            n = 2000
            n_train = 1000
            nn = 20
            batch_size = int(n/20)
            ADDRFGLS = False
            #if n_train <= 1000: ADDRFGLS = True
            Sparse = False
            if n > 10000: Sparse = True

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

            CI = False
            resample = 'decor'
            if CI:
                n_rep = 100
                n_sub = 0.8
                BRISC_mat = np.zeros(shape=(n_rep, N))
                GAM_mat = np.zeros(shape=(n_rep, N))
                GAMGLS_mat = np.zeros(shape=(n_rep, N))
                RF_mat = np.zeros(shape=(n_rep, N))
                NN_mat = np.zeros(shape=(n_rep, N))
                NNGLS_mat = np.zeros(shape=(n_rep, N))

            name = fun + 'phi%i' % phi + "sig%i" % (theta[0]) + "tau%i" % (int(100 * tau)) + 'mtd' + method
            #name = 'test'
            MISE_BRISC = np.empty(0)
            RMSE_BRISC = np.empty(0)
            MISE_GAM = np.empty(0)
            RMSE_GAM = np.empty(0)
            RMSE_GAM_krig = np.empty(0)
            RMSE_GAM_latlong = np.empty(0)
            MISE_GAMGLS = np.empty(0)
            RMSE_GAMGLS = np.empty(0)
            RMSE_GAMGLS_krig = np.empty(0)
            MISE_RF = np.empty(0)
            RMSE_RF = np.empty(0)
            RMSE_RF_krig = np.empty(0)
            MISE_RFGLS = np.empty(0)
            RMSE_RFGLS = np.empty(0)
            RMSE_RFGLS_krig = np.empty(0)
            MISE_NN = np.empty(0)
            RMSE_NN = np.empty(0)
            RMSE_NNlatlong = np.empty(0)
            RMSE_NNDK = np.empty(0)
            RMSE_NN_krig = np.empty(0)
            MISE_NNGLS = np.empty(0)
            RMSE_NNGLS_krig = np.empty(0)

            GMSE_BRISC = np.empty(0)
            GMSE_GAM = np.empty(0)
            GMSE_GAM_krig = np.empty(0)
            GMSE_GAM_latlong = np.empty(0)
            GMSE_GAMGLS = np.empty(0)
            GMSE_GAMGLS_krig = np.empty(0)
            GMSE_RF = np.empty(0)
            GMSE_RF_krig = np.empty(0)
            GMSE_RFGLS = np.empty(0)
            GMSE_RFGLS_krig = np.empty(0)
            GMSE_NN = np.empty(0)
            GMSE_NNlatlong = np.empty(0)
            GMSE_NNDK = np.empty(0)
            GMSE_NN_krig = np.empty(0)
            GMSE_NNGLS_krig = np.empty(0)

            df_CI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
            df_CI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
            df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
            df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

            for rand in range(1, 100+1):
                torch.manual_seed(2023+rand)
                np.random.seed(2023+rand)
                if method == '2':
                    X, Y, rank, coord, corerr = utils_NN.Simulate_mis(n, p, funXY, nn, theta, corerr_gen, a=0, b=b)
                else:
                    X, Y, I_B, F_diag, rank, coord, cov, corerr = utils_NN.Simulate_NNGP(n, p, funXY, nn, theta, method=method, a=0,
                                                                                         b=b, sparse = Sparse, meanshift= False)

                if ordered:
                    X, Y, coord = order(X, Y, coord)
                    I_B, F_diag, rank, cov = utils_NN.bf_from_theta(theta, coord, nn, sparse=Sparse)

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

                if False:
                    torch.manual_seed(2021)
                    X_train = X[range(n_train), :]
                    Y_train = Y[range(n_train)]
                    train_loader, val_loader = utils_NN.set_loader(X_train, Y_train, prop=0.8, batch_size=batch_size)
                    idx_train = np.array([], dtype = int)
                    for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
                        idx_train = np.append(idx_train, idx)
                    idx_val = np.array([], dtype = int)
                    for batch_idx, (x_batch, y_batch, idx) in enumerate(val_loader):
                        idx_val = np.append(idx_val, idx)
                    id = range(n_train)
                    mask = np.ones(n, dtype=bool)
                    mask[id] = False
                    data.test_mask = torch.from_numpy(mask)
                    mask = np.zeros(n, dtype=bool)
                    mask[idx_val] = True
                    data.val_mask = torch.from_numpy(mask)
                    mask = np.zeros(n, dtype=bool)
                    mask[idx_train] = True
                    data.train_mask = torch.from_numpy(mask)

                Y_test = Y[data.test_mask]

                torch.manual_seed(2023+rand)
                data = batch_gen(data, batch_size)

                I_B_test, F_test, _, _ = utils_NN.bf_from_theta(theta, coord[data.test_mask, :], nn, sparse=Sparse,
                                                              version='sparseB')
                F_test = F_test.detach().numpy()
                I_B_test = I_B_test.detach().numpy();
                FI_B_test = (I_B_test.T * np.sqrt(np.reciprocal(F_test))).T
                Y_test_decor = np.array(decor_dense(Y_test, FI_B_test))
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                beta, theta_hat_linear = utils_NN.BRISC_estimation(Y[~data.test_mask,], X_int[~data.test_mask,], coord[~data.test_mask,])
                def model_BRISC(X, edge_index = 0):
                    if isinstance(X, np.ndarray):
                        X = torch.from_numpy(X).float()
                    return(torch.matmul(X, torch.from_numpy(beta).float()))
                Est_MISE_BRISC = model_BRISC(X_MISE_int, edge_index).detach().numpy().reshape(-1)
                MISE_BRISC = np.append(MISE_BRISC, RMSE(Est_MISE_BRISC, Y_MISE_np.reshape(-1)))

                Pred_BRISC = utils_NN.krig_pred(model_BRISC, X_int[~data.test_mask,], X_int[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_linear)
                RMSE_BRISC = np.append(RMSE_BRISC, RMSE(Pred_BRISC[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_BRISC = np.append(GMSE_BRISC, RMSE(np.array(decor_dense(Pred_BRISC[0], FI_B_test)), Y_test_decor))
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                gam = mygam.my_LinearGAM()
                gam.fit(X[~data.test_mask,], Y[~data.test_mask])
                Xspline = gam._modelmat(X[~data.test_mask, :])
                gam.my_fit(X[~data.test_mask,], Xspline, Y[~data.test_mask])
                PI_GAM = gam.confidence_intervals(X[data.test_mask])
                def model_GAM(X, edge_index=0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    return (torch.from_numpy(gam.predict(X)).reshape(-1))
                Est_MISE_GAM = model_GAM(X_MISE, edge_index).detach().numpy().reshape(-1)
                Pred_GAM = model_GAM(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_GAM = np.append(RMSE_GAM, RMSE(Pred_GAM, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAM = np.append(GMSE_GAM, RMSE(np.array(decor_dense(Pred_GAM, FI_B_test)), Y_test_decor))
                MISE_GAM = np.append(MISE_GAM, RMSE(Est_MISE_GAM, Y_MISE_np.reshape(-1)))

                Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_GAM = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_GAM = utils_NN.krig_pred(model_GAM, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAM)
                RMSE_GAM_krig = np.append(RMSE_GAM_krig,
                                            RMSE(Pred_GAM[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAM_krig = np.append(GMSE_GAM_krig, RMSE(np.array(decor_dense(Pred_GAM[0], FI_B_test)), Y_test_decor))
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                X_coord = np.concatenate((X, coord), axis=1)
                gam = mygam.my_LinearGAM()
                gam.fit(X_coord[~data.test_mask,], Y[~data.test_mask])
                Xspline = gam._modelmat(X_coord[~data.test_mask, :])
                gam.my_fit(X_coord[~data.test_mask,], Xspline, Y[~data.test_mask])
                PI_GAM_latlong = gam.confidence_intervals(X_coord[data.test_mask])

                def model_GAM(X, edge_index=0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    return (torch.from_numpy(gam.predict(X)).reshape(-1))

                Pred_GAM_latlong = model_GAM(X_coord[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_GAM_latlong = np.append(RMSE_GAM_latlong, RMSE(Pred_GAM_latlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAM_latlong = np.append(GMSE_GAM_latlong, RMSE(np.array(decor_dense(Pred_GAM_latlong, FI_B_test)), Y_test_decor))
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                gam = mygam.my_LinearGAM()
                gam.fit(X[~data.test_mask,], Y[~data.test_mask])
                Xspline = gam._modelmat(X[~data.test_mask, :])
                I_B_GAM, F_GAM, _, _ = utils_NN.bf_from_theta(theta_hat_GAM, coord[~data.test_mask, :], nn, sparse=Sparse,
                                                              version='sparseB')
                F_GAM = F_GAM.detach().numpy()
                if Sparse:
                    FI_B_GAM = I_B_GAM.Fmul(np.sqrt(np.reciprocal(F_GAM))).to_tensor()
                    gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(decor_sparse(Xspline.todense(), FI_B_GAM))),
                           np.array(decor_sparse(Y[~data.test_mask], FI_B_GAM)))
                else:
                    I_B_GAM = I_B_GAM.detach().numpy();
                    FI_B_GAM = (I_B_GAM.T * np.sqrt(np.reciprocal(F_GAM))).T
                    gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(decor_dense(Xspline.todense(), FI_B_GAM))),
                               np.array(decor_dense(Y[~data.test_mask], FI_B_GAM)))
                del I_B_GAM, F_GAM, FI_B_GAM
                PI_GAMGLS = gam.confidence_intervals(X[data.test_mask])
                def model_GAMGLS(X, edge_index=0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    return (torch.from_numpy(gam.predict(X)).reshape(-1))
                Est_MISE_GAMGLS = model_GAMGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                Pred_GAMGLS = model_GAMGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_GAMGLS = np.append(RMSE_GAMGLS, RMSE(Pred_GAMGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAMGLS = np.append(GMSE_GAMGLS, RMSE(np.array(decor_dense(Pred_GAMGLS, FI_B_test)), Y_test_decor))
                MISE_GAMGLS = np.append(MISE_GAMGLS, RMSE(Est_MISE_GAMGLS, Y_MISE_np.reshape(-1)))

                Y_hat = model_GAMGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_GAMGLS = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_GAMGLS = utils_NN.krig_pred(model_GAMGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAMGLS)
                RMSE_GAMGLS_krig = np.append(RMSE_GAMGLS_krig,
                                            RMSE(Pred_GAMGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAMGLS_krig = np.append(GMSE_GAMGLS_krig, RMSE(np.array(decor_dense(Pred_GAMGLS[0], FI_B_test)), Y_test_decor))
                ########################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                n_tree = 60
                node_size = 150 if sigma == 5 and p == 5 else 60

                rf = RandomForestRegressor(n_estimators = n_tree, min_samples_split=node_size)
                rf.fit(X[~data.test_mask,], Y[~data.test_mask,])
                def model_RF(X, edge_index = 0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    return(torch.from_numpy(rf.predict(X)))
                Est_MISE_RF = model_RF(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_RF = np.append(MISE_RF, RMSE(Est_MISE_RF, Y_MISE_np.reshape(-1)))
                print(RMSE(Est_MISE_RF, Y_MISE_np.reshape(-1)))
                Pred_RF = model_RF(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_RF = np.append(RMSE_RF, RMSE(Pred_RF, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_RF = np.append(GMSE_RF, RMSE(np.array(decor_dense(Pred_RF, FI_B_test)), Y_test_decor))

                Y_hat = model_RF(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_RF = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_RF = utils_NN.krig_pred(model_RF, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RF)
                RMSE_RF_krig = np.append(RMSE_RF_krig,
                                            RMSE(Pred_RF[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_RF_krig = np.append(GMSE_RF_krig, RMSE(np.array(decor_dense(Pred_RF[0], FI_B_test)), Y_test_decor))
                ########################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                if ADDRFGLS:
                    robjects.globalenv['.Random.seed'] = 1
                    rfgls = utils_NN.RFGLS_prediction(X[~data.test_mask,], Y[~data.test_mask], coord[~data.test_mask,],
                                                      n_tree=n_tree, node_size=node_size)
                    def model_RFGLS(X, edge_index = 0):
                        if torch.is_tensor(X):
                            X = X.detach().numpy()
                        X_r = utils_NN.robjects.FloatVector(X.transpose().reshape(-1))
                        X_r = utils_NN.robjects.r['matrix'](X_r, ncol=X.shape[1])
                        predict = utils_NN.RFGLS.RFGLS_predict(rfgls, X_r)[1]
                        return(torch.from_numpy(np.array(predict)))
                    Est_MISE_RFGLS = model_RFGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                    Pred_RFGLS = model_RFGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                    RMSE_RFGLS = np.append(RMSE_RFGLS, RMSE(Pred_RFGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                    GMSE_RFGLS = np.append(GMSE_RFGLS, RMSE(np.array(decor_dense(Pred_RFGLS, FI_B_test)), Y_test_decor))
                    MISE_RFGLS = np.append(MISE_RFGLS, RMSE(Est_MISE_RFGLS, Y_MISE_np.reshape(-1)))

                    Y_hat = model_RFGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                    residual = data.y - torch.from_numpy(Y_hat)
                    residual_train = residual[~data.test_mask]
                    residual_train = residual_train.detach().numpy()
                    _, theta_hat_RFGLS = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                    Pred_RFGLS = utils_NN.krig_pred(model_RFGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                    coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RFGLS)
                    RMSE_RFGLS_krig = np.append(RMSE_RFGLS_krig,
                                                RMSE(Pred_RFGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                    GMSE_RFGLS_krig = np.append(GMSE_RFGLS_krig, RMSE(np.array(decor_dense(Pred_RFGLS[0], FI_B_test)), Y_test_decor))
                ######################################################################################################################
                torch.manual_seed(2023)
                model_NN = Netp(p, k)
                optimizer = torch.optim.Adam(model_NN.parameters(), lr=lr)
                patience_half = 10
                patience = 20

                _, _, model_NN = utils_NN.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                                      patience = patience, patience_half = patience_half)
                Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NN = np.append(GMSE_NN, RMSE(np.array(decor_dense(Pred_NN, FI_B_test)), Y_test_decor))
                Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_NN = np.append(MISE_NN, RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
                ####################################################################################
                torch.manual_seed(2023)
                model_NNlatlong = Netp(p + 2, k, q)
                optimizer = torch.optim.Adam(model_NNlatlong.parameters(), lr=lr)
                patience_half = 10
                patience = 20
                data_latlong = copy.copy(data)

                data_latlong.x = torch.concatenate((data.x, torch.from_numpy(data.coord)), axis=1).float()

                _, _, model_NNlatlong = utils_NN.train_gen_new(model_NNlatlong, optimizer, data_latlong, epoch_num=1000,
                                                               patience=patience, patience_half=patience_half)
                Pred_NNlatlong = model_NNlatlong(data_latlong.x[data_latlong.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_NNlatlong = np.append(RMSE_NNlatlong, RMSE(Pred_NNlatlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NNlatlong = np.append(GMSE_NNlatlong, RMSE(np.array(decor_dense(Pred_NNlatlong, FI_B_test)), Y_test_decor))
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
                _, _, model_NNDK = utils_NN.train_gen_new(model_NNDK, optimizer, data_DK, epoch_num=1000,
                                                          patience=patience, patience_half=patience_half)
                Pred_NNDK = model_NNDK(data_DK.x[data_DK.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_NNDK = np.append(RMSE_NNDK, RMSE(Pred_NNDK, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NNDK = np.append(GMSE_NNDK, RMSE(np.array(decor_dense(Pred_NNDK, FI_B_test)), Y_test_decor))
                ####################################################################################################################
                Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                beta_hat, theta_hat = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                theta_hat0 = theta_hat.copy()

                Pred_NN = utils_NN.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
                RMSE_NN_krig = np.append(RMSE_NN_krig,
                                            RMSE(Pred_NN[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NN_krig = np.append(GMSE_NN_krig, RMSE(np.array(decor_dense(Pred_NN[0], FI_B_test)), Y_test_decor))
                ###################################################################################
                torch.manual_seed(2023)
                model_NNGLS = Netp(p, k, q)
                optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
                patience_half = 10
                patience = 20

                _, _, model_NNGLS = utils_NN.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse = Sparse,
                                                        Update=False, patience=patience, patience_half=patience_half)
                Y_hat_NNGLS = model_NNGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual_NNGLS = data.y - torch.from_numpy(Y_hat_NNGLS)
                residual_NNGLS_np = residual_NNGLS.detach().numpy()
                beta_hat_NNGLS, theta_hat_NNGLS = utils_NN.BRISC_estimation(residual_NNGLS_np[~data.test_mask,], X[~data.test_mask,],
                                                                            coord[~data.test_mask,])

                Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
                Pred_NNGLS = utils_NN.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
                RMSE_NNGLS_krig = np.append(RMSE_NNGLS_krig,
                                            RMSE(Pred_NNGLS[0], Y_test)/RMSE(Y_test, np.mean(Y_test)))
                GMSE_NNGLS_krig = np.append(GMSE_NNGLS_krig, RMSE(np.array(decor_dense(Pred_NNGLS[0], FI_B_test)), Y_test_decor))
                ####################################################################################################################
                df_PI1_temp = {'BRISC': int_coverage(Y_test, Pred_BRISC[1], Pred_BRISC[2]),
                              'GAM': int_coverage(Y_test, Pred_GAM[1], Pred_GAM[2]),
                              'GAM_latlong': int_coverage(Y_test, PI_GAM_latlong[:,0], PI_GAM_latlong[:,1]),
                              'GAMGLS': int_coverage(Y_test, Pred_GAMGLS[1], Pred_GAMGLS[2]),
                              'RF': int_coverage(Y_test, Pred_RF[1], Pred_RF[2]),
                              #'RFGLS': int_coverage(Y_test, Pred_RFGLS[1], Pred_RFGLS[2]),
                              'NN': int_coverage(Y_test, Pred_NN[1], Pred_NN[2]),
                              'NNGLS': int_coverage(Y_test, Pred_NNGLS[1], Pred_NNGLS[2])}

                df_PI2_temp = {'BRISC': int_score(Y_test, Pred_BRISC[1], Pred_BRISC[2], 0.95),
                              'GAM': int_score(Y_test, Pred_GAM[1], Pred_GAM[2], 0.95),
                              'GAM_latlong': int_score(Y_test, PI_GAM_latlong[:, 0], PI_GAM_latlong[:, 1], 0.95),
                              'GAMGLS': int_score(Y_test, Pred_GAMGLS[1], Pred_GAMGLS[2], 0.95),
                              'RF': int_score(Y_test, Pred_RF[1], Pred_RF[2], 0.95),
                              #'RFGLS': int_score(Y_test, Pred_RFGLS[1], Pred_RFGLS[2], 0.95),
                              'NN': int_score(Y_test, Pred_NN[1], Pred_NN[2], 0.95),
                              'NNGLS': int_score(Y_test, Pred_NNGLS[1], Pred_NNGLS[2], 0.95)}
                ####################################################################################################################
                if CI:
                    if resample == 'decor' or resample == 'oracle':
                        I_B_temp, F_diag_temp, _, _ = utils_NN.bf_from_theta(theta_hat_NNGLS, coord, nn, sparse=Sparse)
                        if not Sparse:
                            I_B_inv_temp = torch.inverse(I_B_temp)
                        sigma_sq_hat, phi_hat, tau_hat = theta_hat_NNGLS
                        tau_sq_hat = tau_hat * sigma_sq_hat
                        dist = utils_NN.distance(coord, coord)
                        cov_hat = sigma_sq_hat * np.exp(-phi_hat * dist) + tau_sq_hat * np.eye(n)
                    for rep in range(n_rep):
                        if resample == 'subsample':
                            np.random.seed(2023 + rep * rand)
                            id = np.random.choice(np.where(data.test_mask == False)[0], int(n_train * n_sub), replace=False)
                            id = np.sort(np.concatenate([id, np.where(data.test_mask == True)[0]], 0))
                            X_sub = X[id, :]
                            Y_sub = Y[id]
                            Y_sub_NNGLS = Y[id]
                            coord_sub = coord[id, :]
                            if not Sparse:
                                I_B_sub = I_B[id, :][:, id]
                                F_diag_sub = F_diag[id]
                        elif resample == 'decor':
                            torch.manual_seed(2023 + rep * rand)
                            np.random.seed(2023 + rep * rand)
                            id = range(n)
                            X_sub = X
                            Y_sub = Y_hat + np.random.choice(residual.detach().numpy(), n, replace=False)
                            if Sparse:
                                # Y_sub = Y_hat + utils_NN.resample_fun_sparseB(residual, coord, nn, theta_hat0)
                                Y_sub_NNGLS = Y_hat_NNGLS + utils_NN.resample_fun_sparseB(residual_NNGLS, coord, nn,
                                                                                          theta_hat_NNGLS)
                            else:
                                # Y_sub = Y_hat + utils_NN.resample_fun(residual, I_B_temp, I_B_inv_temp, F_diag_temp).detach().numpy()
                                Y_sub_NNGLS = Y_hat_NNGLS + utils_NN.resample_fun(residual_NNGLS, I_B_temp, I_B_inv_temp,
                                                                                  F_diag_temp).detach().numpy()
                            coord_sub = coord
                            I_B_sub = I_B_temp
                            F_diag_sub = F_diag_temp

                        neigh = NearestNeighbors(n_neighbors=nn)
                        neigh.fit(coord_sub)

                        A = neigh.kneighbors_graph(coord_sub)
                        A.toarray()
                        edge_index_sub = torch.from_numpy(np.stack(A.nonzero()))

                        torch.manual_seed(2023 + rand)
                        data_sub = Data(x=torch.from_numpy(X_sub).float(), edge_index=edge_index_sub,
                                        y=torch.from_numpy(Y_sub).float(), coord = coord_sub)
                        data_sub.train_mask = data.train_mask[id]
                        data_sub.val_mask = data.val_mask[id]
                        data_sub.test_mask = data.test_mask[id]
                        data_sub.n = data_sub.x.shape[0]

                        torch.manual_seed(2023 + rand)
                        data_sub_NNGLS = Data(x=torch.from_numpy(X_sub).float(), edge_index=edge_index_sub,
                                        y=torch.from_numpy(Y_sub_NNGLS).float(), coord=coord_sub)
                        data_sub_NNGLS.train_mask = data.train_mask[id]
                        data_sub_NNGLS.val_mask = data.val_mask[id]
                        data_sub_NNGLS.test_mask = data.test_mask[id]
                        data_sub_NNGLS.n = data_sub_NNGLS.x.shape[0]

                        torch.manual_seed(2023 + rand)
                        data_sub = batch_gen(data_sub, int(batch_size * n_sub))
                        torch.manual_seed(2023 + rand)
                        data_sub_NNGLS = batch_gen(data_sub_NNGLS, int(batch_size * n_sub))
                        ####################################################################################
                        torch.manual_seed(2023 + rand)
                        np.random.seed(2023 + rand)
                        beta, theta_hat_linear = utils_NN.BRISC_estimation(Y_sub[~data_sub.test_mask,], X_sub[~data_sub.test_mask,],
                                                                           coord_sub[~data_sub.test_mask,])
                        def model_BRISC(X, edge_index=0):
                            if isinstance(X, np.ndarray):
                                X = torch.from_numpy(X).float()
                            return (torch.matmul(X, torch.from_numpy(beta).float()))
                        BRISC_mat[rep, :] = model_BRISC(X_MISE, edge_index).detach().numpy().reshape(-1)
                        ####################################################################################
                        torch.manual_seed(2023 + rand)
                        np.random.seed(2023 + rand)
                        gam = mygam.my_LinearGAM()
                        gam.fit(X[~data.test_mask,], Y[~data.test_mask])
                        Xspline = gam._modelmat(X[~data.test_mask, :])
                        gam.my_fit(X[~data.test_mask,], Xspline, Y[~data.test_mask])
                        def model_GAM(X, edge_index=0):
                            if torch.is_tensor(X):
                                X = X.detach().numpy()
                            return (torch.from_numpy(gam.predict(X)).reshape(-1))
                        GAM_mat[rep, :] = model_GAM(X_MISE, edge_index_sub).detach().numpy().reshape(-1)
                        ####################################################################################
                        '''
                        torch.manual_seed(2023 + rand)
                        np.random.seed(2023 + rand)
                        gam = mygam.my_LinearGAM()
                        gam.fit(X[~data.test_mask,], Y[~data.test_mask])
                        Xspline = gam._modelmat(X[~data.test_mask, :])
                        I_B_GAM, F_GAM, _, _ = utils_NN.bf_from_theta(theta_hat_GAM, coord[~data.test_mask, :], nn, sparse=Sparse,
                                                                      version='sparseB')
                        F_GAM = F_GAM.detach().numpy()
                        if Sparse:
                            FI_B_GAM = I_B_GAM.Fmul(np.sqrt(np.reciprocal(F_GAM))).to_tensor()
                            gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(decor_sparse(Xspline.todense(), FI_B_GAM))),
                                       np.array(decor_sparse(Y[~data.test_mask], FI_B_GAM)))
                        else:
                            I_B_GAM = I_B_GAM.detach().numpy();
                            FI_B_GAM = (I_B_GAM.T * np.sqrt(np.reciprocal(F_GAM))).T
                            gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(decor_dense(Xspline.todense(), FI_B_GAM))),
                                       np.array(decor_dense(Y[~data.test_mask], FI_B_GAM)))
                        del I_B_GAM, F_GAM, FI_B_GAM
                        def model_GAMGLS(X, edge_index=0):
                            if torch.is_tensor(X):
                                X = X.detach().numpy()
                            return (torch.from_numpy(gam.predict(X)).reshape(-1))
                        GAMGLS_mat[rep, :] = model_GAMGLS(X_MISE, edge_index_sub).detach().numpy().reshape(-1)
                        '''
                        ####################################################################################
                        torch.manual_seed(2023 + rand)
                        np.random.seed(2023 + rand)
                        rf = RandomForestRegressor(min_samples_split=5)
                        rf.fit(X_sub[~data_sub.test_mask,], Y_sub[~data_sub.test_mask,])
                        def model_RF(X, edge_index=0):
                            if torch.is_tensor(X):
                                X = X.detach().numpy()
                            return (torch.from_numpy(rf.predict(X)))

                        RF_mat[rep, :] = model_RF(X_MISE, edge_index_sub).detach().numpy().reshape(-1)
                        ####################################################################################
                        torch.manual_seed(2023 + rand)
                        model0 = Netp(p, k, q)
                        optimizer = torch.optim.Adam(model0.parameters(), lr=0.1)
                        patience_half = 5
                        patience = 20

                        _, _, model0 = utils_NN.train_gen_new(model0, optimizer, data_sub, epoch_num=1000,
                                                              patience=patience, patience_half=patience_half)

                        NN_mat[rep, :] = model0(X_MISE, edge_index_sub).detach().numpy().reshape(-1)
                        ###################################################################################
                        torch.manual_seed(2023 + rand)
                        model1 = Netp(p, k, q)
                        optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)
                        patience_half = 5
                        patience = 20

                        if resample == 'subsample' and Sparse:
                            _, _, model1 = utils_NN.train_decor_new(model1, optimizer, data_sub_NNGLS, 1000, theta_hat_NNGLS,
                                                                    sparse=Sparse, sparseB=True,
                                                                    Update=False, patience=patience, patience_half=patience_half)
                        else:
                            _, _, model1 = utils_NN.train_decor_new(model1, optimizer, data_sub_NNGLS, 1000, theta_hat_NNGLS,
                                                                    sparse=Sparse, sparseB=True, BF=[I_B_sub, F_diag_sub],
                                                                    Update=False, patience=patience, patience_half=patience_half)

                        NNGLS_mat[rep, :] = model1(X_MISE, edge_index_sub).detach().numpy().reshape(-1)

                    CI_U_BRISC, CI_L_BRISC = np.quantile(BRISC_mat, 0.975, axis=0), np.quantile(BRISC_mat, 0.025, axis=0)
                    CI_U_GAM, CI_L_GAM = np.quantile(GAM_mat, 0.975, axis=0), np.quantile(GAM_mat, 0.025, axis=0)
                    CI_U_GAMGLS, CI_L_GAMGLS = np.quantile(GAMGLS_mat, 0.975, axis=0), np.quantile(GAMGLS_mat, 0.025, axis=0)
                    CI_U_RF, CI_L_RF = np.quantile(RF_mat, 0.975, axis=0), np.quantile(RF_mat, 0.025, axis=0)
                    CI_U_NN, CI_L_NN = np.quantile(NN_mat, 0.975, axis=0), np.quantile(NN_mat, 0.025, axis=0)
                    CI_U_NNGLS, CI_L_NNGLS = np.quantile(NNGLS_mat, 0.975, axis=0), np.quantile(NNGLS_mat, 0.025, axis=0)

                    df_CI1_temp = {'BRISC': int_coverage(Y_MISE_np, CI_U_BRISC, CI_L_BRISC),
                                   'GAM': int_coverage(Y_MISE_np, CI_U_GAM, CI_L_GAM),
                                   'GAMGLS': int_coverage(Y_MISE_np, CI_U_GAMGLS, CI_L_GAMGLS),
                                   'RF': int_coverage(Y_MISE_np, CI_U_RF, CI_L_RF),
                                   'NN': int_coverage(Y_MISE_np, CI_U_NN, CI_L_NN),
                                   'NNGLS': int_coverage(Y_MISE_np, CI_U_NNGLS, CI_L_NNGLS)}
                    df_CI2_temp = {'BRISC': int_score(Y_MISE_np, CI_U_BRISC, CI_L_BRISC, 0.95),
                                  'GAM': int_score(Y_MISE_np, CI_U_GAM, CI_L_GAM, 0.95),
                                  'GAMGLS': int_score(Y_MISE_np, CI_U_GAMGLS, CI_L_GAMGLS, 0.95),
                                  'RF': int_score(Y_MISE_np, CI_U_RF, CI_L_RF, 0.95),
                                  'NN': int_score(Y_MISE_np, CI_U_NN, CI_L_NN, 0.95),
                                  'NNGLS': int_score(Y_MISE_np, CI_U_NNGLS, CI_L_NNGLS, 0.95)}

                    df_CI1 = df_CI1.append(df_CI1_temp, ignore_index=True)
                    df_CI1.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_CI_cov.csv')
                    df_CI2 = df_CI2.append(df_CI2_temp, ignore_index=True)
                    df_CI2.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_CI_score.csv')
                ####################################################################################################################
                df_PI1 = df_PI1.append(df_PI1_temp, ignore_index=True)
                df_PI1.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_PI_cov.csv')
                df_PI2 = df_PI2.append(df_PI2_temp, ignore_index=True)
                df_PI2.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_PI_score.csv')
                df_MISE = pd.DataFrame(
                    {'BRISC': MISE_BRISC, 'GAM': MISE_GAM, 'GAMGLS': MISE_GAMGLS, 'RF': MISE_RF,
                     'NN': MISE_NN, 'NNGLS': MISE_NNGLS})
                df_RMSE = pd.DataFrame(
                    {'BRISC': RMSE_BRISC, 'GAM': RMSE_GAM, 'GAM_krig': RMSE_GAM_krig, 'GAM_latlong': RMSE_GAM_latlong,
                     'GAMGLS': RMSE_GAMGLS, 'GAMGLS_krig': RMSE_GAMGLS_krig,
                     'RF': RMSE_RF, 'RF_krig': RMSE_RF_krig,
                     'NN_latlong': RMSE_NNlatlong, 'NNDK': RMSE_NNDK,
                     'NN': RMSE_NN, 'NN_krig': RMSE_NN_krig, 'NNGLS_krig': RMSE_NNGLS_krig})
                df_GMSE = pd.DataFrame(
                    {'BRISC': GMSE_BRISC, 'GAM': GMSE_GAM, 'GAM_krig': GMSE_GAM_krig, 'GAM_latlong': GMSE_GAM_latlong,
                     'GAMGLS': GMSE_GAMGLS, 'GAMGLS_krig': GMSE_GAMGLS_krig,
                     'RF': GMSE_RF, 'RF_krig': GMSE_RF_krig,
                     'NN_latlong': GMSE_NNlatlong, 'NNDK': GMSE_NNDK,
                     'NN': GMSE_NN, 'NN_krig': GMSE_NN_krig, 'NNGLS_krig': GMSE_NNGLS_krig})
                if ADDRFGLS:
                    df_MISE['RFGLS'] = MISE_RFGLS
                    df_RMSE['RFGLS'] = RMSE_RFGLS
                    df_RMSE['RFGLS_krig'] = RMSE_RFGLS_krig
                    df_GMSE['RFGLS'] = GMSE_RFGLS
                    df_GMSE['RFGLS_krig'] = GMSE_RFGLS_krig
                df_MISE.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_MISE.csv')
                df_RMSE.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_RMSE.csv')
                df_GMSE.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_GMSE.csv')

