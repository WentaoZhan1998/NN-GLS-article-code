#### This file produces simulation results for large sample behavior in section S4.7, Figure S16 or Figure 2(e, f).

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
import time

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

p = 5; funXY = utils.f5
Netp = utils.Netp_sig

sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
k = 50
q = 1
lr = 0.01
max_epoch = 1000 # = 10 for figure S16(b)

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

name = 'running_time'
MISE_BRISC = np.empty(0)
MISE_GAM = np.empty(0)
MISE_GAMGLS = np.empty(0)
MISE_RF = np.empty(0)
MISE_RFGLS = np.empty(0)
MISE_NN = np.empty(0)
MISE_NNGLS = np.empty(0)
size = np.empty(0)
epoch_NN = np.empty(0)
epoch_NNlatlong = np.empty(0)
epoch_NNDK = np.empty(0)
epoch_NNGLS = np.empty(0)

df_t = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS',
                             'NN', 'NN_latlong', 'NNDK', 'NNGLS',
                             'Sim', 'Krig', 'Size'])
for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]:
    for rand in range(1, 10 + 1):
        size = np.append(size, n)
        n = int(n)
        b = np.sqrt(n/1000)
        n_train = int(n/2)
        nn = 20
        n_batch = 20
        batch_size = int(n / n_batch)
        ADDRFGLS = False
        Sparse = False
        if n_train <= 1000: ADDRFGLS = True
        if n > 10000: Sparse = True
        t_start = time.time()
        torch.manual_seed(2023+rand)
        np.random.seed(2023+rand)
        X, Y, I_B, F_diag, _, coord, _, _ = utils.Simulate(n, p, funXY, nn, theta, method=method, a=0,
                                                           b=b, sparse = Sparse)
        X_int = np.concatenate((X, np.repeat(1, n).reshape(n, 1)), axis=1)
        edge_index = 0
        
        torch.manual_seed(2023+rand)
        data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord = coord)
        transform = T.RandomNodeSplit(num_train_per_class=int(0.3*n_train), num_val=int(0.2*n_train), num_test=int(n-n_train))
        data = transform(data)
        data.n = data.x.shape[0]
        
        Y_test = Y[data.test_mask]
        
        torch.manual_seed(2023+rand)
        data = utils.batch_gen(data, batch_size)
        t_simulate = time.time() - t_start
        ####################################################################################################################
        t_start = time.time()
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        beta, theta_hat_linear = utils.BRISC_estimation(Y[~data.test_mask,], X_int[~data.test_mask,],
                                                           coord[~data.test_mask,])
        def model_BRISC(X, edge_index=0):
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            return (torch.matmul(X, torch.from_numpy(beta).float()))
        t_BRISC = time.time() - t_start

        Est_MISE_BRISC = model_BRISC(X_MISE_int, edge_index).detach().numpy().reshape(-1)
        MISE_BRISC = np.append(MISE_BRISC, RMSE(Est_MISE_BRISC, Y_MISE_np.reshape(-1)))
        ####################################################################################################################
        t_start = time.time()
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        gam = utils_pygam.my_LinearGAM()
        gam.fit(X[~data.test_mask,], Y[~data.test_mask])
        Xspline = gam._modelmat(X[~data.test_mask, :])
        gam.my_fit(X[~data.test_mask,], Xspline, Y[~data.test_mask])
        def model_GAM(X, edge_index=0):
            if torch.is_tensor(X):
                X = X.detach().numpy()
            return (torch.from_numpy(gam.predict(X)).reshape(-1))

        t_GAM = time.time() - t_start
        Est_MISE_GAM = model_GAM(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_GAM = np.append(MISE_GAM, RMSE(Est_MISE_GAM, Y_MISE_np.reshape(-1)))

        Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = torch.from_numpy(Y_hat) - data.y
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        _, theta_hat_GAM = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        ########################################################################################################################
        t_start = time.time()
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        rf = RandomForestRegressor(min_samples_split=5)
        rf.fit(X[~data.test_mask,], Y[~data.test_mask,])
        def model_RF(X, edge_index = 0):
            if torch.is_tensor(X):
                X = X.detach().numpy()
            return(torch.from_numpy(rf.predict(X)))

        t_RF = time.time() - t_start
        Est_MISE_RF = model_RF(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_RF = np.append(MISE_RF, RMSE(Est_MISE_RF, Y_MISE_np.reshape(-1)))
        ########################################################################################################################
        t_start = time.time()
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        if ADDRFGLS:
            robjects.globalenv['.Random.seed'] = 1
            rfgls = utils.RFGLS_prediction(X[~data.test_mask,], Y[~data.test_mask], coord[~data.test_mask,], X_MISE)
            def model_RFGLS(X, edge_index = 0):
                if torch.is_tensor(X):
                    X = X.detach().numpy()
                X_r = utils.robjects.FloatVector(X.transpose().reshape(-1))
                X_r = utils.robjects.r['matrix'](X_r, ncol=X.shape[1])
                predict = utils.RFGLS.RFGLS_predict(rfgls, X_r)[1]
                return(torch.from_numpy(np.array(predict)))
            Est_MISE_RFGLS = model_RFGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
            MISE_RFGLS = np.append(MISE_RFGLS, RMSE(Est_MISE_RFGLS, Y_MISE_np.reshape(-1)))
        else: MISE_RFGLS = np.append(MISE_RFGLS, 0)
        t_RFGLS = time.time() - t_start
        ######################################################################################################################
        t_start = time.time()
        torch.manual_seed(2023)
        model_NN = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NN.parameters(), lr=lr)
        patience_half = 10
        patience = 20

        epoch_num_NN, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=max_epoch,
                                              patience = patience, patience_half = patience_half)
        t_NN = time.time() - t_start
        Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
        print(RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
        MISE_NN = np.append(MISE_NN, RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
        epoch_NN = np.append(epoch_NN, epoch_num_NN)
        ####################################################################################
        t_start = time.time()
        torch.manual_seed(2023)
        model_NNlatlong = Netp(p + 2, k, q)
        optimizer = torch.optim.Adam(model_NNlatlong.parameters(), lr=lr)
        patience_half = 10
        patience = 20
        data_latlong = copy.copy(data)

        data_latlong.x = torch.concatenate((data.x, torch.from_numpy(data.coord)), axis=1).float()

        epoch_num_NNlatlong, _, model_NNlatlong = utils.train_gen_new(model_NNlatlong, optimizer, data_latlong, epoch_num=max_epoch,
                                                       patience=patience, patience_half=patience_half)
        t_NNlatlong = time.time() - t_start
        epoch_NNlatlong = np.append(epoch_NNlatlong, epoch_num_NNlatlong)
        ####################################################################################
        t_start = time.time()
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
        model_NNDK = Netp(p + K, k, q)
        optimizer = torch.optim.Adam(model_NNDK.parameters(), lr=lr)
        patience_half = 10
        patience = 20
        data_DK = copy.copy(data)
        data_DK.x = torch.concatenate((data.x, torch.from_numpy(phi_temp)), axis=1).float()
        epoch_num_NNDK, _, model_NNDK = utils.train_gen_new(model_NNDK, optimizer, data_DK, epoch_num=max_epoch,
                                                  patience=patience, patience_half=patience_half)
        t_NNDK = time.time() - t_start
        epoch_NNDK = np.append(epoch_NNDK, epoch_num_NNDK)
        ####################################################################################################################
        t_start = time.time()
        Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = torch.from_numpy(Y_hat) - data.y
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        theta_hat0 = theta_hat

        Pred_NN = utils.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
        t_krig = time.time() - t_start
        ####################################################################################################################
        t_start = time.time()
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
            gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(utils.decor_sparse_np(Xspline.todense(), FI_B_GAM))),
                       np.array(utils.decor_sparse_np(Y[~data.test_mask], FI_B_GAM)))
        else:
            I_B_GAM = I_B_GAM.detach().numpy()
            FI_B_GAM = (I_B_GAM.T * np.sqrt(np.reciprocal(F_GAM))).T
            gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(utils.decor_dense_np(Xspline.todense(), FI_B_GAM))),
                       np.array(utils.decor_dense_np(Y[~data.test_mask], FI_B_GAM)))
        del I_B_GAM, F_GAM, FI_B_GAM
        def model_GAMGLS(X, edge_index=0):
            if torch.is_tensor(X):
                X = X.detach().numpy()
            return (torch.from_numpy(gam.predict(X)).reshape(-1))

        t_GAMGLS = time.time() - t_start
        Est_MISE_GAMGLS = model_GAMGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_GAMGLS = np.append(MISE_GAMGLS, RMSE(Est_MISE_GAMGLS, Y_MISE_np.reshape(-1)))
        ###################################################################################
        t_start = time.time()
        torch.manual_seed(2023)
        model_NNGLS = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=lr)
        patience_half = 10
        patience = 20

        epoch_num_NNGLS, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, max_epoch, theta_hat0,
                                                     sparse = Sparse, sparseB = True,
                                                     Update=False, patience=patience, patience_half=patience_half)
        t_NNGLS = time.time() - t_start
        Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
        epoch_NNGLS = np.append(epoch_NNGLS, epoch_num_NNGLS)

        df_t_temp = {'BRISC': t_BRISC,
                     'GAM': t_GAM, 'GAMGLS': t_GAMGLS,
                     'RF': t_RF, 'RFGLS': t_RFGLS,
                     'NN': t_NN, 'NN_latlong': t_NNlatlong, 'NNDK': t_NNDK, 'NNGLS': t_NNGLS,
                     'Sim':t_simulate, 'Krig': t_krig,
                     'Size':n}

        df_t = df_t.append(df_t_temp, ignore_index=True)
        df_t.to_csv(".//simulation//large//" + name + '_t.csv')

        df_MISE = pd.DataFrame(
            {'BRISC': MISE_BRISC,
             'GAM': MISE_GAM, 'GAMGLS': MISE_GAMGLS,
             'RF': MISE_RF, 'RFGLS': MISE_RFGLS,
             'NN': MISE_NN, 'NNGLS': MISE_NNGLS,
             'Size':size})
        df_MISE.to_csv(".//simulation//large//" + name + '_MISE.csv')

        df_epoch = pd.DataFrame(
            {'NN': epoch_NN, 'NN-latlong': epoch_NNlatlong, 'NNDK': epoch_NNGLS, 'NNGLS': epoch_NNGLS,
             'Size': size})
        df_epoch.to_csv(".//simulation//large//" + name + '_epoch.csv')
