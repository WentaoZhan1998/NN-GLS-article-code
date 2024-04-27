#### This file produces simulation results for estimation, prediction, and prediction interval for f1

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

fun = 'friedman'
p = 5; funXY = utils.f5; Netp = utils.Netp_sig
method = '2' #### '0' is the normal case, '1' is the first misspecification case, and '2' is the second one

for sigma in [1, 5]:
    for phi in [1, 3, 6]:
        for tau in [0.01, 0.1, 0.25]:
            theta = [sigma, phi / np.sqrt(2), tau]
            k = 50
            q = 1
            b = 10 #### Generate coordinates from [0, b]^2 square

            if method == '2':
                np.random.seed(2022)
                n = 100
                coord0 = np.random.uniform(low=0, high=b, size=(n, 2))
                theta0 = theta.copy()

                I_B, F_diag, rank, cov = utils.bf_from_theta(theta, coord0, 20, method='0', nu=1.5, sparse=False)

                X = np.random.uniform(size=(n, p))
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

            n = 2000
            n_train = 1000
            nn = 20
            batch_size = int(n/20)
            ADDRFGLS = False
            if n_train <= 1000: ADDRFGLS = True
            Sparse = False
            if n > 10000: Sparse = True
            lr = 0.1

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

            name = fun + 'phi%i' % phi + "sig%i" % (theta[0]) + "tau%i" % (int(100 * tau)) + 'mtd' + method

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

            df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
            df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

            for rand in range(1, 100+1):
                torch.manual_seed(2023+rand)
                np.random.seed(2023+rand)
                if method == '2':
                    X, Y, rank, coord, corerr = utils.Simulate_mis(n, p, funXY, nn, theta, corerr_gen, a=0, b=b)
                else:
                    X, Y, I_B, F_diag, rank, coord, cov, corerr = utils.Simulate(n, p, funXY, nn, theta, method=method,
                                                                                 a=0, b=b, sparse = Sparse, meanshift= False)

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

                I_B_test, F_test, _, _ = utils.bf_from_theta(theta, coord[data.test_mask, :], nn, sparse=Sparse,
                                                              version='sparseB')
                F_test = F_test.detach().numpy()
                I_B_test = I_B_test.detach().numpy()
                FI_B_test = (I_B_test.T * np.sqrt(np.reciprocal(F_test))).T
                Y_test_decor = np.array(utils.decor_dense_np(Y_test, FI_B_test))
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                beta, theta_hat_linear = utils.BRISC_estimation(Y[~data.test_mask,], X_int[~data.test_mask,], coord[~data.test_mask,])
                def model_BRISC(X, edge_index = 0):
                    if isinstance(X, np.ndarray):
                        X = torch.from_numpy(X).float()
                    return(torch.matmul(X, torch.from_numpy(beta).float()))
                Est_MISE_BRISC = model_BRISC(X_MISE_int, edge_index).detach().numpy().reshape(-1)
                MISE_BRISC = np.append(MISE_BRISC, RMSE(Est_MISE_BRISC, Y_MISE_np.reshape(-1)))

                Pred_BRISC = utils.krig_pred(model_BRISC, X_int[~data.test_mask,], X_int[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_linear)
                RMSE_BRISC = np.append(RMSE_BRISC, RMSE(Pred_BRISC[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_BRISC = np.append(GMSE_BRISC, RMSE(np.array(utils.decor_dense_np(Pred_BRISC[0], FI_B_test)), Y_test_decor))
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
                Pred_GAM = model_GAM(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_GAM = np.append(RMSE_GAM, RMSE(Pred_GAM, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAM = np.append(GMSE_GAM, RMSE(np.array(utils.decor_dense_np(Pred_GAM, FI_B_test)), Y_test_decor))
                MISE_GAM = np.append(MISE_GAM, RMSE(Est_MISE_GAM, Y_MISE_np.reshape(-1)))

                Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_GAM = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_GAM = utils.krig_pred(model_GAM, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAM)
                ####################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                X_coord = np.concatenate((X, coord), axis=1)
                gam = utils_pygam.my_LinearGAM()
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
                GMSE_GAM_latlong = np.append(GMSE_GAM_latlong, RMSE(np.array(utils.decor_dense_np(Pred_GAM_latlong, FI_B_test)), Y_test_decor))
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
                    gam.my_fit(X[~data.test_mask,], sparse.csr_matrix(np.array(utils.decor_dense_np(Xspline.todense(), FI_B_GAM))),
                               np.array(utils.decor_dense_np(Y[~data.test_mask], FI_B_GAM)))
                del I_B_GAM, F_GAM, FI_B_GAM
                PI_GAMGLS = gam.confidence_intervals(X[data.test_mask])
                def model_GAMGLS(X, edge_index=0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    return (torch.from_numpy(gam.predict(X)).reshape(-1))
                Est_MISE_GAMGLS = model_GAMGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                Pred_GAMGLS = model_GAMGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_GAMGLS = np.append(RMSE_GAMGLS, RMSE(Pred_GAMGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_GAMGLS = np.append(GMSE_GAMGLS, RMSE(np.array(utils.decor_dense_np(Pred_GAMGLS, FI_B_test)), Y_test_decor))
                MISE_GAMGLS = np.append(MISE_GAMGLS, RMSE(Est_MISE_GAMGLS, Y_MISE_np.reshape(-1)))

                Y_hat = model_GAMGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_GAMGLS = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_GAMGLS = utils.krig_pred(model_GAMGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAMGLS)
                ########################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                n_tree = 60
                node_size = 20

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
                GMSE_RF = np.append(GMSE_RF, RMSE(np.array(utils.decor_dense_np(Pred_RF, FI_B_test)), Y_test_decor))

                Y_hat = model_RF(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_RF = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_RF = utils.krig_pred(model_RF, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RF)
                ########################################################################################################################
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                if ADDRFGLS:
                    robjects.globalenv['.Random.seed'] = 1
                    rfgls = utils.RFGLS_prediction(X[~data.test_mask,], Y[~data.test_mask], coord[~data.test_mask,],
                                                      n_tree=n_tree, node_size=node_size)
                    def model_RFGLS(X, edge_index = 0):
                        if torch.is_tensor(X):
                            X = X.detach().numpy()
                        X_r = utils.robjects.FloatVector(X.transpose().reshape(-1))
                        X_r = utils.robjects.r['matrix'](X_r, ncol=X.shape[1])
                        predict = utils.RFGLS.RFGLS_predict(rfgls, X_r)[1]
                        return(torch.from_numpy(np.array(predict)))
                    Est_MISE_RFGLS = model_RFGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                    Pred_RFGLS = model_RFGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                    RMSE_RFGLS = np.append(RMSE_RFGLS, RMSE(Pred_RFGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                    GMSE_RFGLS = np.append(GMSE_RFGLS, RMSE(np.array(utils.decor_dense_np(Pred_RFGLS, FI_B_test)), Y_test_decor))
                    MISE_RFGLS = np.append(MISE_RFGLS, RMSE(Est_MISE_RFGLS, Y_MISE_np.reshape(-1)))

                    Y_hat = model_RFGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                    residual = data.y - torch.from_numpy(Y_hat)
                    residual_train = residual[~data.test_mask]
                    residual_train = residual_train.detach().numpy()
                    _, theta_hat_RFGLS = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                    Pred_RFGLS = utils.krig_pred(model_RFGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                    coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RFGLS)
                    RMSE_RFGLS_krig = np.append(RMSE_RFGLS_krig,
                                                RMSE(Pred_RFGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
                    GMSE_RFGLS_krig = np.append(GMSE_RFGLS_krig, RMSE(np.array(utils.decor_dense_np(Pred_RFGLS[0], FI_B_test)), Y_test_decor))
                ######################################################################################################################
                torch.manual_seed(2023)
                model_NN = Netp(p, k)
                optimizer = torch.optim.Adam(model_NN.parameters(), lr=lr)
                patience_half = 10
                patience = 20

                _, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                                      patience = patience, patience_half = patience_half)
                Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NN = np.append(GMSE_NN, RMSE(np.array(utils.decor_dense_np(Pred_NN, FI_B_test)), Y_test_decor))
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

                _, _, model_NNlatlong = utils.train_gen_new(model_NNlatlong, optimizer, data_latlong, epoch_num=1000,
                                                               patience=patience, patience_half=patience_half)
                Pred_NNlatlong = model_NNlatlong(data_latlong.x[data_latlong.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_NNlatlong = np.append(RMSE_NNlatlong, RMSE(Pred_NNlatlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))
                GMSE_NNlatlong = np.append(GMSE_NNlatlong, RMSE(np.array(utils.decor_dense_np(Pred_NNlatlong, FI_B_test)), Y_test_decor))
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
                GMSE_NNDK = np.append(GMSE_NNDK, RMSE(np.array(utils.decor_dense_np(Pred_NNDK, FI_B_test)), Y_test_decor))
                ####################################################################################################################
                Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual = data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                theta_hat0 = theta_hat.copy()

                Pred_NN = utils.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
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

                Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
                Pred_NNGLS = utils.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_NNGLS)
                RMSE_NNGLS_krig = np.append(RMSE_NNGLS_krig,
                                            RMSE(Pred_NNGLS[0], Y_test)/RMSE(Y_test, np.mean(Y_test)))
                GMSE_NNGLS_krig = np.append(GMSE_NNGLS_krig, RMSE(np.array(utils.decor_dense_np(Pred_NNGLS[0], FI_B_test)), Y_test_decor))
                ####################################################################################################################
                df_PI1_temp = {'BRISC': int_coverage(Y_test, Pred_BRISC[1], Pred_BRISC[2]),
                              'GAM': int_coverage(Y_test, Pred_GAM[1], Pred_GAM[2]),
                              'GAM_latlong': int_coverage(Y_test, PI_GAM_latlong[:,0], PI_GAM_latlong[:,1]),
                              'GAMGLS': int_coverage(Y_test, Pred_GAMGLS[1], Pred_GAMGLS[2]),
                              'RF': int_coverage(Y_test, Pred_RF[1], Pred_RF[2]),
                              'NN': int_coverage(Y_test, Pred_NN[1], Pred_NN[2]),
                              'NNGLS': int_coverage(Y_test, Pred_NNGLS[1], Pred_NNGLS[2])}

                df_PI2_temp = {'BRISC': int_score(Y_test, Pred_BRISC[1], Pred_BRISC[2], 0.95),
                              'GAM': int_score(Y_test, Pred_GAM[1], Pred_GAM[2], 0.95),
                              'GAM_latlong': int_score(Y_test, PI_GAM_latlong[:, 0], PI_GAM_latlong[:, 1], 0.95),
                              'GAMGLS': int_score(Y_test, Pred_GAMGLS[1], Pred_GAMGLS[2], 0.95),
                              'RF': int_score(Y_test, Pred_RF[1], Pred_RF[2], 0.95),
                              'NN': int_score(Y_test, Pred_NN[1], Pred_NN[2], 0.95),
                              'NNGLS': int_score(Y_test, Pred_NNGLS[1], Pred_NNGLS[2], 0.95)}
                if ADDRFGLS:
                    df_PI1_temp['RFGLS'] = int_coverage(Y_test, Pred_RFGLS[1], Pred_RFGLS[2])
                    df_PI2_temp['RFGLS'] = int_score(Y_test, Pred_RFGLS[1], Pred_RFGLS[2], 0.95)
                ####################################################################################################################
                df_PI1 = df_PI1.append(df_PI1_temp, ignore_index=True)
                df_PI1.to_csv(".//simulation//compare//" + name + '_PI_cov.csv')
                df_PI2 = df_PI2.append(df_PI2_temp, ignore_index=True)
                df_PI2.to_csv(".//simulation//compare//" + name + '_PI_score.csv')
                df_MISE = pd.DataFrame(
                    {'BRISC': MISE_BRISC, 'GAM': MISE_GAM, 'GAMGLS': MISE_GAMGLS, 'RF': MISE_RF,
                     'NN': MISE_NN, 'NNGLS': MISE_NNGLS})
                df_RMSE = pd.DataFrame(
                    {'BRISC': RMSE_BRISC, 'GAM': RMSE_GAM, 'GAM_latlong': RMSE_GAM_latlong,
                     'GAMGLS': RMSE_GAMGLS,
                     'RF': RMSE_RF,
                     'NN_latlong': RMSE_NNlatlong, 'NNDK': RMSE_NNDK,
                     'NN': RMSE_NN, 'NNGLS_krig': RMSE_NNGLS_krig})
                df_GMSE = pd.DataFrame(
                    {'BRISC': GMSE_BRISC, 'GAM': GMSE_GAM, 'GAM_latlong': GMSE_GAM_latlong,
                     'GAMGLS': GMSE_GAMGLS,
                     'RF': GMSE_RF,
                     'NN_latlong': GMSE_NNlatlong, 'NNDK': GMSE_NNDK,
                     'NN': GMSE_NN, 'NNGLS_krig': GMSE_NNGLS_krig})
                if ADDRFGLS:
                    df_MISE['RFGLS'] = MISE_RFGLS
                    df_RMSE['RFGLS'] = RMSE_RFGLS
                    df_RMSE['RFGLS_krig'] = RMSE_RFGLS_krig
                    df_GMSE['RFGLS'] = GMSE_RFGLS
                    df_GMSE['RFGLS_krig'] = GMSE_RFGLS_krig
                df_MISE.to_csv(".//simulation//compare//" + name + '_MISE.csv')
                df_RMSE.to_csv(".//simulation//compare//" + name + '_RMSE.csv')
                df_GMSE.to_csv(".//simulation//compare//" + name + '_GMSE.csv')

