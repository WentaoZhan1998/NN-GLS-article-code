#### This file produces simulation results of confidence intervals

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

def int_width(x, u, l):
    return (u - l)

def int_coverage(x, u, l):
    score = np.logical_and(x>=l, x<=u)
    return (np.mean(score))

fun = 'sin'
p = 1; funXY = utils.f1; Netp = utils.Netp_sig
#fun = 'friedman'
#p = 1; funXY = utils.f1; Netp = utils.Netp_sig

for sigma in [1, 5]:
    for phi in [1, 3, 6]:
        for tau in [0.01, 0.25]:
            method = '0'
            theta = [sigma, phi / np.sqrt(2), tau]
            k = 50
            q = 1
            lr = 0.01
            b = 10
            batch_size = 50
            resample = 'decor'
            ordered = True
            Sparse = False
            regenerate = True #IF True, regenearte error from the normal distribution with empirical variance, else, resample from the
            # (decorrelated) residual

            name = 'CI'

            n = 1000
            n_train = 1000
            n_sub = 0.5
            n_rep = 100
            nn = 20

            N = 1000
            n_small = int(N / 100)
            np.random.seed(2021)
            X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
            for i in range(99):
                temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
                X_MISE = np.concatenate((X_MISE, temp))

            Y_MISE = funXY(X_MISE).reshape(-1)
            X_MISE = torch.from_numpy(X_MISE).float()

            NN_mat = np.zeros(shape=(n_rep, N))
            NNGLS_mat = np.zeros(shape=(n_rep, N))

            df_CI1 = pd.DataFrame(columns=['NN', 'NNGLS'])
            df_CI2 = pd.DataFrame(columns=['NN', 'NNGLS'])
            df_CI3 = pd.DataFrame(columns=['NN', 'NNGLS'])

            for rand in range(1, 101):
                torch.manual_seed(2023 + rand)
                np.random.seed(2023 + rand)
                X, Y, I_B, F_diag, rank, coord, cov, corerr = utils.Simulate(n, p, funXY, nn, theta, method=method, a=0,
                                                                             b=b, sparse=Sparse, meanshift=True)

                neigh = NearestNeighbors(n_neighbors=nn)
                neigh.fit(coord)

                A = neigh.kneighbors_graph(coord)
                A.toarray()
                edge_index = torch.from_numpy(np.stack(A.nonzero()))

                torch.manual_seed(2023 + rand)
                data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord=coord)
                transform = T.RandomNodeSplit(num_train_per_class=int(0.3 * n_train), num_val=int(0.2 * n_train),
                                              num_test=int(n - n_train))
                data = transform(data)
                data.n = data.x.shape[0]

                torch.manual_seed(2023 + rand)
                data = utils.batch_gen(data, batch_size)
                ######################################################################################################################
                torch.manual_seed(2023)
                model_NN = Netp(p, k, q)
                optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
                patience_half = 10
                patience = 20

                _, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                                        patience=patience, patience_half=patience_half)
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
                optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=lr)
                patience_half = 10
                patience = 20

                _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                                             Update=True, patience=patience, patience_half=patience_half)
                Y_hat_NNGLS = model_NNGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual_NNGLS = data.y - torch.from_numpy(Y_hat_NNGLS)
                residual_NNGLS_np = residual_NNGLS.detach().numpy()
                beta_hat_NNGLS, theta_hat_NNGLS = utils.BRISC_estimation(residual_NNGLS_np, X[~data.test_mask,],
                                                                            coord[~data.test_mask,])

                Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
                Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                ####################################################################################################################
                I_B_temp, F_diag_temp, _, _ = utils.bf_from_theta(theta_hat_NNGLS, coord, nn, sparse=Sparse)
                if not Sparse:
                    I_B_inv_temp = torch.inverse(I_B_temp)
                sigma_sq_hat, phi_hat, tau_hat = theta_hat_NNGLS
                tau_sq_hat = tau_hat * sigma_sq_hat
                dist = utils.distance(coord, coord)
                cov_hat = sigma_sq_hat * np.exp(-phi_hat * dist) + tau_sq_hat * np.eye(n)
                for rep in range(n_rep):
                    torch.manual_seed(2023 + rep * rand)
                    np.random.seed(2023 + rep * rand)
                    id = range(n)
                    X_sub = X
                    if regenerate:
                        Y_sub = Y_hat + np.std(residual_train) * np.random.randn(n)
                    else:
                        Y_sub = Y_hat + np.random.choice(residual_train, n, replace=False)
                    if Sparse:
                        Y_sub_NNGLS = Y_hat_NNGLS + \
                                      utils.resample_fun_sparseB(residual_NNGLS, coord,
                                                                    nn, theta_hat_NNGLS, regenerate=regenerate)
                    else:
                        Y_sub_NNGLS = Y_hat_NNGLS + utils.resample_fun(residual_NNGLS, I_B_temp, I_B_inv_temp,
                                                                          F_diag_temp,
                                                                          regenerate=regenerate).detach().numpy()
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
                                    y=torch.from_numpy(Y_sub).float(), coord=coord_sub)
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
                    data_sub = utils.batch_gen(data_sub, int(batch_size * n_sub))
                    torch.manual_seed(2023 + rand)
                    data_sub_NNGLS = utils.batch_gen(data_sub_NNGLS, int(batch_size * n_sub))
                    ####################################################################################
                    torch.manual_seed(2023 + rand)
                    model0 = Netp(p, k, q)
                    optimizer = torch.optim.Adam(model0.parameters(), lr=lr)
                    patience_half = 10
                    patience = 20

                    _, _, model0 = utils.train_gen_new(model0, optimizer, data_sub, epoch_num=1000,
                                                          patience=patience, patience_half=patience_half)

                    NN_mat[rep, :] = model0(X_MISE, edge_index_sub).detach().numpy().reshape(-1)
                    ###################################################################################
                    torch.manual_seed(2023 + rand)
                    model1 = Netp(p, k, q)
                    optimizer = torch.optim.Adam(model1.parameters(), lr=lr)
                    patience_half = 10
                    patience = 20

                    _, _, model1 = utils.train_decor_new(model1, optimizer, data_sub_NNGLS, 1000, theta_hat0,
                                                         sparse=Sparse, BF=[I_B_sub, F_diag_sub],
                                                         Update=False, patience=patience, patience_half=patience_half)

                    NNGLS_mat[rep, :] = model1(X_MISE, edge_index_sub).detach().numpy().reshape(-1)

                CI_U_NN, CI_L_NN = np.quantile(NN_mat, 0.975, axis=0), np.quantile(NN_mat, 0.025, axis=0)
                CI_U_NNGLS, CI_L_NNGLS = np.quantile(NNGLS_mat, 0.975, axis=0), np.quantile(NNGLS_mat, 0.025, axis=0)

                df_CI1_temp = {'NN': int_coverage(Y_MISE, CI_U_NN, CI_L_NN),
                               'NNGLS': int_coverage(Y_MISE, CI_U_NNGLS, CI_L_NNGLS)}
                df_CI2_temp = {'NN': int_score(Y_MISE, CI_U_NN, CI_L_NN, 0.95),
                               'NNGLS': int_score(Y_MISE, CI_U_NNGLS, CI_L_NNGLS, 0.95)}
                df_CI3_temp = {'NN': int_width(Y_MISE, CI_U_NN, CI_L_NN),
                               'NNGLS': int_width(Y_MISE, CI_U_NNGLS, CI_L_NNGLS)}

                df_CI1 = df_CI1.append(df_CI1_temp, ignore_index=True)
                df_CI1.to_csv(".//simulation//CI//" + name + '_CI_cov.csv')
                df_CI2 = df_CI2.append(df_CI2_temp, ignore_index=True)
                df_CI2.to_csv(".//simulation//CI//" + name + '_CI_score.csv')
                df_CI3 = df_CI3.append(df_CI3_temp, ignore_index=True)
                df_CI3.to_csv(".//simulation//CI//" + name + '_CI_width.csv')
