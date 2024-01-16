#### This file produces simulation results for the comparison between NN-GLS and NN-GLS-oracle

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

fun = 'sin'
p = 1; funXY = utils.f1; Netp = utils.Netp_sig

for sigma in [1, 5]:
    for phi in [1, 3, 6]:
        for tau in [0.01, 0.1, 0.25]:
            method = '0' #### '0' is the normal case, '1' is the first misspecification case, and '2' is the second one
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

            MISE_NN = np.empty(0)
            MISE_NNGLS = np.empty(0)
            MISE_NNGLS_oracle = np.empty(0)

            df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
            df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

            for rand in range(1, 100+1):
                torch.manual_seed(2023+rand)
                np.random.seed(2023+rand)
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
                ####################################################################################################################
                torch.manual_seed(2023)
                model_NN = Netp(p, k)
                optimizer = torch.optim.Adam(model_NN.parameters(), lr=lr)
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
                theta_hat0 = theta_hat.copy()
                ###################################################################################
                torch.manual_seed(2023)
                model_NNGLS = Netp(p, k, q)
                optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=lr)
                patience_half = 10
                patience = 20

                _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse = Sparse,
                                                        Update=True, patience=patience, patience_half=patience_half)
                Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
                ###################################################################################
                torch.manual_seed(2023)
                model_NNGLS_oracle = Netp(p, k, q)
                optimizer = torch.optim.Adam(model_NNGLS_oracle.parameters(), lr=lr)
                patience_half = 10
                patience = 20

                _, _, model_NNGLS_oracle = utils.train_decor_new(model_NNGLS_oracle, optimizer, data, 1000, theta, sparse = Sparse,
                                                        Update=True, patience=patience, patience_half=patience_half)
                Est_MISE_NNGLS_oracle = model_NNGLS_oracle(X_MISE, edge_index).detach().numpy().reshape(-1)
                MISE_NNGLS_oracle = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
                ####################################################################################################################
                df_MISE = pd.DataFrame(
                    {'NN': MISE_NN, 'NNGLS': MISE_NNGLS, 'NNGLS_oracle': MISE_NNGLS_oracle})
                df_MISE.to_csv(".//simulation//compare//" + str(p) + "dim//" + name + '_MISE.csv')

