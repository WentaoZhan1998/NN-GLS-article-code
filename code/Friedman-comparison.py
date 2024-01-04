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

class Netp(torch.nn.Module):
    def __init__(self, p, k = 50, q = 1):
        super(Netp, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)
        #self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x, edge_index = 0):
        x = torch.sigmoid(self.l1(x))
        #x = torch.sigmoid(self.l2(x))
        return self.l2(x)

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

sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
p = 5
k = 50
q = 1
lr = 0.01
b = 10
ordered = False

n = 1000
n_train = 500
nn = 20
batch_size = int(n/20)
ADDRFGLS = True
if n_train <= 600: ADDRFGLS = True
Sparse = False
if n > 10000: Sparse = True

N = 1000
n_small = int(N / 100)
np.random.seed(2021)
X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
for i in range(99):
    temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
    X_MISE = np.concatenate((X_MISE, temp))

X_MISE = torch.from_numpy(X_MISE).float()

name = 'phi%i' % phi + "sig%i" % (theta[0]) + "tau%i" % (int(100 * tau)) + 'mtd' + method
#name = 'test'
rho_vec = np.empty(0)
MISE_BRISC = np.empty(0)
RMSE_BRISC = np.empty(0)
MISE_GAM = np.empty(0)
RMSE_GAM = np.empty(0)
RMSE_GAM_krig = np.empty(0)
MISE_GAMGLS = np.empty(0)
RMSE_GAMGLS = np.empty(0)
RMSE_GAMGLS_krig = np.empty(0)
MISE_NN = np.empty(0)
RMSE_NN = np.empty(0)
RMSE_NN_krig = np.empty(0)
MISE_NNGLS = np.empty(0)
RMSE_NNGLS_krig = np.empty(0)

df_CI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
df_CI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'NN', 'NNGLS'])
df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

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
        if method == '2':
            X, Y, rank, coord, corerr = utils_NN.Simulate_mis(n, p, funXY, nn, theta, corerr_gen, a=0, b=b)
        else:
            X, Y, I_B, F_diag, rank, coord, cov, corerr = utils_NN.Simulate_NNGP(n, p, funXY, nn, theta, method=method, a=0,
                                                                                 b=b, sparse = Sparse)

        if ordered:
            X, Y, coord = order(X, Y, coord)
            I_B, F_diag, rank, cov = utils_NN.bf_from_theta(theta, coord, nn, sparse=Sparse)

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
        data = batch_gen(data, batch_size)
        ####################################################################################################################
        torch.manual_seed(2023 + rand)
        np.random.seed(2023 + rand)
        beta, theta_hat_linear = utils_NN.BRISC_estimation(Y[~data.test_mask,], X[~data.test_mask,], coord[~data.test_mask,])
        def model_BRISC(X, edge_index = 0):
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            return(torch.matmul(X, torch.from_numpy(beta).float()))
        Est_MISE_BRISC = model_BRISC(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_BRISC = np.append(MISE_BRISC, RMSE(Est_MISE_BRISC, Y_MISE_np.reshape(-1)))

        Pred_BRISC = utils_NN.krig_pred(model_BRISC, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat_linear)
        RMSE_BRISC = np.append(RMSE_BRISC,
                                    RMSE(Pred_BRISC[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
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
        Est_MISE_GAM = model_GAM(X_MISE, edge_index).detach().numpy().reshape(-1)
        Pred_GAM = model_GAM(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_GAM = np.append(RMSE_GAM, RMSE(Pred_GAM, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        MISE_GAM = np.append(MISE_GAM, RMSE(Est_MISE_GAM, Y_MISE_np.reshape(-1)))

        Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = torch.from_numpy(Y_hat) - data.y
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        _, theta_hat_GAM = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        Pred_GAM = utils_NN.krig_pred(model_GAM, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAM)
        RMSE_GAM_krig = np.append(RMSE_GAM_krig,
                                    RMSE(Pred_GAM[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
        #from pygam import terms
        from mygam import my_LinearGAM
        #rom group_lasso import GroupLasso
        #from scipy import sparse
        #from sklearn.linear_model import LinearRegression, Lasso
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
        Est_MISE_GAMGLS = model_GAMGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        Pred_GAMGLS = model_GAMGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_GAMGLS = np.append(RMSE_GAMGLS, RMSE(Pred_GAMGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        MISE_GAMGLS = np.append(MISE_GAMGLS, RMSE(Est_MISE_GAMGLS, Y_MISE_np.reshape(-1)))

        Y_hat = model_GAMGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = torch.from_numpy(Y_hat) - data.y
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        _, theta_hat_GAMGLS = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        Pred_GAMGLS = utils_NN.krig_pred(model_GAMGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAMGLS)
        RMSE_GAMGLS_krig = np.append(RMSE_GAMGLS_krig,
                                    RMSE(Pred_GAMGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
        ######################################################################################################################
        torch.manual_seed(2023)
        model_NN = Netp(p, k, q)
        optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
        patience_half = 10
        patience = 20

        _, _, model_NN = utils_NN.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                              patience = patience, patience_half = patience_half)
        Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
        RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))
        Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NN = np.append(MISE_NN, RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
        ####################################################################################################################
        Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
        residual = data.y - torch.from_numpy(Y_hat)
        residual_train = residual[~data.test_mask]
        residual_train = residual_train.detach().numpy()
        beta_hat, theta_hat = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
        theta_hat0 = theta_hat

        Pred_NN = utils_NN.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
        RMSE_NN_krig = np.append(RMSE_NN_krig,
                                    RMSE(Pred_NN[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))
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

        Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
        MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))
        Pred_NNGLS = utils_NN.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                        coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
        RMSE_NNGLS_krig = np.append(RMSE_NNGLS_krig,
                                    RMSE(Pred_NNGLS[0], Y_test)/RMSE(Y_test, np.mean(Y_test)))
        ####################################################################################################################
        rho_vec = np.append(rho_vec, rho)

    df_MISE = pd.DataFrame(
        {'BRISC': MISE_BRISC, 'GAM': MISE_GAM, 'GAMGLS': MISE_GAMGLS,
         'NN': MISE_NN, 'NNGLS': MISE_NNGLS, 'rho': rho_vec})
    df_RMSE = pd.DataFrame(
        {'BRISC': RMSE_BRISC, 'GAM': RMSE_GAM, 'GAM_krig': RMSE_GAM_krig,
         'GAMGLS': RMSE_GAMGLS, 'GAMGLS_krig': RMSE_GAMGLS_krig,
         'NN': RMSE_NN, 'NN_krig': RMSE_NN_krig, 'NNGLS_krig': RMSE_NNGLS_krig, 'rho': rho_vec})
    df_MISE.to_csv(".//simulation//friedman//" + name + '_MISE.csv')
    df_RMSE.to_csv(".//simulation//friedman//" + name + '_RMSE.csv')