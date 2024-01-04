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
from pygam import LinearGAM, s, f
from sklearn.ensemble import RandomForestRegressor
import lhsmdu
import copy
import time


def RMSE(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return (np.sqrt(np.sum(np.square(x - y)) / n))


def order(X, Y, coord):
    s_sum = coord[:, 0] + coord[:, 1]
    order = s_sum.argsort()
    X_new = X[order, :]
    Y_new = Y[order]
    coord_new = coord[order, :]
    return X_new, Y_new, coord_new


class Netp(torch.nn.Module):
    def __init__(self, p, k=50, q=1):
        super(Netp, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, q)
        # self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x, edge_index=0):
        x = torch.sigmoid(self.l1(x))
        # x = torch.sigmoid(self.l2(x))
        return self.l2(x)


x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
var1 = multivariate_normal(mean=[0.25, 0.25], cov=[[0.01, 0], [0, 0.01]])
var2 = multivariate_normal(mean=[0.6, 0.9], cov=[[0.01, 0], [0, 0.01]])
mean = np.mean(var1.pdf(pos) + var2.pdf(pos))
var = np.var(var1.pdf(pos) + var2.pdf(pos))


def corerr_gen(pos):  # must be designed for unit square
    n = pos.shape[0]
    return ((var1.pdf(pos) + var2.pdf(pos) - mean) * np.sqrt(sigma) / np.sqrt(var) + np.sqrt(
        sigma * tau) * np.random.randn(n))


def partition(list_in, n):
    idx = torch.randperm(list_in.shape[0])
    list_in = list_in[idx]
    return [torch.sort(list_in[i::n])[0] for i in range(n)]

'''
def decor(y, I_B_local, F_diag_local):
    y_decor = (np.array(np.matmul(I_B_local, y)).T * np.sqrt(np.reciprocal(F_diag_local))).T
    return (y_decor)


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
            y_decor[i, :] = np.dot(FI_B_local.B[i, :], y[FI_B_local.Ind_list[i, :], :])
    elif np.ndim(y) == 1:
        y_decor = np.zeros(n)
        for i in range(n):
            y_decor[i] = np.dot(FI_B_local.B[i, :], y[FI_B_local.Ind_list[i, :]])
    return (y_decor)
'''



def batch_gen(data, k):
    for mask in ['train_mask', 'val_mask', 'test_mask']:
        data[mask + '_batch'] = partition(torch.tensor(range(data.n))[data[mask]],
                                          int(torch.sum(data[mask]) / k))
    return (data)


def int_score(x, u, l, coverage=0.95):
    alpha = 1 - coverage
    score = u - l + 2 * ((l - x) * (l > x) + (x - u) * (x > u)) / alpha
    return (np.mean(score))


def int_coverage(x, u, l):
    score = np.logical_and(x >= l, x <= u)
    return (np.mean(score))


sigma = 1
phi = 3
tau = 0.01
method = '0'
theta = [sigma, phi / np.sqrt(2), tau]
p = 5;
funXY = utils_NN.f5
k = 50
q = 1
lr = 0.01
b = 10
ordered = False

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

name = 'large_sparseB_explore_comp_NN'
# name = 'large' + 'phi%i' % phi + "sig%i" % (theta[0]) + "tau%i" % (int(100 * tau)) + 'mtd' + method
MISE_BRISC = np.empty(0)
MISE_GAM = np.empty(0)
MISE_GAMGLS = np.empty(0)
MISE_RF = np.empty(0)
MISE_RFGLS = np.empty(0)
MISE_NN = np.empty(0)
MISE_NNGLS = np.empty(0)

df_t = pd.DataFrame(columns=['NN', 'NNGLS', 'Sim', 'Krig', 'Size'])

df_NNGLS_t = pd.DataFrame(columns=['model', 'loss', 'backward', 'optimize', 'Size'])

rand = 1
for n in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]:
    n = int(n)
    n_train = int(n / 2)
    nn = 20
    batch_size = int(n / 20)
    ADDRFGLS = False
    Sparse = False
    if n_train <= 600: ADDRFGLS = True
    if n > 10000: Sparse = True
    t_start = time.time()
    torch.manual_seed(2023 + rand)
    np.random.seed(2023 + rand)
    if method == '2':
        X, Y, _, coord, _ = utils_NN.Simulate_mis(n, p, funXY, nn, theta, corerr_gen, a=0, b=b)
    else:
        X, Y, I_B, F_diag, _, coord, _, _ = utils_NN.Simulate_NNGP(n, p, funXY, nn, theta, method=method, a=0,
                                                                   b=b, sparse=Sparse)

    if ordered:
        X, Y, coord = order(X, Y, coord)

    '''
    neigh = NearestNeighbors(n_neighbors=nn)
    neigh.fit(coord)

    A = neigh.kneighbors_graph(coord)
    A.toarray()
    edge_index = torch.from_numpy(np.stack(A.nonzero()))
    '''
    edge_index = 0

    torch.manual_seed(2023 + rand)
    data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord=coord)
    transform = T.RandomNodeSplit(num_train_per_class=int(0.3 * n_train), num_val=int(0.2 * n_train),
                                  num_test=int(n - n_train))
    data = transform(data)
    data.n = data.x.shape[0]

    Y_test = Y[data.test_mask]

    torch.manual_seed(2023 + rand)
    data = batch_gen(data, batch_size)
    t_simulate = time.time() - t_start
    ####################################################################################################################
    t_start = time.time()
    torch.manual_seed(2023)
    model_NN = Netp(p, k, q)
    optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
    patience_half = 10
    patience = 20

    _, _, model_NN = utils_NN.train_gen_new(model_NN, optimizer, data, epoch_num=1,
                                            patience=patience, patience_half=patience_half)
    t_NN = time.time() - t_start
    Est_MISE_NN = model_NN(X_MISE, edge_index).detach().numpy().reshape(-1)
    MISE_NN = np.append(MISE_NN, RMSE(Est_MISE_NN, Y_MISE_np.reshape(-1)))
    ####################################################################################################################
    t_start = time.time()
    Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
    residual = torch.from_numpy(Y_hat) - data.y
    residual_train = residual[~data.test_mask]
    residual_train = residual_train.detach().numpy()
    beta_hat, theta_hat = utils_NN.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
    theta_hat0 = theta_hat

    Pred_NN = utils_NN.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                 coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
    t_krig = time.time() - t_start
    ####################################################################################################################
    t_start = time.time()
    torch.manual_seed(2023)
    model_NNGLS = Netp(p, k, q)
    optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
    patience_half = 10
    patience = 20

    _, _, model_NNGLS = utils_NN.train_decor_new(model_NNGLS, optimizer, data, 1, theta_hat0,
                                                 sparse=Sparse, sparseB=True,
                                                 Update=False, patience=patience, patience_half=patience_half)
    t_NNGLS = time.time() - t_start
    Est_MISE_NNGLS = model_NNGLS(X_MISE, edge_index).detach().numpy().reshape(-1)
    MISE_NNGLS = np.append(MISE_NNGLS, RMSE(Est_MISE_NNGLS, Y_MISE_np.reshape(-1)))

    loss_fn = torch.nn.MSELoss(reduction='mean')
    sparse = Sparse
    sparseB = True

    ####################################################################################################################
    def train_decor(model, data, FI_B, idx, decor):
        model.train()
        optimizer.zero_grad()
        t_start = time.time()
        out = model(data.x, data.edge_index)
        out = torch.reshape(out, (-1,))
        t_model = time.time() - t_start
        t_start = time.time()
        loss = loss_fn(decor(out,  FI_B, idx), decor(data.y,  FI_B, idx))
        t_loss = time.time() - t_start
        t_start = time.time()
        loss.backward()
        t_backward = time.time() - t_start
        t_start = time.time()
        optimizer.step()
        t_optimize = time.time() - t_start
        return t_model, t_loss, t_backward, t_optimize
    @torch.no_grad()
    def test(model, data):
        model.eval()
        pred = model(data.x, data.edge_index)
        pred = torch.reshape(pred, (-1,))
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(np.square(pred[mask] - data.y[mask]).mean() / int(mask.sum()))
        return accs
    lr_scheduler = utils_NN.LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = utils_NN.EarlyStopping(patience=patience, min_delta=0.00001)
    losses = []
    val_losses = []
    best_val_loss = final_test_loss = 100

    n = data.n
    coord = data.coord
    Y_local = data.y
    X_local = data.x
    theta_hat = np.array(theta_hat0.copy())

    t_start = time.time()
    if sparse:
        decor = utils_NN.decor_sparse_SparseB
        sparse_decor_fun = utils_NN.sparse_decor_sparseB
        FI_B = sparse_decor_fun(coord, nn, theta_hat)
        if sparseB: FI_B = FI_B.to_tensor()
    else:
        decor = utils_NN.decor_dense
        I_B, F_diag, rank, _ = utils_NN.bf_from_theta(theta_hat, coord, nn, sparse=sparse)
        FI_B = (I_B.T*torch.sqrt(torch.reciprocal(F_diag))).T

    for epoch in range(1):
        for idx in data.train_mask_batch:
            t_model, t_loss, t_backward, t_optimize = train_decor(model_NNGLS, data, FI_B, idx, decor)
            df_NNGLS_t_temp = {'model': t_model, 'loss': t_loss,
                         'backward': t_backward, 'optimize': t_optimize, 'Size': n}
            df_NNGLS_t = df_NNGLS_t.append(df_NNGLS_t_temp, ignore_index=True)
        train_loss, val_loss, tmp_test_loss = test(model_NNGLS, data)

    df_t_temp = {'NN': t_NN, 'NNGLS': t_NNGLS,
                 'Sim': t_simulate, 'Krig': t_krig, 'Size': n}

    df_t = df_t.append(df_t_temp, ignore_index=True)
    df_t.to_csv(".//simulation//large//test_t.csv")
    df_NNGLS_t.to_csv(".//simulation//large//test_NNGLS_t.csv")



