#### This file produces simulation results for large sample behavior in section S4.7, Figure S16(a, b).

import utils

import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import lhsmdu
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
ordered = True
max_epoch = 1000

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

name = 'running_time_details'
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
    b = np.sqrt(n / 1000)
    n_train = int(n / 2)
    nn = 20
    batch_size = int(n / 20)
    ADDRFGLS = False
    Sparse = False
    if n_train <= 1000: ADDRFGLS = True
    if n > 10000: Sparse = True
    t_start = time.time()
    torch.manual_seed(2023 + rand)
    np.random.seed(2023 + rand)
    X, Y, I_B, F_diag, _, coord, _, _ = utils.Simulate(n, p, funXY, nn, theta, method=method, a=0,
                                                       b=b, sparse=Sparse)
    edge_index = 0

    torch.manual_seed(2023 + rand)
    data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord=coord)
    transform = T.RandomNodeSplit(num_train_per_class=int(0.3 * n_train), num_val=int(0.2 * n_train),
                                  num_test=int(n - n_train))
    data = transform(data)
    data.n = data.x.shape[0]

    Y_test = Y[data.test_mask]

    torch.manual_seed(2023 + rand)
    data = utils.batch_gen(data, batch_size)
    t_simulate = time.time() - t_start
    ####################################################################################################################
    t_start = time.time()
    torch.manual_seed(2023)
    model_NN = Netp(p, k, q)
    optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
    patience_half = 10
    patience = 20

    _, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1,
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
    beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
    theta_hat0 = theta_hat

    Pred_NN = utils.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                 coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
    t_krig = time.time() - t_start
    ####################################################################################################################
    t_start = time.time()
    torch.manual_seed(2023)
    model_NNGLS = Netp(p, k, q)
    optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
    patience_half = 10
    patience = 20

    _, _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1, theta_hat0,
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
    lr_scheduler = utils.LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = utils.EarlyStopping(patience=patience, min_delta=0.00001)
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
        decor = utils.decor_sparse_SparseB
        sparse_decor_fun = utils.sparse_decor_sparseB
        FI_B = sparse_decor_fun(coord, nn, theta_hat)
        if sparseB: FI_B = FI_B.to_tensor()
    else:
        decor = utils.decor_dense
        I_B, F_diag, rank, _ = utils.bf_from_theta(theta_hat, coord, nn, sparse=sparse)
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
    df_t.to_csv(".//simulation//large//" + name + ".csv")
    df_NNGLS_t.to_csv(".//simulation//large//" + name + "_NNGLS.csv")



