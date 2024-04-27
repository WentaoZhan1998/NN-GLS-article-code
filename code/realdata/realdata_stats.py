#### This file check the assumptions on the real data. (Section 5.4, Figure S31)

import os
os.environ['R_HOME'] = '/users/wzhan/anaconda3/envs/torch_geom/lib/R'
import utils
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skgstat as skg

def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

def block_rand(n, k):
    lx = np.empty(0)
    ly = np.empty(0)
    for i in range(k):
        if i == 0:
            ix = np.random.choice(range(n), 1)
            iy = np.random.choice(range(n), 1)
        else:
            ix = np.random.choice(np.delete(range(n), lx),1)
            iy = np.random.choice(np.delete(range(n), ly),1)
        lx = np.append(lx, ix).astype(int)
        ly = np.append(ly, iy).astype(int)
    return lx, ly

name = '0605'
df1 = pd.read_csv('covariate' + name + '.csv')
df2 = pd.read_csv('pm25_' + name + '.csv')
split = 6

### Data process #######################################################################################################
covariates = df1.values[:,3:]
aqs_lonlat=df2.values[:,[1,2]]

from scipy import spatial
near = df1.values[:,[1,2]]
tree = spatial.KDTree(list(zip(near[:,0].ravel(), near[:,1].ravel())))
tree.data
idx = tree.query(aqs_lonlat)[1]
df2_new = df2.assign(neighbor = idx)
df_pm25 = df2_new.groupby('neighbor')['PM25'].mean()
df_pm25_class = pd.cut(df_pm25,bins=[0,12.1,35.5],labels=["0","1"])
idx_new = df_pm25.index.values

pm25 = df_pm25.values
pm25_class = np.array(df_pm25_class.values)
z = pm25[:,None]
z_class = pm25_class[:,None]

lon = df1.values[:,1]
lat = df1.values[:,2]
normalized_lon = (lon-min(lon))/(max(lon)-min(lon))
normalized_lat = (lat-min(lat))/(max(lat)-min(lat))

s_obs = np.vstack((normalized_lon[idx_new],normalized_lat[idx_new])).T
X = covariates[idx_new,:]
normalized_X = X
for i in range(X.shape[1]):
    normalized_X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))

X = normalized_X
Y = z.reshape(-1)
coord = s_obs
b = 1
n = coord.shape[0]
p = X.shape[1]
####################################################################################################################
Netp = utils.Netp_sig

k = 50
q = 1
lr = 0.01
n_train = int(n*1)
batch_size = 50
nn = 10
ADDRFGLS = True
Sparse = False
if n > 10000: Sparse = True

neigh = NearestNeighbors(n_neighbors=nn)
neigh.fit(coord)

A = neigh.kneighbors_graph(coord)
A.toarray()
edge_index = torch.from_numpy(np.stack(A.nonzero()))

df_P = pd.DataFrame(columns=['ind', 'p-value'])

for rand in range(1):
    torch.manual_seed(2023+rand)
    np.random.seed(2023+rand)
    data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord=coord)
    data.n = data.x.shape[0]
    if split == 0:
        transform = T.RandomNodeSplit(num_train_per_class=int(0.3*n_train), num_val=int(0.2*n_train), num_test=int(n-n_train))
        data = transform(data)
    elif split == 1:
        test_mask = coord[:,0]> np.quantile(coord[:,0], 0.75)
        data.test_mask = torch.tensor(test_mask)
        id_train = np.random.choice(np.where(~test_mask)[0], int(0.2 * n_train), replace=False)
        train_mask = ~test_mask
        train_mask[id_train] = False
        val_mask = ~np.logical_or(train_mask, test_mask)
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)
    else:
        n_temp = split
        k_temp = split
        lx, ly = block_rand(n_temp, k_temp)
        xspc = np.linspace(0, b, n_temp + 1)
        yspc = np.linspace(0, b, n_temp + 1)
        test_mask = np.zeros(n, dtype=bool)
        for i in range(k_temp):
            mask_temp = np.logical_and((coord[:, 0] > xspc[lx[i]]) * (coord[:, 0] <= xspc[lx[i] + 1]),
                                       (coord[:, 1] > yspc[ly[i]]) * (coord[:, 1] <= yspc[ly[i] + 1]))
            test_mask = np.logical_or(test_mask, mask_temp)
        data.test_mask = torch.tensor(test_mask)
        id_train = np.random.choice(np.where(~test_mask)[0], int(0.2 * n_train), replace=False)
        train_mask = ~test_mask
        train_mask[id_train] = False
        val_mask = ~np.logical_or(train_mask, test_mask)
        data.train_mask = torch.tensor(train_mask)
        data.val_mask = torch.tensor(val_mask)

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
    Pred_NN = model_NN(data.x[data.test_mask,], edge_index).detach().numpy().reshape(-1)
    RMSE_NN = np.append(RMSE_NN, RMSE(Pred_NN, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*
    ####################################################################################
    Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
    residual =data.y - torch.from_numpy(Y_hat)
    residual_train = residual[~data.test_mask]
    residual_train = residual_train.detach().numpy()
    beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
    theta_hat0 = theta_hat
    ###################################################################################
    torch.manual_seed(2023)
    model_NNGLS = Netp(p, k, q)
    optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
    patience_half = 10
    patience = 20

    theta_hat, _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                                  Update=True, patience=patience, patience_half=patience_half,
                                                  Update_method='optimization', Update_init=20, Update_step=20,
                                                  Update_bound=100)

    plt.clf()
    V = skg.Variogram(coord, data.y, n_lags = 25, maxlag=1.0)
    V.plot()
    plt.savefig(".//simulation//realdata//Variog" + name +  "_Y.png")

    Pred_NNGLS_full = utils.krig_pred(model_NNGLS, X[~data.test_mask,], X, Y[~data.test_mask],
                                         coord[~data.test_mask,], coord, theta_hat0)
    Error = Y - Pred_NNGLS_full[0]
    plt.clf()
    plt.hist(Error, 40)
    plt.savefig(".//simulation//realdata//Hist" + name + ".png")
    df_Error = pd.DataFrame({'err': Error})
    df_Error.to_csv(".//simulation//realdata//Hist" + name + ".csv")


