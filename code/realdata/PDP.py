#### This file produces PDP plots for the data June 18th 2022 with random training-testing split.

import utils
import utils_PDP

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

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

name = '0618' #### for other dates "0618" "0704" corresponding to 2022.06.18 and 2022.07.04
split = 6

### Data process #######################################################################################################
df1 = pd.read_csv('.//data//data_by_date//covariate' + name + '.csv')
#df2 = pd.read_csv('pm25_0628.csv')
df2 = pd.read_csv('.//data//data_by_date//pm25_' + name + '.csv')

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
X_int = np.concatenate((X, np.repeat(1, n).reshape(n, 1)), axis=1)
####################################################################################################################
Netp = utils.Netp_sig

variable_names = ['Precipitation accumulation', 'Air temperature', 'Pressure', 'Relative humidity', 'U-wind', 'V-wind']

k = 50
q = 1
lr = 0.01
n_train = int(n*0.8)
batch_size = 50
nn = 10
ADDRFGLS = True
Sparse = False

neigh = NearestNeighbors(n_neighbors=nn)
neigh.fit(coord)

A = neigh.kneighbors_graph(coord)
A.toarray()
edge_index = torch.from_numpy(np.stack(A.nonzero()))

######################################################################################################################
rand = 1
torch.manual_seed(2023+rand)
np.random.seed(2023+rand)
data = Data(x=torch.from_numpy(X).float(), edge_index=edge_index, y=torch.from_numpy(Y).float(), coord=coord)
data.n = data.x.shape[0]
transform = T.RandomNodeSplit(num_train_per_class=int(0.3*n_train), num_val=int(0.2*n_train), num_test=int(n-n_train))
data = transform(data)
Y_test = Y[data.test_mask]

torch.manual_seed(2023+rand)
data = utils.batch_gen(data, batch_size) # a
######################################################################################################################
torch.manual_seed(2023)
model_NN = Netp(p, k, q)
optimizer = torch.optim.Adam(model_NN.parameters(), lr=0.1)
patience_half = 10
patience = 20

_, _, model_NN = utils.train_gen_new(model_NN, optimizer, data, epoch_num=1000,
                                      patience = patience, patience_half = patience_half)
####################################################################################################################
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

theta_hat, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                              Update=True, patience=patience, patience_half=patience_half,
                                              Update_method='optimization')

utils_PDP.plot_PDP_realdata(model_NNGLS, X[~data.test_mask, :], variable_names)