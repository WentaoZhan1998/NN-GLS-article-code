import pandas as pd
import numpy as np

#df1 = pd.read_csv('covariate0628_corrected.csv')
df1 = pd.read_csv('covariate0605.csv')
#df2 = pd.read_csv('pm25_0628.csv')
df2 = pd.read_csv('pm25_0605.csv')
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

'''
N = lon.shape[0]
num_basis = [10**2,19**2,37**2]
knots_1dx = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
knots_1dy = [np.linspace(0,1,int(np.sqrt(i))) for i in num_basis]
##Wendland kernel
basis_size = 0
phi = np.zeros((N, sum(num_basis)))
for res in range(len(num_basis)):
    theta = 1/np.sqrt(num_basis[res])*2.5
    knots_x, knots_y = np.meshgrid(knots_1dx[res],knots_1dy[res])
    knots = np.column_stack((knots_x.flatten(),knots_y.flatten()))
    for i in range(num_basis[res]):
        d = np.linalg.norm(np.vstack((normalized_lon,normalized_lat)).T-knots[i,:],axis=1)/theta
        for j in range(len(d)):
            if d[j] >= 0 and d[j] <= 1:
                phi[j,i + basis_size] = (1-d[j])**6 * (35 * d[j]**2 + 18 * d[j] + 3)/3
            else:
                phi[j,i + basis_size] = 0
    basis_size = basis_size + num_basis[res]

## Romove the all-zero columns
idx_zero = np.array([], dtype=int)
for i in range(phi.shape[1]):
    if sum(phi[:,i]!=0)==0:
        idx_zero = np.append(idx_zero,int(i))

phi_reduce = np.delete(phi,idx_zero,1)
print(phi.shape)
print(phi_reduce.shape)
phi_obs = phi_reduce[idx_new,:]
'''

s_obs = np.vstack((normalized_lon[idx_new],normalized_lat[idx_new])).T
X = covariates[idx_new,:]
normalized_X = X
for i in range(X.shape[1]):
    normalized_X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))

import torch
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import *
import utils_NN

import importlib
importlib.reload(utils_NN)

def mat_OLS(idx, matrices):
    return(torch.eye(len(idx)))

def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

def RMSE_model(model, X, Y, mask, coord, theta_hat):
    X_train = X[mask, :]
    Y_train = Y[mask]
    X_test = X[np.invert(mask), :]
    Y_test = Y[np.invert(mask)]
    residual_train = torch.from_numpy(Y_train).reshape(-1) - model(torch.from_numpy(X_train).float()).reshape(-1)
    residual_train = residual_train.detach().numpy()
    df = pd.DataFrame(coord, columns=['x', 'y'])
    dist = distance_matrix(df.values, df.values)
    cov = utils_NN.make_cov(theta_hat, dist)
    theta_hat[2] = 0
    C = utils_NN.make_cov(theta_hat, dist)
    del df
    del dist
    residual_test = np.matmul(C[np.invert(mask), :][:, mask], np.linalg.solve(cov[mask, :][:, mask], residual_train))
    del cov
    del C
    Y_test_hat0 = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    return(RMSE(Y_test_hat0, Y_test)/RMSE(Y_test, np.mean(Y_test)))

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

k = 50
lr = 0.01
batch_size = 50
nn = 10

X = normalized_X
Y = z.reshape(-1)
coord = s_obs
n = coord.shape[0]
p = X.shape[1]
b = 1
model_fun = utils_NN.Netp
s = coord
n_train = int(n *0.8)

MISE = np.empty(0)
RMSE0 = np.empty(0)
RMSE_test = np.empty(0)
RMSE_test2 = np.empty(0)
RMSE_test3 = np.empty(0)

RMSE_DK1 = np.empty(0)
RMSE_DK2 = np.empty(0)
RMSE_DK3 = np.empty(0)

for j in range(100):
    print('rep ' + str(j))
    np.random.seed(j)
    id = np.random.choice(n, n_train, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[id] = True
    '''
    n_temp = 10
    k_temp = 10
    lx, ly = block_rand(n_temp, k_temp)
    xspc = np.linspace(0, b, n_temp + 1)
    yspc = np.linspace(0, b, n_temp + 1)
    mask = np.zeros(n, dtype=bool)
    for i in range(k_temp):
        mask_temp = np.logical_and((s[:, 0] > xspc[lx[i]]) * (s[:, 0] <= xspc[lx[i] + 1]),
                                   (s[:, 1] > yspc[ly[i]]) * (s[:, 1] <= yspc[ly[i] + 1]))
        mask = np.logical_or(mask, mask_temp)

    mask = np.invert(mask)    
    '''


    X_train = X[mask, :]
    Y_train = Y[mask]
    X_test = X[np.invert(mask), :]
    Y_test = Y[np.invert(mask)]
    s_train = coord[mask, :]
    s_test = coord[np.invert(mask), :]
    # matrices = I_B, F_diag, cov, cov_inv
    matrices = []

    torch.manual_seed(2021)
    train_loader, val_loader = utils_NN.set_loader(X_train, Y_train, prop=0.8, batch_size=batch_size)

    X_MISE = torch.from_numpy(X).float()
    Y_MISE = torch.from_numpy(Y).float()
    batch_num = 16

    MSE = torch.nn.MSELoss(reduction='mean')

    torch.manual_seed(2021)
    model0 = model_fun(p=p, k=k)
    optimizer = torch.optim.Adam(model0.parameters(), lr=0.1)
    losses, val_losses, model0 = utils_NN.train_gen(model0, optimizer, mat_OLS, 500, matrices,
                                                    train_loader, val_loader, X_MISE, Y_MISE, batch_num=batch_num)
    Y_test_hat1 = model0(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1)
    print("RMSE is %f" % float(RMSE(Y_test_hat1, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    RMSE_DK1 = np.append(RMSE_DK1, RMSE(Y_test_hat1, Y_test) / RMSE(Y_test, np.mean(Y_test)))
    ########################################################################################################################
    torch.manual_seed(2021)
    train_loader1, val_loader1 = utils_NN.set_loader(np.concatenate((X_train, s_train), axis=1), Y_train, prop=0.8,
                                                     batch_size=batch_size)
    model1 = model_fun(p=p + 2, k=k)
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.1)
    losses1, val_losses1, model1 = utils_NN.train_gen(model1, optimizer, mat_OLS, 500, matrices,
                                                      train_loader1, val_loader1, X_MISE, Y_MISE, batch_num,
                                                      losses_out=False)

    Xs_test = np.concatenate((X_test, s_test), axis=1)
    Y_test_hat2 = model1(torch.from_numpy(Xs_test).float()).detach().numpy().reshape(-1)
    print("RMSE is %f" % float(RMSE(Y_test_hat2, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    RMSE_DK2 = np.append(RMSE_DK2, RMSE(Y_test_hat2, Y_test) / RMSE(Y_test, np.mean(Y_test)))

    ########################################################################################################################
    torch.manual_seed(2021)
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
            d = np.linalg.norm(s - knots[i, :], axis=1) / theta_temp
            for j in range(len(d)):
                if d[j] >= 0 and d[j] <= 1:
                    phi_temp[j, i + K] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3
                else:
                    phi_temp[j, i + K] = 0
        K = K + num_basis[res]

    XRBF_train = np.hstack((X_train, phi_temp[mask, :]))
    # XRBF_train = np.concatenate((XRBF_train, s_train), axis=1)
    XRBF_test = np.hstack((X_test, phi_temp[np.invert(mask), :]))
    # XRBF_test = np.concatenate((XRBF_test, s_test), axis=1)

    train_loader2, val_loader2 = utils_NN.set_loader(XRBF_train, Y_train, prop=0.8, batch_size=batch_size)
    model2 = model_fun(p=p + K, k=100)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.1)
    losses2, val_losses2, model2 = utils_NN.train_gen(model2, optimizer, mat_OLS, 500, matrices,
                                                      train_loader2, val_loader2, X_MISE, Y_MISE, batch_num,
                                                      losses_out=False)
    Y_test_hat3 = model2(torch.from_numpy(XRBF_test).float()).detach().numpy().reshape(-1)
    print("RMSE is %f" % float(RMSE(Y_test_hat3, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    RMSE_DK3 = np.append(RMSE_DK3, RMSE(Y_test_hat3, Y_test) / RMSE(Y_test, np.mean(Y_test)))
    ####################################################################################################################
    residual_train = model0(torch.from_numpy(X_train).float()).reshape(-1) - torch.from_numpy(Y_train)
    residual_train = residual_train.detach().numpy()
    print('residual shape ' + str(residual_train.shape) + ' X ' + str(X_train.shape) + ' coord ' + str(s_train.shape))
    theta_hat = utils_NN.BRISC_estimation(residual_train, X_train, s_train)
    theta_hat0 = theta_hat
    print("RMSE is %f" % RMSE_model(model0, X, Y, mask, coord, theta_hat))
    RMSE0 = np.append(RMSE0, RMSE_model(model0, X, Y, mask, coord, theta_hat))
    #######################################################################################################################
    torch.manual_seed(2021)
    model_test = model_fun(p=p, k=k)
    optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.1)  # 0.00005 for SGD
    # train_step_test = make_train_step_test(model_test, optimizer_test)

    losses_test2, val_losses_test2, model_test, theta_hat = utils_NN.train_decor(X_train, Y_train,
                                                                                 theta_hat0, s_train, nn,
                                                                                 X_MISE, Y_MISE, 500,
                                                                                 model_test, optimizer_test, MSE,
                                                                                 train_loader, val_loader,
                                                                                 batch_num, shift=0, patience=20,
                                                                                 Update=False)

    print("RMSE is %f" % RMSE_model(model_test, X, Y, mask, coord, theta_hat))
    # MISE_test = np.append(MISE_test, losses_test2[torch.argmin(torch.stack(val_losses_test2))].detach().numpy())
    RMSE_test = np.append(RMSE_test, RMSE_model(model_test, X, Y, mask, coord, theta_hat))

    #######################################################################################################################
    torch.manual_seed(2021)
    model_test = model_fun(p=p, k=k)
    optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.1)  # 0.00005 for SGD
    # train_step_test = make_train_step_test(model_test, optimizer_test)

    losses_test2, val_losses_test2, model_test, theta_hat = utils_NN.train_decor(X_train, Y_train,
                                                                                 theta_hat0, s_train, nn,
                                                                                 X_MISE, Y_MISE, 500,
                                                                                 model_test, optimizer_test, MSE,
                                                                                 train_loader, val_loader,
                                                                                 batch_num, shift=0, patience=20,
                                                                                 Update=True,
                                                                                 Update_method='optimization',
                                                                                 Update_init=50,
                                                                                 Update_step=50,
                                                                                 Update_bound=10)

    # MISE_test2 = np.append(MISE_test2, losses_test2[torch.argmin(torch.stack(val_losses_test2))].detach().numpy())
    # RMSE_test2 = np.append(RMSE_test2, RMSE_model(model_test, X, Y, mask, coord, theta_hat))
    print("RMSE is %f" % RMSE_model(model_test, X, Y, mask, coord, theta_hat))
    RMSE_test3 = np.append(RMSE_test3, RMSE_model(model_test, X, Y, mask, coord, theta_hat))

df_RMSE = pd.DataFrame(
        {'RMSE_OLS': RMSE0, 'RMSE_GLS': RMSE_test, 'RMSE_GLS_update_50': RMSE_test3,
         'RMSE_DK1': RMSE_DK1, 'RMSE_DK2': RMSE_DK2, 'RMSE_DK3': RMSE_DK3,})
df_RMSE.to_csv(".//simulation//realdata//RMSE_0704.csv")
