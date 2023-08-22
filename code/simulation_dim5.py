'''
Run the following code after utils_NN.Simulate_NNGP
'''
import sys
import torch
import numpy as np

import lhsmdu
import pandas as pd
'''
import os
os.environ['R_HOME']
os.environ['R_HOME'] = 'C:\\Program Files\\R\\R-4.1.0'
os.environ['R_HOME']
os.environ['R_LIBS_USER'] = 'C:\\Users\\15211\\Documents\\R\\win-library\\4.1'
'''
import utils_NN
from scipy.stats import multivariate_normal
import time

import importlib
importlib.reload(utils_NN)

def mat_OLS(idx, matrices):
    return(torch.eye(len(idx)))

def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

########################################################################################################################
x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
var1 = multivariate_normal(mean=[0.25, 0.25], cov=[[0.01, 0], [0, 0.01]])
var2 = multivariate_normal(mean=[0.6, 0.9], cov=[[0.01, 0], [0, 0.01]])
mean = np.mean(var1.pdf(pos) + var2.pdf(pos))
var = np.var(var1.pdf(pos) + var2.pdf(pos))

def corerr_gen(pos): # must be designed for unit square
    n = pos.shape[0]
    return ((var1.pdf(pos) + var2.pdf(pos) - mean) * np.sqrt(sigma) / np.sqrt(var) + np.sqrt(sigma*tau)*np.random.randn(n))

########################################################################################################################
sigma = int(sys.argv[1])
phi = int(sys.argv[2])
tau = float(sys.argv[3])
method = str(sys.argv[4])
theta = [sigma, phi / np.sqrt(2), tau]
p = 5
model_fun = utils_NN.Netp
funXY = utils_NN.f5
k = 50
lr = 0.01
b = 10
batch_size = 100
ordered = False

n = 2000
n_train = int(n/2)
nn = 10

N = 1000
n_small = int(N / 100)
np.random.seed(2021)
X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
for i in range(99):
    temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
    X_MISE = np.concatenate((X_MISE, temp))

Y_MISE = funXY(X_MISE)
X_MISE = torch.from_numpy(X_MISE).float()

Y_MISE = torch.from_numpy(Y_MISE).float()
Y_MISE = torch.reshape(Y_MISE, (N, 1))

MISE0 = np.empty(0)
RMSE0 = np.empty(0)
MISE_test_o = np.empty(0)
MISE_test = np.empty(0)
RMSE_test = np.empty(0)
MISE_test2 = np.empty(0)
RMSE_test2 = np.empty(0)
MISE_test3 = np.empty(0)
RMSE_test3 = np.empty(0)

RMSE_DK1 = np.empty(0)
RMSE_DK2 = np.empty(0)
RMSE_DK3 = np.empty(0)

MISE_DK1 = np.empty(0)
MISE_DK2 = np.empty(0)
MISE_DK3 = np.empty(0)

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

def order(X, Y, coord):
    s_sum = coord[:, 0] + coord[:, 1]
    order = s_sum.argsort()
    X_new = X[order, :]
    Y_new = Y[order]
    coord_new = coord[order, :]
    return X_new, Y_new, coord_new

for j in range(500):
    print(j)
    np.random.seed(j)
    if method == '2':
        X, Y, rank, coord, corerr = utils_NN.Simulate_mis(n, p, funXY, nn, theta, corerr_gen, a=0, b=b)
    else:
        X, Y, I_B, F_diag, rank, coord, cov, corerr = utils_NN.Simulate_NNGP(n, p, funXY, nn, theta, method=method, a=0,
                                                                             b=b)

    s = coord
    np.random.seed(2022)
    # id = np.random.choice(n, n_train, replace=False)
    id = range(n_train)
    mask = np.zeros(n, dtype=bool)
    mask[id] = True

    '''
    n_temp = 10
    k_temp = 10
    lx, ly = block_rand(n_temp, k_temp)
    lspc = np.linspace(0, b, n_temp + 1)
    mask = np.zeros(n, dtype=bool)
    for i in range(k_temp):
        mask_temp = np.logical_and((s[:, 0] > lspc[lx[i]]) * (s[:, 0] <= lspc[lx[i] + 1]),
                                   (s[:, 1] > lspc[ly[i]]) * (s[:, 1] <= lspc[ly[i] + 1]))
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

    if ordered ==  True:
        X_train, Y_train, s_train = order(X_train, Y_train, s_train)

    torch.manual_seed(2021)
    train_loader, val_loader = utils_NN.set_loader(X_train, Y_train, prop=0.8, batch_size=batch_size)

    batch_num = 16

    MSE = torch.nn.MSELoss(reduction='mean')

    print("Mean shift is %f" % np.mean(corerr))

    ########################################################################################################################
    # predict = utils_NN.RF_prediction(X, Y, coord, X_MISE)
    # MISE_RF = np.append(MISE_RF, MSE(predict, Y_MISE.reshape(-1)))
    ########################################################################################################################
    torch.manual_seed(2021)
    model0 = model_fun(p=p, k=k)
    optimizer = torch.optim.Adam(model0.parameters(), lr=0.1)
    losses, val_losses, model0 = utils_NN.train_gen(model0, optimizer, mat_OLS, 500, matrices,
                                                    train_loader, val_loader, X_MISE, Y_MISE, batch_num)
    Y_test_hat1 = model0(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1)
    # print("RMSE is %f" % float(RMSE(Y_test_hat1, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    RMSE_DK1 = np.append(RMSE_DK1, RMSE(Y_test_hat1, Y_test) / RMSE(Y_test, np.mean(Y_test)))
    MISE_DK1 = np.append(MISE_DK1, losses[torch.argmin(torch.stack(val_losses))])

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
    # print("RMSE is %f" % float(RMSE(Y_test_hat2, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    MISE_DK2 = np.append(MISE_DK2,
                         np.square(RMSE(Y_MISE.detach().numpy(), utils_NN.PDP(model1, X_MISE.detach().numpy(), s))))
    RMSE_DK2 = np.append(RMSE_DK2, RMSE(Y_test_hat2, Y_test) / RMSE(Y_test, np.mean(Y_test)))

    ########################################################################################################################
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
            d = np.linalg.norm(s / b - knots[i, :], axis=1) / theta_temp
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

    torch.manual_seed(2021)
    train_loader2, val_loader2 = utils_NN.set_loader(XRBF_train, Y_train, prop=0.8, batch_size=batch_size)
    model2 = model_fun(p=p + K, k=100)
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.1)
    losses2, val_losses2, model2 = utils_NN.train_gen(model2, optimizer, mat_OLS, 500, matrices,
                                                      train_loader2, val_loader2, X_MISE, Y_MISE, batch_num,
                                                      losses_out=False)
    Y_test_hat3 = model2(torch.from_numpy(XRBF_test).float()).detach().numpy().reshape(-1)
    # print("RMSE is %f" % float(RMSE(Y_test_hat3, Y_test) / RMSE(Y_test, np.mean(Y_test))))
    # print("MISE is %f" % float(RMSE(Y_MISE.detach().numpy(), utils_NN.PDP(model2, X_MISE.detach().numpy(), phi_temp))))
    MISE_DK3 = np.append(MISE_DK3,
                         np.square(
                             RMSE(Y_MISE.detach().numpy(), utils_NN.PDP(model2, X_MISE.detach().numpy(), phi_temp))))
    RMSE_DK3 = np.append(RMSE_DK3, RMSE(Y_test_hat3, Y_test) / RMSE(Y_test, np.mean(Y_test)))
    ####################################################################################################################
    residual_train = model0(torch.from_numpy(X_train).float()).reshape(-1) - torch.from_numpy(Y_train)
    residual_train = residual_train.detach().numpy()
    theta_hat = utils_NN.BRISC_estimation(residual_train, X_train, s_train)
    theta_hat0 = theta_hat
    # print("RMSE is %f" % utils_NN.RMSE_model(model0, X, Y, mask, coord, theta_hat))
    RMSE0 = np.append(RMSE0, utils_NN.RMSE_model(model0, X, Y, mask, coord, theta_hat))
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
                                                                                 batch_num, shift=0,
                                                                                 patience=20, patience_half=5,
                                                                                 Update=False)

    MISE_test = np.append(MISE_test, losses_test2[torch.argmin(torch.stack(val_losses_test2))].detach().numpy())
    RMSE_test = np.append(RMSE_test, utils_NN.RMSE_model(model_test, X, Y, mask, coord, theta_hat))
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
                                                                                 batch_num, shift=0,
                                                                                 patience=20, patience_half=5,
                                                                                 Update=True,
                                                                                 Update_method='optimization')

    MISE_test2 = np.append(MISE_test2, losses_test2[torch.argmin(torch.stack(val_losses_test2))].detach().numpy())
    RMSE_test2 = np.append(RMSE_test2, utils_NN.RMSE_model(model_test, X, Y, mask, coord, theta_hat))

name = "p%i"%p + 'phi%i'%phi + "sig%i"%(theta[0]) + "tau%i"%(int(100*tau)) + 'mtd' + method
df_MISE = pd.DataFrame(
        {'MISE_GLS': MISE_test, 'MISE_GLS_update': MISE_test2,
         'MISE_DK1': MISE_DK1, 'MISE_DK2': MISE_DK2, 'MISE_DK3': MISE_DK3,})
df_MISE.to_csv(".//simulation//MISE//1dim//" + name + '_rand.csv')
df_RMSE = pd.DataFrame(
        {'NN_krig': RMSE0, 'NNGLS_krig': RMSE_test, 'NNGLS_update_krig': RMSE_test2,
         'NN': RMSE_DK1, 'DK': RMSE_DK2, 'DK_spline': RMSE_DK3,})
df_RMSE.to_csv(".//simulation//RMSE//1dim//" + name + '_rand.csv')
