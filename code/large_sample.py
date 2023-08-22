import sys
import torch
import numpy as np
import random
import matplotlib

#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import lhsmdu
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.optimize import *
import utils_NN
from scipy.stats import multivariate_normal
import time

def mat_OLS(idx, matrices):
    return(torch.eye(len(idx)))

########################################################################################################################
x, y = np.mgrid[0:1:.01, 0:1:.01]
pos = np.dstack((x, y))
var1 = multivariate_normal(mean=[0.25, 0.25], cov=[[0.0025, 0], [0, 0.0025]])
var2 = multivariate_normal(mean=[0.6, 0.9], cov=[[0.0025, 0], [0, 0.0025]])
mean = np.mean(var1.pdf(pos) + var2.pdf(pos))
var = np.var(var1.pdf(pos) + var2.pdf(pos))
########################################################################################################################
sigma = int(sys.argv[1])
phi = int(sys.argv[2])
tau = float(sys.argv[3])
theta = [sigma, phi / np.sqrt(2), tau]
p = 5
model_fun = utils_NN.Netp
funXY = utils_NN.f5
k = 50
method = str(sys.argv[4])
lr = 0.01
b = 10

def corerr_gen(pos): # must be designed for unit square
    n = pos.shape[0]
    return ((var1.pdf(pos) + var2.pdf(pos) - mean) * np.sqrt(sigma) / np.sqrt(var) + np.sqrt(sigma*tau)*np.random.randn(n))

nn = 10

MISE = np.empty(0)
MISE_test3 = np.empty(0)
time1 = np.empty(0)
time2 = np.empty(0)
time3 = np.empty(0)
time_BRISC = np.empty(0)
label = np.empty(0)

random.seed(2021)
Theta_df = pd.DataFrame({'sigma': [], 'tau': [], 'phi':[]})
Theta_df3 = pd.DataFrame({'sigma': [], 'tau': [], 'phi':[]})

for n in np.concatenate((100*(np.array(range(9))+1), 1000*(np.array(range(10))+1)), axis = 0):
    batch_size = int(50*n/1000)
    for j in range(10):
        start_time = time.time()
        print(j)
        if method == '2':
            X, Y, rank, coord, corerr = utils_NN.Simulate_mis(n, p, funXY, nn, corerr_gen, a=0, b=b)
        else:
            X, Y, I_B, F_diag, rank, coord, cov, corerr = utils_NN.Simulate_NNGP(n, p, funXY, nn, theta, method=method,
                                                                                 a=0,
                                                                                 b=b)

        df = pd.DataFrame(coord, columns=['x', 'y'])
        dist = distance_matrix(df.values, df.values)

        # matrices = I_B, F_diag, cov, cov_inv
        matrices = []

        N = 1000
        n_small = int(N / 100)
        random.seed(2021)
        X_MISE = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
        for i in range(99):
            temp = np.array(lhsmdu.sample(p, n_small).reshape((n_small, p)))
            X_MISE = np.concatenate((X_MISE, temp))

        Y_MISE = funXY(X_MISE)
        X_MISE = torch.from_numpy(X_MISE).float()

        Y_MISE = torch.from_numpy(Y_MISE).float()
        Y_MISE = torch.reshape(Y_MISE, (N, 1))

        train_loader, val_loader = utils_NN.set_loader(X, Y, prop=0.8, batch_size=batch_size)
        batch_num = 16

        MSE = torch.nn.MSELoss(reduction='mean')

        print("Mean shift is %f" % np.mean(corerr))
        time1 = np.append(time1, time.time() - start_time)
        ########################################################################################################################
        start_time = time.time()
        model0 = model_fun(p=p, k=k)
        optimizer = torch.optim.Adam(model0.parameters(), lr=0.1)
        losses, val_losses, model0 = utils_NN.train_gen(model0, optimizer, mat_OLS, 500, matrices,
                                                        train_loader, val_loader, X_MISE, Y_MISE, batch_num)
        MISE = np.append(MISE, losses[torch.argmin(torch.stack(val_losses))])
        time2 = np.append(time2, time.time() - start_time)
        ####################################################################################################################
        start_time = time.time()
        residual = model0(torch.from_numpy(X).float()).reshape(-1) - torch.from_numpy(Y)
        residual = residual.detach().numpy()

        theta_hat = utils_NN.BRISC_estimation(residual, X, coord)
        theta_hat0 = theta_hat
        Theta_df = pd.concat(
            [Theta_df, pd.DataFrame(np.array([np.append([], theta_hat)]), columns=['sigma', 'phi', 'tau'])], axis=0)
        time_BRISC = np.append(time_BRISC, time.time() - start_time)
        ####################################################################################################################
        start_time = time.time()
        model_test = model_fun(p=p, k=k)
        optimizer_test = torch.optim.Adam(model_test.parameters(), lr=0.1)  # 0.00005 for SGD
        # train_step_test = make_train_step_test(model_test, optimizer_test)

        losses_test2, val_losses_test2, model_test, theta_hat = utils_NN.train_decor(X, Y,
                                                                                     theta_hat0, coord, nn,
                                                                                     X_MISE, Y_MISE, 500,
                                                                                     model_test, optimizer_test, MSE,
                                                                                     train_loader, val_loader,
                                                                                     batch_num, shift=0, patience=20,
                                                                                     Update=True,
                                                                                     Update_method='optimization')

        MISE_test3 = np.append(MISE_test3, losses_test2[torch.argmin(torch.stack(val_losses_test2))].detach().numpy())
        Theta_df3 = pd.concat(
            [Theta_df3, pd.DataFrame(np.array([np.append([], theta_hat)]), columns=['sigma', 'phi', 'tau'])], axis=0)
        time3 = np.append(time3, time.time() - start_time)
        #######################################################################################################################
        label = np.append(label, n)


df = pd.DataFrame(
        {'NN': MISE, 'NNGLS_update': MISE_test3,
         't_gen': time1, 't_NN':time2, 't_BRISC':time_BRISC, 't_NNGLS':time3})

name = "p%i"%p + 'phi%i'%(int(phi)) + "sig%i"%(int(sigma)) + "tau%i"%(int(100*tau)) + 'mtd' + method
df.to_csv('.//simulation//MISE//large//' + name + '.csv')
#Theta_df.to_csv(name + '_theta.csv')
#Theta_df3.to_csv(name + '_theta2.csv')
