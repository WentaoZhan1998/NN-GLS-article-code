import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance_matrix
from scipy.optimize import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import random_split
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import time

def rmvn(m, mu, cov):
    p = len(mu)
    D = np.linalg.cholesky(cov)
    res = np.matmul(np.random.randn(m, p), np.matrix.transpose(D)) + mu
    return  res.reshape(-1) # * np.ones((m, p))

def make_cov(theta, dist):
    n = dist.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    cov = sigma_sq * np.exp(-phi * dist) + tau_sq * np.eye(n)
    return (cov)

def make_rank(dist, nn):
    n = dist.shape[0]
#    r = np.arange(n) ##12.29
#    mask = r[:, None] < r
#    dist[mask] = dist[mask] + np.max(dist)
    rank = np.argsort(dist, axis=-1)
    rank = rank[:, 1:(nn + 1)]
    return(rank)

def make_bf(cov, rank):
    n = cov.shape[0]
    B = np.zeros((n, n))
    F = np.zeros(n)
    for i in range(n):
        F[i] = cov[i, i]
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = cov[ind, :][:, ind]
        F[i] = cov[i, i]
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            bi = np.linalg.solve(cov_sub, cov[ind, i])
            B[i, ind] = bi
            F[i] = cov[i, i] - np.inner(cov[ind, i], bi)

    return B, F

def bf(cov, dist, nn):
    rank = make_rank(dist, nn)

    B, F = make_bf(cov, rank)

    return B, F, rank;

def bf_from_theta(theta, coord, nn, method = '0', nu = 1.5):
    n = coord.shape[0]
    df = pd.DataFrame(coord, columns=['x', 'y'])
    dist = distance_matrix(df.values, df.values)
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if method == '0':
        cov = sigma_sq * np.exp(-phi * dist) + tau_sq * np.eye(n)
    elif method == '1':
        cov = sigma_sq*pow((dist*phi),nu)/(pow(2,(nu-1))*scipy.special.gamma(nu))*scipy.special.kv(nu, dist*phi)
        cov[range(n), range(n)] = sigma_sq + tau_sq

    B, F_diag, rank = bf(cov, dist, nn)
    B = torch.from_numpy(B)
    I_B = torch.eye(n) - B
    F_diag = torch.from_numpy(F_diag)

    return I_B, F_diag, rank, cov

def Simulate_NNGP(n, p, fx, nn, theta, method = '0', nu = 1.5, a = 0, b = 1):
    #n = 1000
    coord = np.random.uniform(low = a, high = b, size=(n, 2))
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    I_B, F_diag, rank, cov = bf_from_theta(theta, coord, nn, method = method, nu = nu)

    X = np.random.uniform(size=(n, p))
    corerr = rmvn(1, np.zeros(n), cov)

    Y = fx(X).reshape(-1) + corerr + np.sqrt(tau_sq) * np.random.randn(n)

    return X, Y, I_B, F_diag, rank, coord, cov, corerr


rmvn_r = robjects.r('''
        Simulate_R = function(n, p, sigma, phi, tau, a = 0, b = 1, seed = 2021){
        rmvn <- function(n, mu = 0, V = matrix(1)){
            p <- length(mu)
            D <- chol(V)
            t(matrix(rnorm(n*p), ncol=p)%*%D + rep(mu,rep(n,p)))
        }

        set.seed(seed)
        coords <- matrix(runif(2*n, a, b), nrow = n)
        x <- matrix(runif(n*p, 0,1), nrow = n)
        D <- as.matrix(dist(coords))
        R <- exp(-phi*D) + tau*diag(n)
        set.seed(seed)
        w <- rmvn(1, rep(0,n), sigma*R)

        res = list()
        res$X = x
        res$corerr = w
        res$coords = coords

        return(res)
        }
        ''')

def Simulate_NNGP_R(n, p, fx, nn, theta, method = '0', nu = 1.5, a = 0, b = 1, seed = 2021):
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    res = rmvn_r(robjects.IntVector([n]), robjects.IntVector([p]), robjects.FloatVector([sigma_sq]),
               robjects.FloatVector([phi]), robjects.FloatVector([tau]), b=robjects.FloatVector([b]),
               seed = robjects.IntVector([seed]))
    X = np.array(res[0])
    corerr = np.array(res[1]).reshape(-1)
    coord = np.array(res[2])
    I_B, F_diag, rank, cov = bf_from_theta(theta, coord, nn, method = method, nu = nu)
    Y = fx(X).reshape(-1) + corerr
    return X, Y, I_B, F_diag, rank, coord, cov, corerr

def Simulate_mis(n, p, fx, nn, theta, corerr_gen, a=0, b=1):
    # n = 1000
    coord = np.random.uniform(low=a, high=b, size=(n, 2))
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    corerr = corerr_gen(coord/(b-a))

    df = pd.DataFrame(coord, columns=['x', 'y'])
    dist = distance_matrix(df.values, df.values)
    rank = np.argsort(dist, axis=-1)
    rank = rank[:, 1:(nn + 1)]

    X = np.random.uniform(size=(n, p))

    Y = fx(X).reshape(-1) + corerr

    return X, Y, rank, coord, corerr

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return (self.x[index], self.y[index], index)

    def __len__(self):
        return len(self.x)

def one_loader(X, Y, batch_size = 50):
    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(Y).float()
    dataset = CustomDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    return loader

def set_loader(X, Y, prop = 0.8, batch_size = 50):
    n = len(Y)
    n_cut = int(n * prop)
    x_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(Y).float()

    dataset = CustomDataset(x_tensor, y_tensor)

    train_dataset, val_dataset = random_split(dataset, [n_cut, n - n_cut])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=n - n_cut, shuffle=False)

    return train_loader, val_loader

# Models ###############################################################################################################

class Net_linear(torch.nn.Module):
    def __init__(self, p = 1, k = 10):
        super(Net_linear, self).__init__()
        self.l1 = torch.nn.Linear(p, 1)

    def forward(self, x):
        return self.l1(x)

class Net(torch.nn.Module):
    def __init__(self, k = 50):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(3, k)
        self.l2 = torch.nn.Linear(k, 1)

    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        return self.l2(x)

class Net_2layer(torch.nn.Module):
    def __init__(self, p, k = 50, k2 = 20):
        super(Net_2layer, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, k2)
        self.l3 = torch.nn.Linear(k2, 1)

    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = F.sigmoid(self.l2(x))
        return self.l3(x)

class Net2(torch.nn.Module):
    def __init__(self, p, k = 50):
        super(Net2, self).__init__()
        self.l1 = torch.nn.Linear(p, 1)

    def forward(self, x):
        return self.l1(x)

class Net3(torch.nn.Module):
    def __init__(self, p, k = 50):
        super(Net3, self).__init__()
        self.l1 = torch.nn.Linear(1, k)
        self.l2 = torch.nn.Linear(k, 1)
        #self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        #x = torch.sigmoid(self.l2(x))
        return self.l2(x)

class Net5(torch.nn.Module):
    def __init__(self, p, k = 50):
        super(Net5, self).__init__()
        self.l1 = torch.nn.Linear(5, k)
        self.l2 = torch.nn.Linear(k, 1)
        #self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        #x = torch.sigmoid(self.l2(x))
        return self.l2(x)

class Netp(torch.nn.Module):
    def __init__(self, p, k = 50):
        super(Netp, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, 1)
        #self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = torch.sigmoid(self.l1(x))
        #x = torch.sigmoid(self.l2(x))
        return self.l2(x)

# Losses ###############################################################################################################

class Decorloss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Decorloss, self).__init__()

    def forward(self, inputs, targets, B0, F0):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        dif = inputs - targets

        temp = torch.sqrt(torch.reciprocal(F0).double()) * torch.transpose(B0.double(), 0, 1)
        dif_decor = torch.matmul(torch.transpose(temp, 0, 1), dif.double())

        return torch.sum(dif_decor.pow(2))/len(dif)

class Decorloss_nh(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Decorloss, self).__init__()

    def forward(self, inputs, targets, neighbors, B0, F0):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        dif = inputs - targets

        temp = torch.sqrt(torch.reciprocal(F0).double()) * torch.transpose(B0.double(), 0, 1)
        dif_decor = torch.matmul(torch.transpose(temp, 0, 1), dif.double())

        return torch.sum(dif_decor.pow(2))/len(dif)

class Invloss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Invloss, self).__init__()

    def forward(self, inputs, targets, Cov):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        dif = inputs - targets

        P = np.linalg.inv(np.linalg.cholesky(Cov))
        C_inv = np.dot(np.transpose(P), P)

        C_invdif = torch.matmul(torch.from_numpy(C_inv), dif.double())

        return torch.dot(C_invdif, dif.double())

class Myloss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Myloss, self).__init__()

    def forward(self, inputs, targets, Mat):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        dif = inputs - targets

        temp = torch.matmul(Mat.double(), dif.double())

        return torch.dot(temp, dif.double())

# Stopping #############################################################################################################

class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=5, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

# Functions ############################################################################################################

def fx(X): return 10 * np.sin(np.pi * X)

def fx_l(x): return 0*x + 2

def fx4(X): return 10 * np.sin(4 * np.pi * X)

def f3(X): return (X[:, 0] + pow(X[:, 1], 1) + 2 * X[:, 2] + 2)

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

def f15(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4] +
                    3/(X[:,5]+1)/(X[:,6]+1) + 4*np.exp(np.square(X[:,7])) + 30*np.square(X[:,8])*X[:,9] +
                    5*(np.exp(np.square(X[:,10]))*np.sin(np.pi*X[:,11]) + np.exp(np.square(X[:,11]))*np.sin(np.pi*X[:,10])) +
                    10*np.square(X[:,12])*np.cos(np.pi*X[:,13]) + 20*np.square(X[:,14]))/6

# R ####################################################################################################################

def import_BRISC():
    BRISC = importr('BRISC')
    return BRISC

def import_RF():
    RF = importr('randomForest')
    return RF

def import_RFGLS():
    RFGLS = importr('RandomForestsGLS')
    return RFGLS

BRISC = import_BRISC()
RF = import_RF()
RFGLS = import_RFGLS()

def BRISC_estimation(residual, X, coord):
    residual_r = robjects.FloatVector(residual)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)

    if X == 'NULL':
        res = BRISC.BRISC_estimation(coord_r, residual_r)
    else:
        Xr = robjects.FloatVector(X.transpose().reshape(-1))
        Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
        res = BRISC.BRISC_estimation(coord_r, residual_r, Xr)

    theta_hat = res[9]
    theta_hat = np.array(theta_hat)
    phi = theta_hat[2]
    tau_sq = theta_hat[1]
    sigma_sq = theta_hat[0]
    theta_hat[1] = phi
    theta_hat[2] = tau_sq / sigma_sq

    return theta_hat

def RF_prediction(X, Y, coord, X_MISE):
    Xr = robjects.FloatVector(X.transpose().reshape(-1))
    Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
    Y_r = robjects.FloatVector(Y)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)

    X_MISE_r = robjects.FloatVector(X_MISE.detach().numpy().transpose().reshape(-1))
    X_MISE_r = robjects.r['matrix'](X_MISE_r, ncol=X.shape[1])

    res = RF.randomForest(Xr, Y_r)
    predict = robjects.r['predict'](res, X_MISE_r)
    del res
    predict = torch.from_numpy(np.array(predict))
    return predict

def RFGLS_prediction(X, Y, coord, X_MISE):
    Xr = robjects.FloatVector(X.transpose().reshape(-1))
    Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
    Y_r = robjects.FloatVector(Y)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)

    X_MISE_r = robjects.FloatVector(X_MISE.detach().numpy().transpose().reshape(-1))
    X_MISE_r = robjects.r['matrix'](X_MISE_r, ncol=X.shape[1])

    res = RFGLS.RFGLS_estimate_spatial(coord_r, Y_r, Xr)
    predict = RFGLS.RFGLS_predict(res, X_MISE_r)[1]
    del res
    predict = torch.from_numpy(np.array(predict))
    return predict

# Training #############################################################################################################

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y, C):
        # Sets model to TRAIN mode
        y = y.view(-1)
        model.train()
        # Makes predictions
        yhat = model(x)
        yhat = yhat.view(-1)
        # Computes loss
        loss = loss_fn(y, yhat, C)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step

def make_train_step_decor(model, optimizer):
    # Builds function that performs a step in the train loop
    loss_fn_local = Decorloss()
    def train_step_decor(x, y, B, F):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn_local(y, yhat, B, F)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step_decor

def make_train_step_inv(model, optimizer):
    # Builds function that performs a step in the train loop
    loss_fn_local = Invloss()
    def train_step_inv(x, y, Cov):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn_local(y, yhat, Cov)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    # Returns the function that will be called inside the train loop
    return train_step_inv

MSE = torch.nn.MSELoss(reduction='mean')

def train_gen(model, optimizer, mat_fun, epoch_num, matrices, train_loader, val_loader, X_MISE, Y_MISE, batch_num,
              shift = 0, patience = 20, patience_half = 10,
              losses_out = True):
    #model = model_fun()
    loss_fn = Myloss()
    train_step = make_train_step(model, loss_fn, optimizer)
    lr_scheduler = LRScheduler(optimizer, patience = patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)

    losses = []
    val_losses = []

    for epoch in range(epoch_num):
        start_time = time.time()
        val_acu = 0
        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            # for x_batch, y_batch in train_loader:
            # x_batch = x_batch.to(device)
            # y_batch = y_batch.to(device)
            mat = mat_fun(idx, matrices)

            loss = train_step(x_batch, y_batch - shift, mat)

            with torch.no_grad():
                for _, (x_val, y_val, idx0) in enumerate(val_loader):
                    # x_val = x_val.to(device)
                    # y_val = y_val.to(device)

                    model.eval()

                    yhat = model(x_val).view(-1)
                    mat = mat_fun(idx0, matrices)
                    val_loss = loss_fn(y_val - shift, yhat, mat)
                    val_acu += val_loss
                    # val_losses.append(val_loss.item())

        with torch.no_grad():
            if losses_out == True:
                Y_MISE_hat = model(X_MISE)
                print(MSE(Y_MISE, Y_MISE_hat))
                losses.append(MSE(Y_MISE, Y_MISE_hat))

        val_losses.append(val_acu / batch_num)
        lr_scheduler(val_acu / batch_num)
        early_stopping(val_acu / batch_num)
        if early_stopping.early_stop:
            break
        #print("--- %s seconds ---" % (time.time() - start_time))
    return losses, val_losses, model

def train_decor(X, Y,
                theta_hat0, coord, nn,
                X_MISE, Y_MISE, iter_max,
                model, optimizer, loss_fn,
                train_loader, val_loader,
                batch_num, shift = 0, patience = 20, patience_half = 10,
                NNGP = True, Update = True,
                Update_method = 'optimization', Update_init = 50, Update_step = 50, Update_bound = 0.1,
                Update_lr_ctrl = False):
    n = len(Y)
    theta_hat = theta_hat0.copy()####12.15 update
    if NNGP != True: Update = False

    df = pd.DataFrame(coord, columns=['x', 'y'])
    dist = distance_matrix(df.values, df.values)

    I_B, F_diag, rank, cov = bf_from_theta(theta_hat, coord, nn)
    if NNGP != True: Chol = np.linalg.inv(np.linalg.cholesky(cov))

    rank_list = []
    for i in range(n):
        rank_list.append(rank[i, :][np.where(rank[i, :] < i)])
    lr_scheduler = LRScheduler(optimizer, patience = patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta = 0.00001)

    Y_test = torch.from_numpy(Y - shift).float()
    losses_test = []
    val_losses_test = []
    model_optim = model
    model.train()

    for epoch in range(iter_max):
        print(epoch)
        if NNGP:
            Y_test_test = torch.sqrt(torch.reciprocal(F_diag).double()) * torch.matmul(I_B.double(), Y_test.double())
        else:
            Y_test_test = torch.matmul(torch.from_numpy(Chol).double(), Y_test.double())#Y
        val_acu_test = 0
        for batch_idx, (x_batch, y_batch, idx) in enumerate(train_loader):
            model.train()
            Y_hat = model(torch.from_numpy(X).float()).reshape(-1).double()
            temp = []
            for i in idx:
                if NNGP:
                    id = np.append(rank_list[i], i)
                    temp.append(
                        torch.sqrt(torch.reciprocal(F_diag[i]).double()) * torch.dot(I_B.double()[i, id].reshape(-1),
                                                                                     Y_hat[id]))
                else:
                    temp.append(torch.dot(torch.from_numpy(Chol).double()[i,:].reshape(-1),Y_hat))
            Y_hat_test = torch.stack(temp).reshape(-1)
            loss = loss_fn(Y_hat_test, Y_test_test[idx])
            loss.backward()
            # Updates parameters and zeroes gradients
            optimizer.step()
            optimizer.zero_grad()

            with torch.no_grad():
                for _, (x_val, y_val, idx_0) in enumerate(val_loader):
                    model.eval()
                    if len(idx_0) == 0:
                        continue

                    temp = []
                    for i in idx_0:
                        if NNGP:
                            id = np.append(rank_list[i], i)
                            temp.append(torch.sqrt(torch.reciprocal(F_diag[i]).double()) * torch.dot(
                                I_B.double()[i, id].reshape(-1),Y_hat[id]))
                        else: temp.append(torch.dot(torch.from_numpy(Chol).double()[i,:].reshape(-1),Y_hat))
                    Y_hat_test = torch.stack(temp).reshape(-1)
                    val_loss = loss_fn(Y_hat_test, Y_test_test[idx_0])
                    val_acu_test += val_loss

        with torch.no_grad():
            Y_MISE_hat = model(X_MISE)
            loss_temp = loss_fn(Y_MISE, Y_MISE_hat)
            #print(loss_fn(Y_MISE, Y_MISE_hat))
            losses_test.append(loss_temp)
            print(loss_temp)

            Y_hat = model(torch.from_numpy(X).float()).reshape(-1).double()

            if (epoch >= Update_init) & (epoch % Update_step == 0) & Update:
                if Update_method == 'optimization':
                    def test2(theta_hat_test):
                        sigma, phi, tau = theta_hat_test
                        cov = sigma * (np.exp(-phi * dist) + tau * np.eye(n))  # need dist, n

                        Y_hat_local = Y_hat.detach().numpy()
                        err = Y_hat_local - Y
                        term1 = 0
                        term2 = 0
                        for i in range(n):
                            ind = rank[i, :][rank[i, :] <= i]
                            id = np.append(ind, i)

                            sub_cov = cov[ind, :][:, ind]
                            if np.linalg.det(sub_cov):
                                bi = np.linalg.solve(cov[ind, :][:, ind], cov[ind, i])
                            else:
                                bi = np.zeros(ind.shape)
                            I_B_i = np.append(-bi, 1)
                            F_i = cov[i, i] - np.inner(cov[ind, i], bi)
                            err_decor = np.sqrt(np.reciprocal(F_i)) * np.dot(I_B_i, err[id])
                            term1 += np.log(F_i)
                            term2 += err_decor ** 2
                        return (term1 + term2)

                    def constraint1(x):
                        return x[2]

                    def constraint2(x):
                        return x[0]

                    cons = [{'type': 'ineq', 'fun': constraint1},
                            {'type': 'ineq', 'fun': constraint2}]

                    res = minimize(test2, theta_hat, constraints=cons)
                    # sigma_temp, phi_temp, tau_temp = theta_hat.detach().numpy()
                    # print('det')
                    # print(np.linalg.det(sigma_temp*(np.exp(-phi_temp * dist) + tau_temp * np.eye(n))))
                    theta_hat_new = res.x
                elif Update_method == 'BRISC':
                    residual_temp = model(torch.from_numpy(X).float()).reshape(-1) - torch.from_numpy(Y)
                    residual_temp = residual_temp.detach().numpy()
                    theta_hat_new = BRISC_estimation(residual_temp, X, coord)

                print(theta_hat_new)
                if np.sum((theta_hat_new - theta_hat) ** 2) / np.sum((theta_hat) ** 2) < Update_bound:
                    theta_hat = theta_hat_new

                    cov_hat = make_cov(theta_hat, dist)

                    B_hat, F_diag = make_bf(cov_hat, rank)
                    B_hat = torch.from_numpy(B_hat)
                    I_B = torch.eye(n) - B_hat
                    F_diag = torch.from_numpy(F_diag)
                    print('theta updated')
                    if Update_lr_ctrl == True:
                        for g in optimizer.param_groups:
                            learning_rate = g['lr']
                        for g in optimizer.param_groups:
                            g['lr'] = 4 * learning_rate
                        early_stopping.counter = -patience_half * 2 - int(patience / 2)
                print(theta_hat)

        val_losses_test.append(val_acu_test / batch_num)
        if (val_acu_test / batch_num) == np.min(val_losses_test): model_optim = model
        lr_scheduler(val_acu_test / batch_num)
        early_stopping(val_acu_test / batch_num)
        #print(theta_hat)
        if early_stopping.early_stop:
            break

    return losses_test, val_losses_test, model_optim, theta_hat

#### Evaluation #######################################################################################################
def RMSE(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    n = x.shape[0]
    return(np.sqrt(np.sum(np.square(x-y))/n))

def PDP(model, X, rand_s):
    n = X.shape[0]
    p = X.shape[1]
    N = rand_s.shape[0]
    Y = np.zeros(n)
    for i in range(n):
        X_rep = np.repeat(X[i,:].reshape(1,p), N).reshape(N,p)
        Xs_rep = np.concatenate((X_rep, rand_s), axis=1)
        Y_rep = model(torch.from_numpy(Xs_rep).float()).detach().numpy().reshape(-1)
        Y[i] = np.mean(Y_rep)
    return Y

def RMSE_model(model, X, Y, mask, coord, theta_hat0):
    theta_hat = theta_hat0.copy() ####12.15 update
    X_train = X[mask, :]
    Y_train = Y[mask]
    X_test = X[np.invert(mask), :]
    Y_test = Y[np.invert(mask)]
    residual_train = torch.from_numpy(Y_train) - model(torch.from_numpy(X_train).float()).reshape(-1)
    residual_train = residual_train.detach().numpy()
    df = pd.DataFrame(coord, columns=['x', 'y'])
    dist = distance_matrix(df.values, df.values)
    cov = make_cov(theta_hat, dist)
    theta_hat[2] = 0
    C = make_cov(theta_hat, dist)
    del df
    del dist
    residual_test = np.matmul(C[np.invert(mask), :][:, mask], np.linalg.solve(cov[mask, :][:, mask], residual_train))
    del cov
    del C
    Y_test_hat0 = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    return(RMSE(Y_test_hat0, Y_test)/RMSE(Y_test, np.mean(Y_test)))
