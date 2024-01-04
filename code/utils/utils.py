#### 20230928
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import scipy
from scipy.spatial import distance_matrix
from scipy.optimize import *
from scipy.sparse import csr_array
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torch.utils.data.dataset import random_split
import time
import logging
import torch_sparse

def solve_I_B_sparseB(I_B, y):
    y = y.reshape(-1)
    n = y.shape[0]
    x = np.empty(0)
    Indlist = I_B.Ind_list[:,1:]
    B = -I_B.B[:,1:]
    for i in range(n):
        ind = Indlist[i,:]
        id = ind>=0
        if len(id) == 0:
            x = np.append(x,y[i])
            continue
        x = np.append(x, y[i] + np.dot(x[ind[id]], B[i,:][id]))
    return x

def rmvn(m, mu, cov, I_B, F_diag, sparse, chol = True):
    p = len(mu)
    if p <= 2000 and chol:
        D = np.linalg.cholesky(cov)
        res = np.matmul(np.random.randn(m, p), np.matrix.transpose(D)) + mu
    elif sparse:
        res = solve_I_B_sparseB(I_B, np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1))
        #res = scipy.sparse.linalg.spsolve_triangular(I_B, np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1))
        #res = scipy.linalg.solve_triangular(I_B.to_dense(), np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1), lower=True)
    else:
        res = scipy.linalg.solve_triangular(I_B, np.sqrt(F_diag) * np.random.randn(m, p).reshape(-1),lower=True)
    return  res.reshape(-1) # * np.ones((m, p))

def make_cov(theta, dist):
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    if isinstance(dist, float) or isinstance(dist, int):
        n = 1
    else:
        n = dist.shape[0]
    cov = sigma_sq * np.exp(-phi * dist) #+ tau_sq * np.eye(n)
    return (cov)

def make_rank(coord, nn, coord_test = None):

    knn = NearestNeighbors(n_neighbors=nn)
    knn.fit(coord)
    if coord_test is None:
        coord_test = coord
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 1:]
    else:
        rank = knn.kneighbors(coord_test)[1]
        return rank[:, 0:]

class Sparse_B():
    def __init__(self, B, Ind_list):
        self.B = B
        self.n = B.shape[0]
        self.Ind_list = Ind_list.astype(int)

    def to_numpy(self):
        if torch.is_tensor(self.B):
           self.B = self.B.detach().numpy()
        return self

    def to_tensor(self):
        if isinstance(self.B, np.ndarray):
            self.B = torch.from_numpy(self.B).float()
        return self

    def matmul(self, X, idx = None):
        if idx == None: idx = np.array(range(self.n))
        if torch.is_tensor(X):
            self.to_tensor()
            result = torch.empty((len(idx)))
            for k in range(len(idx)):
                i = idx[k]
                ind = self.Ind_list[i,:][self.Ind_list[i,:] >= 0]
                result[k] = torch.dot(self.B[i,range(len(ind))].reshape(-1),X[ind])
        elif isinstance(X, np.ndarray):
            self.to_numpy()
            if np.ndim(X) == 1:
                result = np.empty((len(idx)))
                for k in range(len(idx)):
                    i = idx[k]
                    ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
                    result[k] = np.dot(self.B[i, range(len(ind))].reshape(-1), X[ind])
            elif np.ndim(X) == 2:
                result = np.empty((len(idx), X.shape[1]))
                for k in range(len(idx)):
                    i = idx[k]
                    ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
                    #result[i,:] = np.dot(self.B[i, range(len(ind))].reshape(-1), C_Ni[ind, :])
                    result[k,:] = np.dot(self.B[i, range(len(ind))].reshape(-1), X[ind,:])
        return(result)

    def Fmul(self, F):
        temp = Sparse_B(self.B.copy(), self.Ind_list.copy())
        for i in range(self.n):
            temp.B[i,:] = F[i]*self.B[i,:]
        return(temp)

    def to_dense(self):
        B = np.zeros((self.n, self.n))
        for i in range(self.n):
            ind = self.Ind_list[i, :][self.Ind_list[i, :] >= 0]
            if len(ind) == 0:
                continue
            B[i, ind] = self.B[i,range(len(ind))]
        return(B)

def make_bf_dense(coord, rank, theta):
    n = coord.shape[0]
    k = rank.shape[1]
    B = np.zeros((n, n))
    F = np.zeros(n)
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq*np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            B[i, ind] = bi
            F[i] = F[i] - np.inner(cov_vec, bi)

    I_B = np.eye(n) - B
    return I_B, F

def make_bf_sparse(coord, rank, theta):
    n = coord.shape[0]
    k = rank.shape[1]
    B = np.zeros((n, k))
    ind_list = np.zeros((n, k)).astype(int) - 1
    F = np.zeros(n)
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            B[i, range(len(ind))] = bi
            ind_list[i, range(len(ind))] = ind
            F[i] = F[i] - np.inner(cov_vec, bi)

    I_B = Sparse_B(np.concatenate([np.ones((n, 1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis=1))

    return I_B, F

def make_bf_sparse_new(coord, rank, theta):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    F = np.zeros(n)
    row_indices = np.empty(0)
    col_indices = np.empty(0)
    values = np.empty(0)
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            F[i] = F[i] - np.inner(cov_vec, bi)
            row_indices = np.append(row_indices, np.repeat(i, len(ind)))
            col_indices = np.append(col_indices, ind)
            values = np.append(values, -bi)

    row_indices = np.append(row_indices, np.array(range(n)))
    col_indices = np.append(col_indices, np.array(range(n)))
    values = np.append(values, np.ones(n))
    I_B = csr_array((values, (row_indices.astype(int), col_indices.astype(int))))
    return I_B, F

def sparse_decor(coord, nn, theta):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    rank = make_rank(coord, nn)
    F = np.zeros(n)
    row_indices = np.empty(0)
    col_indices = np.empty(0)
    values = np.empty(0)
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            F[i] = np.sqrt(np.reciprocal(F[i]))
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            F[i] = np.sqrt(np.reciprocal(F[i] - np.inner(cov_vec, bi)))
            row_indices = np.append(row_indices, np.repeat(i, len(ind)))
            col_indices = np.append(col_indices, ind)
            values = np.append(values, -bi * F[i])

    row_indices = np.append(row_indices, np.array(range(n)))
    col_indices = np.append(col_indices, np.array(range(n)))
    l = row_indices.shape[0]
    values = np.append(values, F)
    FI_B = [np.concatenate([row_indices.reshape(1, l), col_indices.reshape(1, l)], axis=0).astype(int), values]
    #FI_B = torch.sparse_csr_tensor(torch.from_numpy(row_indices).int(),
    #                            torch.from_numpy(col_indices).int(),
    #                            torch.from_numpy(values).float())

    return FI_B

def sparse_decor_sparseB(coord, nn, theta):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    rank = make_rank(coord, nn)
    ind_list = np.zeros((n, nn)).astype(int) - 1
    B = np.zeros((n, nn))
    F = np.zeros(n)
    values = np.empty(0)
    for i in range(n):
        F[i] = make_cov(theta, 0) + tau_sq
        ind = rank[i, :][rank[i, :] <= i]
        if len(ind) == 0:
            F[i] = np.sqrt(np.reciprocal(F[i]))
            continue
        cov_sub = make_cov(theta, distance(coord[ind, :], coord[ind, :])) + tau_sq * np.eye(len(ind))
        if np.linalg.matrix_rank(cov_sub) == cov_sub.shape[0]:
            cov_vec = make_cov(theta, distance(coord[ind, :], coord[i, :])).reshape(-1)
            bi = np.linalg.solve(cov_sub, cov_vec)
            F[i] = np.sqrt(np.reciprocal(F[i] - np.inner(cov_vec, bi)))
            B[i, range(len(ind))] = bi * F[i]
            ind_list[i, range(len(ind))] = ind

    FI_B = Sparse_B(np.concatenate([(np.ones(n)*F).reshape((n,1)), -B], axis=1),
                   np.concatenate([np.arange(0, n).reshape(n, 1), ind_list], axis=1))

    return FI_B

def distance(coord1, coord2):
    if coord1.ndim == 1:
        m = 1
        p = coord1.shape[0]
        coord1 = coord1.reshape((1, p))
    else:
        m = coord1.shape[0]
    if coord2.ndim == 1:
        n = 1
        p = coord2.shape[0]
        coord2 = coord2.reshape((1, p))
    else:
        n = coord2.shape[0]

    dists = np.zeros((m, n))
    for i in range(m):
        dists[i, :] = np.sqrt(np.sum((coord1[i] - coord2) ** 2, axis=1))
    return(dists)

def bf_from_theta(theta, coord, nn, method = '0', nu = 1.5, sparse = True, version = 'sparseB'):
    n = coord.shape[0]
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq
    cov = 0
    if n<= 2000:
        dist = distance(coord, coord)
        if method == '0':
            cov = sigma_sq * np.exp(-phi * dist) + tau_sq * np.eye(n)
        elif method == '1':
            cov = sigma_sq * pow((dist * phi), nu) / (pow(2, (nu - 1)) * scipy.special.gamma(nu)) * \
                  scipy.special.kv(nu,dist * phi)
            cov[range(n), range(n)] = sigma_sq + tau_sq
    rank = make_rank(coord, nn)
    if sparse and version == 'new':
        I_B, F_diag = make_bf_sparse_new(coord, rank, theta)
    elif sparse and version == 'sparseB':
        I_B, F_diag = make_bf_sparse(coord, rank, theta)
    else:
        I_B, F_diag = make_bf_dense(coord, rank, theta)
        I_B = torch.from_numpy(I_B)
    F_diag = torch.from_numpy(F_diag)

    return I_B, F_diag, rank, cov

def Simulate_NNGP(n, p, fx, nn, theta, method = '0', nu = 1.5, a = 0, b = 1, sparse = True, meanshift = False):
    #n = 1000
    coord = np.random.uniform(low = a, high = b, size=(n, 2))
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    I_B, F_diag, rank, cov = bf_from_theta(theta, coord, nn, method = method, nu = nu, sparse = sparse)

    X = np.random.uniform(size=(n, p))
    corerr = rmvn(1, np.zeros(n), cov, I_B, F_diag, sparse)
    if meanshift:
        corerr = corerr - np.mean(corerr)

    Y = fx(X).reshape(-1) + corerr + np.sqrt(tau_sq) * np.random.randn(n)

    return X, Y, I_B, F_diag, rank, coord, cov, corerr

def Simulate_mis(n, p, fx, nn, theta, corerr_gen, a=0, b=1):
    # n = 1000
    coord = np.random.uniform(low=a, high=b, size=(n, 2))
    sigma_sq, phi, tau = theta
    tau_sq = tau * sigma_sq

    corerr = corerr_gen(coord)
    #corerr = corerr_gen(coord/(b-a))

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

class Netp(torch.nn.Module):
    def __init__(self, p, k = 50):
        super(Netp, self).__init__()
        self.l1 = torch.nn.Linear(p, k)
        self.l2 = torch.nn.Linear(k, 1)
        #self.l3 = torch.nn.Linear(10, 1)

    def forward(self, x, edge_idx = 0):
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

def fx_l(x): return 5*x + 2

def fx_c(x): return 0*x

def fx4(X): return 10 * np.sin(4 * np.pi * X)

#def f3(X): return (X[:, 0] + pow(X[:, 1], 1) + 2 * X[:, 2] + 2)
def f3(X): return ((X[:,2]+1)*np.sin(np.pi*X[:,0]*X[:,1]))

def f5(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] +5*X[:,4])/6

def f15(X): return (10*np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2]-0.5)**2 + 10*X[:,3] + 5*X[:,4] +
                    3/(X[:,5]+1)/(X[:,6]+1) + 4*np.exp(np.square(X[:,7])) + 30*np.square(X[:,8])*X[:,9] +
                    5*(np.exp(np.square(X[:,10]))*np.sin(np.pi*X[:,11]) + np.exp(np.square(X[:,11]))*np.sin(np.pi*X[:,10])) +
                    10*np.square(X[:,12])*np.cos(np.pi*X[:,13]) + 20*np.square(X[:,14]))/6

# R ####################################################################################################################
def import_BRISC():
    BRISC = importr('BRISC')
    return BRISC


BRISC = import_BRISC()


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

    if X is None:
        res = BRISC.BRISC_estimation(coord_r, residual_r)
    else:
        Xr = robjects.FloatVector(X.transpose().reshape(-1))
        Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
        res = BRISC.BRISC_estimation(coord_r, residual_r, Xr)

    theta_hat = res[9]
    beta = res[8]
    beta = np.array(beta)
    theta_hat = np.array(theta_hat)
    phi = theta_hat[2]
    tau_sq = theta_hat[1]
    sigma_sq = theta_hat[0]
    theta_hat[1] = phi
    theta_hat[2] = tau_sq / sigma_sq

    return beta, theta_hat


def RF_prediction(X, Y, coord, X_MISE):
    Xr = robjects.FloatVector(X.transpose().reshape(-1))
    Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
    Y_r = robjects.FloatVector(Y)

    X_MISE_r = robjects.FloatVector(X_MISE.detach().numpy().transpose().reshape(-1))
    X_MISE_r = robjects.r['matrix'](X_MISE_r, ncol=X.shape[1])

    res = RF.randomForest(Xr, Y_r)
    predict = robjects.r['predict'](res, X_MISE_r)
    del res
    predict = torch.from_numpy(np.array(predict))
    return predict

#RFGLS_estimate_spatial_r = robjects.r('''function(coord, Y, X){library(RandomForestsGLS);
#.Random.seed = as.integer(1); RFGLS_estimate_spatial(coord, Y, X)}''')

robjects.globalenv['.Random.seed'] = 1

def RFGLS_prediction(X, Y, coord, X_MISE = 0, n_tree = 100, node_size = 20):
    #n_tree_r = robjects.IntVector([n_tree])
    #node_size_r = robjects.IntVector([node_size])
    Xr = robjects.FloatVector(X.transpose().reshape(-1))
    Xr = robjects.r['matrix'](Xr, ncol=X.shape[1])
    Y_r = robjects.FloatVector(Y)
    coord_r = robjects.FloatVector(coord.transpose().reshape(-1))
    coord_r = robjects.r['matrix'](coord_r, ncol=2)
    robjects.globalenv['.Random.seed'] = 1
    res = RFGLS.RFGLS_estimate_spatial(coord_r, Y_r, Xr, nthsize = node_size, ntree = n_tree)
    #res = RFGLS_estimate_spatial_r(coord_r, Y_r, Xr)
    # predict = RFGLS.RFGLS_predict(res, X_MISE_r)[1]
    # del res
    # predict = torch.from_numpy(np.array(predict))
    return res
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

MSE = torch.nn.MSELoss(reduction='mean')

def train_gen_new(model, optimizer, data, epoch_num, loss_fn = MSE,
                  patience = 20, patience_half = 10):
    def train(model, data, idx):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = torch.reshape(out, (-1,))
        loss = loss_fn(out[idx], data.y[idx])
        loss.backward()
        optimizer.step()
        return loss
    @torch.no_grad()
    def test(model, data):
        model.eval()
        pred = model(data.x, data.edge_index)
        pred = torch.reshape(pred, (-1,))
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(np.square(pred[mask] - data.y[mask]).mean())
        return accs

    lr_scheduler = LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)
    losses = []
    val_losses = []
    best_val_loss = final_test_loss = 100
    for epoch in range(1, epoch_num):
        for idx in data.train_mask_batch:
            loss = train(model, data, idx)
        train_loss, val_loss, tmp_test_loss = test(model, data)
        losses.append(train_loss)
        print(test(model, data))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = tmp_test_loss
        lr_scheduler(val_loss)
        early_stopping(val_loss)
        val_losses.append(val_loss)
        if early_stopping.early_stop:
            print('End at epoch' + str(epoch))
            break
    return epoch, val_losses, model

def decor_dense(y, FI_B_local, idx = None):
    if idx is None: idx = range(y.shape[0])
    y = y.reshape(-1)
    y_decor = torch.matmul(FI_B_local[idx,:].double(), y.double())
    return(y_decor.float())

'''
def decor_sparse_old(y, FI_B_local, idx = None):
    if idx is None: idx = range(y.shape[0])
    y = y.reshape(-1)
    n = len(idx)
    y_decor = torch.zeros(n)
    for i in range(n):
        y_decor[i] = torch.dot(FI_B_local.B[i,:], y[FI_B_local.Ind_list[i,:]])
    return(y_decor.float())

'''

def decor_sparse(y, FI_B_local, idx = None):
    n = y.shape[0]
    y = y.reshape((n, 1))
    y_decor = torch_sparse.spmm(torch.from_numpy(FI_B_local[0]), torch.from_numpy(FI_B_local[1]), n, n, y)
    #y_decor = torch.sparse.mm(FI_B_local, y)
    return(y_decor.reshape(-1))

def decor_sparse_SparseB(y, FI_B_local, idx = None):
    if idx is None: idx = range(y.shape[0])
    y = y.reshape(-1)
    n = len(idx)
    y_decor = torch.zeros(n)
    for i in range(n):
        y_decor[i] = torch.dot(FI_B_local.B[idx[i],:], y[FI_B_local.Ind_list[idx[i],:]])
    return(y_decor.float())

def undecor(y_decor, I_B_inv_local, F_diag_local):
    y = torch.matmul(I_B_inv_local,
                     torch.sqrt(F_diag_local.double()) * y_decor.double())
    return(y)

def resample_fun_sparseB (residual, coord, nn, theta):
        FI_B = sparse_decor_sparseB(coord, nn, theta)
        FI_B = FI_B.to_tensor()
        residual_decor = decor_sparse_SparseB(residual, FI_B).detach().numpy()
        rank = make_rank(coord, nn)
        I_B, F_diag = make_bf_sparse(coord, rank, theta)
        idx = torch.randperm(residual_decor.shape[0])
        residual_decor = residual_decor[idx]#*np.random.choice([-1,1], residual_decor.shape[0])
        res = solve_I_B_sparseB(I_B, np.sqrt(F_diag) * residual_decor.reshape(-1))
        return(res)

def resample_fun (residual, I_B, I_B_inv, F_diag, resample = 'shuffle'):
    FI_B = (I_B.T * torch.sqrt(torch.reciprocal(F_diag))).T
    residual_decor = decor_dense(residual, FI_B)
    if resample == 'shuffle':
        idx = torch.randperm(residual_decor.shape[0])
    elif resample == 'choice':
        idx = np.random.choice(residual_decor.shape[0], residual_decor.shape[0])
    residual_decor = residual_decor[idx]#*torch.from_numpy(np.random.choice([-1,1], residual_decor.shape[0]))
    return(undecor(residual_decor, I_B_inv, F_diag))

def train_decor_new(model, optimizer, data, epoch_num, theta_hat0, BF = None, sparse = None, sparseB = True,
                    loss_fn = MSE, nn = 20,
                    patience = 20, patience_half = 10,
                    Update=True, Update_method='optimization',
                    Update_init=50, Update_step=50, Update_bound=0.1,
                    Update_lr_ctrl=False
                    ):
    def train_decor(model, data, FI_B, idx, decor):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        out = torch.reshape(out, (-1,))
        loss = loss_fn(decor(out,  FI_B, idx), decor(data.y,  FI_B, idx))
        loss.backward()
        optimizer.step()
        return loss
    @torch.no_grad()
    def test(model, data):
        model.eval()
        pred = model(data.x, data.edge_index)
        pred = torch.reshape(pred, (-1,))
        accs = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            accs.append(np.square(pred[mask] - data.y[mask]).mean())
        return accs
    lr_scheduler = LRScheduler(optimizer, patience=patience_half, factor=0.5)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.00001)
    losses = []
    val_losses = []
    best_val_loss = final_test_loss = 100

    n = data.n
    coord = data.coord
    Y = data.y
    X = data.x
    theta_hat = np.array(theta_hat0.copy())

    if sparse == None and n >= 10000: sparse = True
    if sparse and sparseB:
        print('Using sparseB family!')

    if sparse:
        decor = decor_sparse_SparseB if sparseB else decor_sparse
        sparse_decor_fun = sparse_decor_sparseB if sparseB else sparse_decor
        FI_B = sparse_decor_fun(coord, nn, theta_hat)
        if sparseB: FI_B = FI_B.to_tensor()
    else:
        decor = decor_dense
        if BF != None:
            I_B, F_diag = BF
            if Update:
                knn = NearestNeighbors(n_neighbors=nn)
                knn.fit(coord)
                rank = knn.kneighbors(coord)[1][:, 1:]
                logging.warning('Theta overwritten by the BF matrix as initial value, update parameters anyway!')
        else:
            I_B, F_diag, rank, _ = bf_from_theta(theta_hat, coord, nn, sparse=sparse)
        FI_B = (I_B.T*torch.sqrt(torch.reciprocal(F_diag))).T

    for epoch in range(1, epoch_num):
        for idx in data.train_mask_batch:
            loss_local = train_decor(model, data, FI_B, idx, decor)
        train_loss, val_loss, tmp_test_loss = test(model, data)
        losses.append(train_loss)
        Y_hat = model(X.float()).reshape(-1).double()

        if (epoch >= Update_init) & (epoch % Update_step == 0) & Update:
            if Update_method == 'optimization':
                def test2(theta_hat_test):
                    sigma, phi, tau = theta_hat_test
                    tau_sq = sigma*tau
                    #dist = distance(coord, coord)
                    #cov = sigma * (np.exp(-phi * dist) + tau * np.eye(n))  # need dist, n

                    Y_hat_local = Y_hat
                    err = (Y_hat_local - Y).detach().numpy()
                    term1 = 0
                    term2 = 0
                    for i in range(n):
                        ind = rank[i, :][rank[i, :] <= i]
                        id = np.append(ind, i)

                        #sub_cov = cov[ind, :][:, ind]
                        #sub_vec = cov[ind, i]
                        sub_cov = make_cov(theta_hat_test, distance(coord[ind, :], coord[ind, :])) + tau_sq*np.eye(len(ind))
                        sub_vec = make_cov(theta_hat_test, distance(coord[i, :], coord[ind, :])).reshape(-1)
                        if np.linalg.det(sub_cov):
                            bi = np.linalg.solve(sub_cov, sub_vec)
                        else:
                            bi = np.zeros(ind.shape)
                        I_B_i = np.append(-bi, 1)
                        F_i = sigma + tau_sq - np.inner(sub_vec, bi)
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
                theta_hat_new = res.x
            elif Update_method == 'BRISC':
                residual_temp = model(X).reshape(-1) - Y
                residual_temp = residual_temp.detach().numpy()
                _, theta_hat_new = BRISC_estimation(residual_temp, X.detach().numpy(), coord)

            print(theta_hat_new)
            if np.sum((theta_hat_new - theta_hat) ** 2) / np.sum((theta_hat) ** 2) < Update_bound:
                theta_hat = theta_hat_new
                if sparse: FI_B = sparse_decor_fun(coord, nn, theta_hat)
                else:
                    I_B, F_diag, rank, _ = bf_from_theta(theta_hat, coord, nn, sparse=sparse)
                    FI_B = (I_B.T*torch.sqrt(torch.reciprocal(F_diag))).T
                print('theta updated')
                if Update_lr_ctrl == True:
                    for g in optimizer.param_groups:
                        learning_rate = g['lr']
                    for g in optimizer.param_groups:
                        g['lr'] = 4 * learning_rate
                    early_stopping.counter = -patience_half * 2 - int(patience / 2)
            print(theta_hat)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            test_loss = tmp_test_loss
        lr_scheduler(val_loss)
        early_stopping(val_loss)
        val_losses.append(val_loss)
        if early_stopping.early_stop:
            print('End at epoch' + str(epoch))
            break
    #return epoch, val_losses, model
    return theta_hat, val_losses, model

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

    I_B, F_diag, rank, cov = bf_from_theta(theta_hat, coord, nn, sparse= False)
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

                    #cov_hat = make_cov(theta_hat, dist)

                    #B_hat, F_diag = make_bf(cov_hat, rank)
                    B_hat, F_diag = make_bf_dense(coord, rank, theta_hat)
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

def krig_pred(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0, q = 0.95):
    theta_hat = theta_hat0.copy() ####12.15 update
    #coord = np.concatenate([coord_train, coord_test],0)
    #mask = np.concatenate([np.repeat(True, coord_train.shape[0]), np.repeat(False, coord_test.shape[0])], 0)
    residual_train = Y_train - model(torch.from_numpy(X_train).float()).detach().numpy().reshape(-1)
    sigma_sq, phi, tau = theta_hat
    tau_sq = tau * sigma_sq
    #n = coord.shape[0]
    #n_train = coord[mask, :].shape[0]
    n_test = coord_test.shape[0]

    rank = make_rank(coord_train, nn = 20, coord_test = coord_test)

    residual_test = np.zeros(n_test)
    sigma_test = (make_cov(theta_hat, 0) + tau_sq) * np.ones(n_test)
    for i in range(n_test):
        ind = rank[i,:]
        C_N = make_cov(theta_hat, distance(coord_train[ind, :], coord_train[ind, :]))
        C_N = C_N + tau_sq * np.eye(C_N.shape[0])
        C_Ni = make_cov(theta_hat, distance(coord_train[ind, :], coord_test[i, :]))
        bi = np.linalg.solve(C_N, C_Ni)
        residual_test[i] = np.dot(bi.T, residual_train[ind])
        sigma_test[i] =  sigma_test[i] - np.dot(bi.reshape(-1), C_Ni)
        '''
        #C_N = cov[mask, :][:, mask]
            C_N = make_cov(theta_hat, distance(coord[mask,:], coord[mask,:]))
            C_N = C_N + tau_sq*np.eye(C_N.shape[0])
            #C_Ni = C[~(mask), :][:, mask]
            rank = make_rank(coord[mask,:], nn = 20)
            B, F_diag = make_bf(coord[mask,:], rank, theta_hat)
            I_B = Sparse_B(np.concatenate([np.ones((n, 1)), -B.B], axis=1),
                            np.concatenate([np.arange(0, n).reshape(n, 1), B.Ind_list], axis=1))
        
            C_Ni = make_cov(theta_hat, distance(coord[mask,:], coord[~mask,:]))
            #decor_C_Ni = (np.array(np.matmul(I_B, C_Ni)).T * np.sqrt(np.reciprocal(F_diag))).T
            #decor_residual = (np.array(np.matmul(I_B, residual_train)).T * np.sqrt(np.reciprocal(F_diag))).T
            decor_C_Ni = I_B.Fmul(F_diag).matmul(C_Ni)
            decor_residual = I_B.Fmul(F_diag).matmul(residual_train)
            if n <= 1000:
                residual_test = np.matmul(C_Ni.T, np.linalg.solve(C_N, residual_train))
                sigma = np.sqrt(make_cov(theta_hat, np.repeat(0, n_test)) -
                            np.diagonal(np.matmul(C_Ni.T, np.linalg.solve(C_N, C_Ni))))
            else:
                residual_test = np.matmul(decor_C_Ni.T, decor_residual)
                sigma = np.sqrt(make_cov(theta_hat, np.repeat(0, n_test)) -
                                np.sum(decor_C_Ni * decor_C_Ni, axis=0))
                                #np.diagonal(np.matmul(decor_C_Ni.T, decor_C_Ni)))
        '''
    p = scipy.stats.norm.ppf((1+q)/2, loc=0, scale=1)
    sigma_test = np.sqrt(sigma_test)
    del C_N
    del C_Ni
    pred = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    pred_U = pred + p*sigma_test
    pred_L = pred - p*sigma_test

    return([pred, pred_U, pred_L])


def krig_pred_fullGP(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0, q=0.95):
    theta_hat = theta_hat0.copy()
    residual_train = Y_train - model(torch.from_numpy(X_train).float()).detach().numpy().reshape(-1)
    sigma_sq, phi, tau = theta_hat
    tau_sq = tau * sigma_sq
    n_test = coord_test.shape[0]

    sigma_test = (make_cov(theta_hat, 0) + tau_sq) * np.ones(n_test)
    C_N = make_cov(theta_hat, distance(coord_train, coord_train))
    C_N = C_N + tau_sq * np.eye(C_N.shape[0])
    C_Ni = make_cov(theta_hat, distance(coord_train, coord_test))
    bi = np.linalg.solve(C_N, C_Ni)
    residual_test = np.dot(bi.T, residual_train).reshape(-1)
    p = scipy.stats.norm.ppf((1 + q) / 2, loc=0, scale=1)
    sigma_test = np.sqrt(sigma_test)
    pred = model(torch.from_numpy(X_test).float()).detach().numpy().reshape(-1) + residual_test
    pred_U = pred + p * sigma_test
    pred_L = pred - p * sigma_test

    return ([pred, pred_U, pred_L])

def RMSE_model(model, X, Y, mask, coord, theta_hat0):
    X_train = X[mask, :]
    Y_train = Y[mask]
    X_test = X[~(mask), :]
    Y_test = Y[~(mask)]
    coord_train = coord[mask, :]
    coord_test = coord[~(mask), :]
    pred = krig_pred(model, X_train, X_test, Y_train, coord_train, coord_test, theta_hat0)[0]
    return(RMSE(pred, Y_test)/RMSE(Y_test, np.mean(Y_test)))

