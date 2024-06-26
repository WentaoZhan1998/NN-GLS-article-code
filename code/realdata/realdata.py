#### This file produces results for prediction, and prediction interval for the real data
#### Include Figures S27, S29, S30, 3(b).

import os
#os.environ['R_HOME'] = {the R lib path where necessary packages are installed}
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import utils as utils
import utils_pygam

import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
import torch_geometric.transforms as T
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import copy

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

for name in ['0605', '0618', '0704']: #### "0605", "0618" "0704" corresponding to 2019.06.05, 2022.06.18 and 2022.07.04
    for split in [0, 3, 6]: #### 0 corresponds to pure random split, 3, 6 correspond to 3x3 and 6x6 block random split.
        ### Data process #######################################################################################################
        df1 = pd.read_csv('.//data//data_by_date//covariate' + name + '.csv')
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
        #columns = ['precipitation', 'temperature', 'air pressure', 'relative humidity', 'U-wind', 'V-wind',
        #           'PM 2.5', 'longitude', 'latitude']
        #df = pd.DataFrame(data=data, index=range(data.shape[0]), columns=columns)
        #df.to_csv('Normalized_PM2.5_20190605.csv')
        b = 1
        n = coord.shape[0]
        p = X.shape[1]
        X_int = np.concatenate((X, np.repeat(1, n).reshape(n, 1)), axis=1)
        ####################################################################################################################
        Netp = utils.Netp_sig

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

        RMSE_GAM = np.empty(0)
        RMSE_GAM_krig = np.empty(0)
        RMSE_GAM_latlong = np.empty(0)
        RMSE_RF = np.empty(0)
        RMSE_RF_krig = np.empty(0)
        RMSE_RFGLS = np.empty(0)
        RMSE_RFGLS_krig = np.empty(0)
        RMSE_NN = np.empty(0)
        RMSE_NNlatlong = np.empty(0)
        RMSE_NNDK = np.empty(0)
        RMSE_NN_krig = np.empty(0)
        RMSE_NNGLS_krig = np.empty(0)
        RMSE_NNGLS2_krig = np.empty(0)

        df_PI1 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])
        df_PI2 = pd.DataFrame(columns=['BRISC', 'GAM', 'GAMGLS', 'RF', 'RFGLS', 'NN', 'NNGLS'])

        for rand in range(100):
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
            data = utils.batch_gen(data, batch_size) # a
            ####################################################################################################################
            torch.manual_seed(2023 + rand)
            np.random.seed(2023 + rand)
            gam = utils_pygam.my_LinearGAM()
            gam.fit(X[~data.test_mask,], Y[~data.test_mask])
            Xspline = gam._modelmat(X[~data.test_mask, :])
            gam.my_fit(X[~data.test_mask,], Xspline, Y[~data.test_mask])
            def model_GAM(X, edge_index=0):
                if torch.is_tensor(X):
                    X = X.detach().numpy()
                return (torch.from_numpy(gam.predict(X)).reshape(-1))
            Pred_GAM = model_GAM(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
            RMSE_GAM = np.append(RMSE_GAM, RMSE(Pred_GAM, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*

            Y_hat = model_GAM(data.x, data.edge_index).reshape(-1).detach().numpy()
            residual =data.y - torch.from_numpy(Y_hat)
            residual_train = residual[~data.test_mask]
            residual_train = residual_train.detach().numpy()
            _, theta_hat_GAM = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
            Pred_GAM = utils.krig_pred(model_GAM, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                            coord[~data.test_mask,], coord[data.test_mask,], theta_hat_GAM)
            ####################################################################################################################
            torch.manual_seed(2023 + rand)
            np.random.seed(2023 + rand)
            X_coord = np.concatenate((X, coord), axis=1)
            gam = utils_pygam.my_LinearGAM()
            gam.fit(X_coord[~data.test_mask,], Y[~data.test_mask])
            Xspline = gam._modelmat(X_coord[~data.test_mask, :])
            gam.my_fit(X_coord[~data.test_mask,], Xspline, Y[~data.test_mask])
            PI_GAM_latlong = gam.confidence_intervals(X_coord[data.test_mask])

            def model_GAM(X, edge_index=0):
                if torch.is_tensor(X):
                    X = X.detach().numpy()
                return (torch.from_numpy(gam.predict(X)).reshape(-1))

            Pred_GAM_latlong = model_GAM(X_coord[data.test_mask,], edge_index).detach().numpy().reshape(-1)
            RMSE_GAM_latlong = np.append(RMSE_GAM_latlong, RMSE(Pred_GAM_latlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))  # RMSE*
            ########################################################################################################################
            torch.manual_seed(2023 + rand)
            np.random.seed(2023 + rand)
            rf = RandomForestRegressor(min_samples_split=10)
            rf.fit(X[~data.test_mask,], Y[~data.test_mask,])
            def model_RF(X, edge_index = 0):
                if torch.is_tensor(X):
                    X = X.detach().numpy()
                return(torch.from_numpy(rf.predict(X)))
            Pred_RF = model_RF(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
            RMSE_RF = np.append(RMSE_RF, RMSE(Pred_RF, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*

            Y_hat = model_RF(data.x, data.edge_index).reshape(-1).detach().numpy()
            residual =data.y - torch.from_numpy(Y_hat)
            residual_train = residual[~data.test_mask]
            residual_train = residual_train.detach().numpy()
            _, theta_hat_RF = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
            Pred_RF = utils.krig_pred(model_RF, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                            coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RF)
            ########################################################################################################################
            torch.manual_seed(2023 + rand)
            np.random.seed(2023 + rand)
            if ADDRFGLS:
                robjects.globalenv['.Random.seed'] = 1
                rfgls = utils.RFGLS_prediction(X[~data.test_mask,], Y[~data.test_mask], coord[~data.test_mask,])
                def model_RFGLS(X, edge_index = 0):
                    if torch.is_tensor(X):
                        X = X.detach().numpy()
                    X_r = utils.robjects.FloatVector(X.transpose().reshape(-1))
                    X_r = utils.robjects.r['matrix'](X_r, ncol=X.shape[1])
                    predict = utils.RFGLS.RFGLS_predict(rfgls, X_r)[1]
                    return(torch.from_numpy(np.array(predict)))
                Pred_RFGLS = model_RFGLS(X[data.test_mask,], edge_index).detach().numpy().reshape(-1)
                RMSE_RFGLS = np.append(RMSE_RFGLS, RMSE(Pred_RFGLS, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*

                Y_hat = model_RFGLS(data.x, data.edge_index).reshape(-1).detach().numpy()
                residual =data.y - torch.from_numpy(Y_hat)
                residual_train = residual[~data.test_mask]
                residual_train = residual_train.detach().numpy()
                _, theta_hat_RFGLS = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
                Pred_RFGLS = utils.krig_pred(model_RFGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                                coord[~data.test_mask,], coord[data.test_mask,], theta_hat_RFGLS)
                RMSE_RFGLS_krig = np.append(RMSE_RFGLS_krig,
                                            RMSE(Pred_RFGLS[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*
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
            torch.manual_seed(2023)
            model_NNlatlong = Netp(p + 2, k, q)
            optimizer = torch.optim.Adam(model_NNlatlong.parameters(), lr=0.1)
            patience_half = 10
            patience = 20
            data_latlong = copy.copy(data)

            data_latlong.x = torch.concatenate((data.x, torch.from_numpy(data.coord)), axis=1).float()

            _, _, model_NNlatlong = utils.train_gen_new(model_NNlatlong, optimizer, data_latlong, epoch_num=1000,
                                                           patience=patience, patience_half=patience_half)
            Pred_NNlatlong = model_NNlatlong(data_latlong.x[data_latlong.test_mask,], edge_index).detach().numpy().reshape(-1)
            RMSE_NNlatlong = np.append(RMSE_NNlatlong, RMSE(Pred_NNlatlong, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*
            ####################################################################################
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
                    d = np.linalg.norm(data.coord / b - knots[i, :], axis=1) / theta_temp
                    for j in range(len(d)):
                        if d[j] >= 0 and d[j] <= 1:
                            phi_temp[j, i + K] = (1 - d[j]) ** 6 * (35 * d[j] ** 2 + 18 * d[j] + 3) / 3
                        else:
                            phi_temp[j, i + K] = 0
                K = K + num_basis[res]

            torch.manual_seed(2023)
            model_NNDK = Netp(p + K, 100, q)
            optimizer = torch.optim.Adam(model_NNDK.parameters(), lr=0.1)
            patience_half = 10
            patience = 20
            data_DK = copy.copy(data)
            data_DK.x = torch.concatenate((data.x, torch.from_numpy(phi_temp)), axis=1).float()
            _, _, model_NNDK = utils.train_gen_new(model_NNDK, optimizer, data_DK, epoch_num=1000,
                                                      patience=patience, patience_half=patience_half)
            Pred_NNDK = model_NNDK(data_DK.x[data_DK.test_mask,], edge_index).detach().numpy().reshape(-1)
            RMSE_NNDK = np.append(RMSE_NNDK, RMSE(Pred_NNDK, Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*
            ####################################################################################################################
            Y_hat = model_NN(data.x, data.edge_index).reshape(-1).detach().numpy()
            residual =data.y - torch.from_numpy(Y_hat)
            residual_train = residual[~data.test_mask]
            residual_train = residual_train.detach().numpy()
            beta_hat, theta_hat = utils.BRISC_estimation(residual_train, X[~data.test_mask,], coord[~data.test_mask,])
            theta_hat0 = theta_hat

            Pred_NN = utils.krig_pred(model_NN, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                            coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
            RMSE_NN_krig = np.append(RMSE_NN_krig,
                                        RMSE(Pred_NN[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))#RMSE*
            ###################################################################################
            torch.manual_seed(2023)
            model_NNGLS = Netp(p, k, q)
            optimizer = torch.optim.Adam(model_NNGLS.parameters(), lr=0.1)
            patience_half = 10
            patience = 20

            _, _, _, model_NNGLS = utils.train_decor_new(model_NNGLS, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                                    Update=False, patience=patience, patience_half=patience_half)
            Pred_NNGLS = utils.krig_pred(model_NNGLS, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                            coord[~data.test_mask,], coord[data.test_mask,], theta_hat0)
            RMSE_NNGLS_krig = np.append(RMSE_NNGLS_krig,
                                        RMSE(Pred_NNGLS[0], Y_test)/RMSE(Y_test, np.mean(Y_test)))
            ###################################################################################
            torch.manual_seed(2023)
            model_NNGLS2 = Netp(p, k, q)
            optimizer = torch.optim.Adam(model_NNGLS2.parameters(), lr=0.1)
            patience_half = 10
            patience = 20

            theta_hat2, _, _, model_NNGLS2 = utils.train_decor_new(model_NNGLS2, optimizer, data, 1000, theta_hat0, sparse=Sparse,
                                                          Update=True, patience=patience, patience_half=patience_half,
                                                          Update_method='optimization', Update_init=20, Update_step=20,
                                                          Update_bound=100)
            Pred_NNGLS2 = utils.krig_pred(model_NNGLS2, X[~data.test_mask,], X[data.test_mask], Y[~data.test_mask],
                                             coord[~data.test_mask,], coord[data.test_mask,], theta_hat2)
            RMSE_NNGLS2_krig = np.append(RMSE_NNGLS2_krig,
                                         RMSE(Pred_NNGLS2[0], Y_test) / RMSE(Y_test, np.mean(Y_test)))  # RMSE*
            ####################################################################################################################
            df_PI1_temp = {'GAM': int_coverage(Y_test, Pred_GAM[1], Pred_GAM[2]),
                          'GAM_latlong': int_coverage(Y_test, PI_GAM_latlong[:, 1], PI_GAM_latlong[:, 0]),
                          'RF': int_coverage(Y_test, Pred_RF[1], Pred_RF[2]),
                          'NN': int_coverage(Y_test, Pred_NN[1], Pred_NN[2]),
                          'NNGLS': int_coverage(Y_test, Pred_NNGLS[1], Pred_NNGLS[2]),
                          'NNGLS2': int_coverage(Y_test, Pred_NNGLS2[1], Pred_NNGLS2[2])
                           }

            df_PI2_temp = {'GAM': int_score(Y_test, Pred_GAM[1], Pred_GAM[2], 0.95),
                          'GAM_latlong': int_coverage(Y_test, PI_GAM_latlong[:, 1], PI_GAM_latlong[:, 0]),
                          'RF': int_score(Y_test, Pred_RF[1], Pred_RF[2], 0.95),
                          'NN': int_score(Y_test, Pred_NN[1], Pred_NN[2], 0.95),
                          'NNGLS': int_score(Y_test, Pred_NNGLS[1], Pred_NNGLS[2], 0.95),
                          'NNGLS2': int_score(Y_test, Pred_NNGLS2[1], Pred_NNGLS2[2], 0.95)
                          }
            if ADDRFGLS:
                df_PI1_temp['RFGLS'] = int_coverage(Y_test, Pred_RFGLS[1], Pred_RFGLS[2])
                df_PI2_temp['RFGLS'] = int_score(Y_test, Pred_RFGLS[1], Pred_RFGLS[2], 0.95)
            df_PI1 = df_PI1.append(df_PI1_temp, ignore_index=True)
            df_PI1.to_csv(".//simulation//realdata//" + name + 'block' + str(split) + '_PI_cov.csv')
            df_PI2 = df_PI2.append(df_PI2_temp, ignore_index=True)
            df_PI2.to_csv(".//simulation//realdata//" + name + 'block' + str(split) + '_PI_score.csv')
            ####################################################################################################################
            df_RMSE = pd.DataFrame(
                {'GAM': RMSE_GAM,
                 'GAM_latlong': RMSE_GAM_latlong,
                 'RF': RMSE_RF,
                 'NN_latlong': RMSE_NNlatlong, 'NNDK': RMSE_NNDK,
                 'NN': RMSE_NN,
                 'NNGLS_krig': RMSE_NNGLS_krig, 'NNGLS2_krig': RMSE_NNGLS2_krig
                })
            if ADDRFGLS:
                df_RMSE['RFGLS'] = RMSE_RFGLS
                df_RMSE['RFGLS_krig'] = RMSE_RFGLS_krig
            df_RMSE.to_csv(".//simulation//realdata//RMSE" + name + 'block' + str(split) +'.csv')

