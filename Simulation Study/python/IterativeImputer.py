from FastIterativeImputation import check_if_binary
from FastIterativeImputation import MixedPredictor
from FastIterativeImputation import CreateSmartPredMat
from FastIterativeImputation import BespokeIterImp
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from pyreadr import read_r
from sklearn.impute import SimpleImputer
from random import sample
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
import os
import pickle
from scipy.spatial.distance import squareform
from copy import copy
from scipy.stats import norm
from scipy.integrate import quadrature
from random import sample
from numpy.random import binomial
from numpy.random import normal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import gzip
#import warnings

#warnings.filterwarnings("error")

my_directory = './miss-based-HC-10-10-23/'

my_result_directory = './miss-based-HC-10-10-23/results/'

env_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

index = 1
for k_temp in [10,50,150]:
    for my_method_temp in ['corr' , 'miss' , 'mix']:
        for set_id_temp in range(1,21):
            if index == env_id:
                my_method = my_method_temp
                set_id = set_id_temp
                k = k_temp
            index += 1

X_miss = read_r(my_directory + 'X_miss_'+str(set_id)+'.rds')[None].to_numpy()

X_miss_MCAR_B = read_r(my_directory + 'X_miss_MCAR_B_'+str(set_id)+'.rds')[None].to_numpy()

X_full = read_r(my_directory + 'X_full_'+str(set_id)+'.rds')[None].to_numpy()


def median_imputer(X_miss):
    X_ret = copy(X_miss)
    for j in range(X_miss.shape[1]):
        X_ret[np.isnan(X_miss[: , j]) , j] = np.nanmedian(X_miss[: , j])
        
    return X_ret

analysis_clus = [2,3]


M = np.isnan(X_miss)





my_pred_mat_obj = CreateSmartPredMat(n_preds_cont=k , n_preds_bin=k , selection_method = my_method).fit(X_miss)

my_init = median_imputer(X_miss)

X_imp = BespokeIterImp(pred_mat=my_pred_mat_obj.pred_mat_ , init_imp=my_init, n_iter=3).fit_transform(X_miss)

for clus in analysis_clus:

    bin_inds_clus = read_r(my_directory + 'bin_inds_clus'+str(clus)+'.rds')[None].to_numpy().reshape(-1).astype('int')-int(1)

    cont_inds_clus = read_r(my_directory + 'cont_inds_clus'+str(clus)+'.rds')[None].to_numpy().reshape(-1).astype('int')-int(1)


    bias_imp_FW_clus = []

    sd_imp_FW_clus = []

    auc_imp_FW_clus = []

    RMSE_imp_FW_clus = []



    for j in bin_inds_clus:
        auc_imp_FW_clus = auc_imp_FW_clus + [balanced_accuracy_score(X_full[: , j][M[:,j] == 1] , X_imp[: , j][M[:,j] == 1])]

    for j in cont_inds_clus:
        RMSE_imp_FW_clus = RMSE_imp_FW_clus + [mean_squared_error(X_full[: , j][M[:,j] == 1] , X_imp[: , j][M[:,j] == 1])**0.5]
        sd_imp_FW_clus = sd_imp_FW_clus + [X_imp[: , j][M[:,j] == 1].std()]
        bias_imp_FW_clus = bias_imp_FW_clus + [(X_full[: , j][M[:,j] == 1] - X_imp[: , j][M[:,j] == 1]).mean()]

    dict_bias = {None:bias_imp_FW_clus}
    pickle.dump(dict_bias , open(my_result_directory + 'bias_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_sd = {None:sd_imp_FW_clus}
    pickle.dump(dict_sd , open(my_result_directory + 'sd_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_auc = {None:auc_imp_FW_clus}
    pickle.dump(dict_auc , open(my_result_directory + 'auc_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_RMSE = {None:RMSE_imp_FW_clus}
    pickle.dump(dict_RMSE , open(my_result_directory + 'RMSE_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))


M = np.isnan(X_miss_MCAR_B)


my_pred_mat_obj = CreateSmartPredMat(n_preds_cont=k , n_preds_bin=k , selection_method = my_method).fit(X_miss_MCAR_B)

my_init = median_imputer(X_miss_MCAR_B)

X_imp = BespokeIterImp(pred_mat=my_pred_mat_obj.pred_mat_ , init_imp=my_init, n_iter=3).fit_transform(X_miss_MCAR_B)

for clus in analysis_clus:


    bin_inds_clus = read_r(my_directory + 'bin_inds_clus'+str(clus)+'.rds')[None].to_numpy().reshape(-1).astype('int')-int(1)

    cont_inds_clus = read_r(my_directory + 'cont_inds_clus'+str(clus)+'.rds')[None].to_numpy().reshape(-1).astype('int')-int(1)


    bias_imp_FW_clus = []

    sd_imp_FW_clus = []

    auc_imp_FW_clus = []

    RMSE_imp_FW_clus = []



    for j in bin_inds_clus:
        auc_imp_FW_clus = auc_imp_FW_clus + [balanced_accuracy_score(X_full[: , j][M[:,j] == 1] , X_imp[: , j][M[:,j] == 1])]

    for j in cont_inds_clus:
        RMSE_imp_FW_clus = RMSE_imp_FW_clus + [mean_squared_error(X_full[: , j][M[:,j] == 1] , X_imp[: , j][M[:,j] == 1])**0.5]
        sd_imp_FW_clus = sd_imp_FW_clus + [X_imp[: , j][M[:,j] == 1].std()]
        bias_imp_FW_clus = bias_imp_FW_clus + [(X_full[: , j][M[:,j] == 1] - X_imp[: , j][M[:,j] == 1]).mean()]

    dict_bias = {None:bias_imp_FW_clus}
    pickle.dump(dict_bias , open(my_result_directory + 'MCAR_B_bias_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_sd = {None:sd_imp_FW_clus}
    pickle.dump(dict_sd , open(my_result_directory + 'MCAR_B_sd_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_auc = {None:auc_imp_FW_clus}
    pickle.dump(dict_auc , open(my_result_directory + 'MCAR_B_auc_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))

    dict_RMSE = {None:RMSE_imp_FW_clus}
    pickle.dump(dict_RMSE , open(my_result_directory + 'MCAR_B_RMSE_imp_clus'+str(clus)+'_FW_'+my_method+'IterativeImputer_' + str(set_id) +'_'+str(k) + '.pkl', 'wb'))
