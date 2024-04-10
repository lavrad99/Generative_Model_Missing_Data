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
import gzip
import warnings

warnings.filterwarnings("error")



k_preds = 150

my_directory = ''

my_corr_mat = pickle.load(open(my_directory+'raw_corr_mat.pkl','rb'))[None]

my_df = pickle.load(open(my_directory+'X_nIDPs_analysis_comp.pkl','rb'))[None]

my_pred_mat = CreateSmartPredMat(n_preds_cont=k_preds, n_preds_bin=k_preds,
                                 selection_method='corr', precomp_corr_mat=my_corr_mat).fit(my_df).pred_mat_

my_init_imp = SimpleImputer(strategy='mean').fit_transform(my_df)

my_df_imp = BespokeIterImp(pred_mat = my_pred_mat , init_imp=my_init_imp ,
                           n_iter=1 ,verbose=True , random_pred=False, fuzzy_pred_bin=True).fit_transform(my_df)

my_dict = {None:my_df_imp}

pickle.dump(my_dict , open(my_directory + 'X_imp_IterativeImputer.pkl', 'wb') , protocol = 4)
