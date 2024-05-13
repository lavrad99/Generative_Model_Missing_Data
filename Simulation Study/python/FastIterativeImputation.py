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

def check_if_binary(z):
    miss_inds = np.isnan(z)
    zero_inds = z == 0
    one_inds = z == 1
    tot_covered = np.logical_or(miss_inds , np.logical_or(zero_inds , one_inds))
    if np.all(tot_covered):
        return True
    else:
        return False

class MixedPredictor(BaseEstimator):
    def __init__(self , C_LR=np.inf , random_pred = False , fuzzy_pred_bin = False, scale = True):
        self.C_LR = C_LR
        self.random_pred = random_pred
        self.fuzzy_pred_bin = fuzzy_pred_bin
        self.scale = scale
    
    def fit(self , X,y):
        X, y = check_X_y(X, y)
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)
        #check if y is binary
        self.one_class_ = False
        
        if self.scale == True:
            my_scaler = StandardScaler()
            X = my_scaler.fit_transform(X)
            self.scaler_ = my_scaler
        
        if check_if_binary(y) == True:
            #we classify
            self.binary_ = True
            if np.unique(y).shape[0] == 1:
                self.one_class_ = True
                self.one_class_value_ = y[0]
            else:
                self.fitted_predictor_ = LogisticRegression(C = self.C_LR , max_iter = 1000).fit(X,y==1)
        else:
            #we regress
            self.fitted_predictor_ = BayesianRidge().fit(X,y)
            self.binary_ = False
        return self
    def predict(self , X):
        if self.scale == True:
            X = self.scaler_.transform(X)
        if self.random_pred == False:
            check_is_fitted(self)
            X = check_array(X)
            try:
                if not self.fuzzy_pred_bin or not self.binary_:
                    if self.one_class_ == True:
                        my_pred = np.full((X.shape[0],) , self.one_class_value_)
                    else:
                        my_pred = self.fitted_predictor_.predict(X)
                else:
                    if self.one_class_ == True:
                        my_pred = np.full((X.shape[0],) , self.one_class_value_)
                    else:
                        my_pred = self.fitted_predictor_.predict_proba(X)[: , self.fitted_predictor_.classes_].reshape(-1)
            except:
                if self.binary_ == True:
                    my_pred = np.full((X.shape[0],) ,self.y_mean_ > 0.5)
                else:
                    my_pred = np.full((X.shape[0],) ,self.y_mean_)
            return my_pred
        else:
            if self.binary_ == True:
                if self.one_class_ == True:
                    return np.full((X.shape[0],) , self.one_class_value_)
                else:
                    my_probs = self.fitted_predictor_.predict_proba(X)[: , self.fitted_predictor_.classes_].reshape(-1)
                    #my_probs = self.fitted_predictor_.predict_proba(X)
                    return binomial(n = np.full(my_probs.shape , 1) , p = my_probs)
            else:
                try:
                    my_pred = self.fitted_predictor_.predict(X, return_std=True)
                except:
                    my_pred = (np.full((X.shape[0],) , self.y_mean_) , np.full((X.shape[0],) , self.y_std_))
                return normal(loc = my_pred[0] , scale = my_pred[1])


class CreateSmartPredMat:
    def __init__(self , n_preds_cont = 10 , n_preds_bin = 10 ,
                 selection_method = 'corr', precomp_corr_mat = None):
        self.n_preds_cont = n_preds_cont
        self.n_preds_bin = n_preds_bin
        self.selection_method = selection_method
        self.precomp_corr_mat = precomp_corr_mat
    def fit(self , X):
        if self.selection_method == 'mix':
            if type(self.precomp_corr_mat) == type(None):
                self.corr_mat_ = pd.DataFrame(X).corr().to_numpy()
            else:
                self.corr_mat_ = self.precomp_corr_mat
            M = np.isnan(X)
            self.V_ = np.matmul(M.T , (1-M))
            self.excess_score_mat_ = np.full((X.shape[1] , X.shape[1]) , np.nan)
            self.pred_mat_ = np.full((X.shape[1] , X.shape[1]) , False)
            self.props_ = np.full((X.shape[1],1) , np.nan)
            for j in range(X.shape[1]):
                if check_if_binary(X[: , j]):
                                self.props_[j] = np.nanmean(X[: , j])
            for j1 in range(X.shape[1]):
                for j2 in range(X.shape[1]):
                    if j1 == j2:
                        self.excess_score_mat_[j1,j2] = -np.inf
                        continue
                    if np.isnan(self.props_[j1]):
                        self.excess_score_mat_[j1,j2] = self.V_[j1,j2]*self.corr_mat_[j1,j2]**2
                    elif np.isnan(self.props_[j2]):
                        pairwise_complete_vec = np.logical_and(M[: , j1] == False , M[: , j2] == False)
                        prop_j1 = np.mean(X[: , j1][pairwise_complete_vec])
                        
                        D = norm.ppf(prop_j1)
                        rho_b = self.corr_mat_[j1,j2]*np.sqrt(prop_j1*(1-prop_j1))/norm.pdf(D)
                        integ_lim = 10
                        if np.abs(D/rho_b) > integ_lim:
                            self.excess_score_mat_[j1,j2] = 1e-15*self.V_[j1,j2]
                            continue
                            
                        
                        
                        
                        
                        if rho_b > 0:

                            integral1 = quadrature(lambda t: norm.pdf(t)*norm.cdf((D-rho_b*t)/np.sqrt(1-rho_b**2)) , a = -integ_lim , b = D/rho_b, tol = 1e-20, rtol = 1e-20)[0]
                            integral2 = quadrature(lambda t: norm.pdf(t)*(1-norm.cdf((D-rho_b*t)/np.sqrt(1-rho_b**2))) , a = D/rho_b , b = integ_lim, tol = 1e-20, rtol = 1e-20)[0]
                            
                            succ_prob_smart_strat = integral1 + integral2
                            succ_prob_dumb_strat = max(prop_j1 , 1-prop_j1)


                            self.excess_score_mat_[j1,j2] = self.V_[j1,j2]*(succ_prob_smart_strat - succ_prob_dumb_strat)
                        elif rho_b < 0: 
                            integral1 = quadrature(lambda t: norm.pdf(t)*(1-norm.cdf((D-rho_b*t)/np.sqrt(1-rho_b**2))) , a = -integ_lim , b = D/rho_b, tol = 1e-16, rtol = 1e-16)[0]
                            integral2 = quadrature(lambda t: norm.pdf(t)*norm.cdf((D-rho_b*t)/np.sqrt(1-rho_b**2)) , a = D/rho_b , b = integ_lim, tol = 1e-16, rtol = 1e-16)[0]
                            succ_prob_smart_strat = integral1 + integral2
                            succ_prob_dumb_strat = max(prop_j1 , 1-prop_j1)


                            self.excess_score_mat_[j1,j2] = self.V_[j1,j2]*(succ_prob_smart_strat - succ_prob_dumb_strat)
                        else:
                            self.excess_score_mat_[j1,j2] = 0
                    else:
                        pairwise_complete_vec = np.logical_and(M[: , j1] == False , M[: , j2] == False)
                        

                        pi_1_dot = np.mean(X[: , j1][pairwise_complete_vec])
                        pi_dot_1 = np.mean(X[: , j2][pairwise_complete_vec])
                        pi_11 = pi_1_dot*pi_dot_1 + self.corr_mat_[j1,j2]*np.sqrt(pi_1_dot*(1-pi_1_dot)*pi_dot_1*(1-pi_dot_1))

                        pi_1_given_1 = pi_11/pi_dot_1
                        pi_given_1_1 = pi_11/pi_1_dot
                        pi_given_1_0 = 1-pi_given_1_1
                        pi_1_given_0 = pi_given_1_0*pi_1_dot/(1-pi_dot_1)

                        succ_prob_smart_strat = pi_dot_1*max(pi_1_given_1 , 1-pi_1_given_1) + (1-pi_dot_1)*max(pi_1_given_0 , 1-pi_1_given_0)
                        succ_prob_dumb_strat = max(pi_1_dot , 1-pi_1_dot)

                        if succ_prob_smart_strat - succ_prob_dumb_strat < 1e-15:
                            self.excess_score_mat_[j1,j2] = 1e-15*self.V_[j1,j2]
                        else:
                            self.excess_score_mat_[j1,j2] = self.V_[j1,j2]*(succ_prob_smart_strat-succ_prob_dumb_strat)
        
            for j in range(X.shape[1]):
                if np.isnan(self.props_[j]):
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_cont)[-self.n_preds_cont:]
                    self.pred_mat_[j , my_inds] = True

                else:
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_bin)[-self.n_preds_bin:]
                    self.pred_mat_[j , my_inds] = True
        if self.selection_method == 'corr':
            if type(self.precomp_corr_mat) == type(None):
                self.excess_score_mat_ = np.abs(pd.DataFrame(X).corr().to_numpy())
            else:
                self.excess_score_mat_ = np.abs(self.precomp_corr_mat)
            for j1 in range(X.shape[1]):
                for j2 in range(X.shape[1]):
                    if j1 == j2:
                        self.excess_score_mat_[j1,j2] = -np.inf
                    if np.isnan(self.excess_score_mat_[j1,j2]):
                        self.excess_score_mat_[j1,j2] = -np.inf
            self.props_ = np.full((X.shape[1],1) , np.nan)
            for j in range(X.shape[1]):
                if check_if_binary(X[: , j]):
                    self.props_[j] = np.nanmean(X[: , j])
            self.pred_mat_ = np.full((X.shape[1] , X.shape[1]) , False)
            for j in range(X.shape[1]):
                if np.isnan(self.props_[j]):
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_cont)[-self.n_preds_cont:]
                    self.pred_mat_[j , my_inds] = True

                else:
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_bin)[-self.n_preds_bin:]
                    self.pred_mat_[j , my_inds] = True
            
        if self.selection_method == 'miss':
            M = np.isnan(X)
            self.excess_score_mat_ = np.matmul(M.T , (1-M))
            for j1 in range(X.shape[1]):
                for j2 in range(X.shape[1]):
                    if j1 == j2:
                        self.excess_score_mat_[j1,j2] = -1
            self.props_ = np.full((X.shape[1],1) , np.nan)
            for j in range(X.shape[1]):
                if check_if_binary(X[: , j]):
                    self.props_[j] = np.nanmean(X[: , j])
            self.pred_mat_ = np.full((X.shape[1] , X.shape[1]) , False)
            for j in range(X.shape[1]):
                if np.isnan(self.props_[j]):
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_cont)[-self.n_preds_cont:]
                    self.pred_mat_[j , my_inds] = True

                else:
                    my_inds = np.argpartition(self.excess_score_mat_[j , :], -self.n_preds_bin)[-self.n_preds_bin:]
                    self.pred_mat_[j , my_inds] = True
        
        return self                    
                    
                    
        
                    
                    
    
class BespokeIterImp:
    def __init__(self , pred_mat , init_imp , n_iter = 10 , C_LR = np.inf,
                 verbose = False , random_pred = False, fuzzy_pred_bin = False, scale = True):
        self.pred_mat = pred_mat
        self.init_imp = init_imp
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_pred = random_pred
        self.C_LR = C_LR
        self.fuzzy_pred_bin = fuzzy_pred_bin
        self.scale = scale
    def fit_transform(self , X):
        n_iter_cutoffs = 1000
        my_iter_cutoffs = [round(i/n_iter_cutoffs*X.shape[1]) for i in range(1,n_iter_cutoffs)]
        M = np.isnan(X)
        X_0 = self.init_imp
        X_1 = copy(X)
        for l in range(self.n_iter):
            if self.verbose == True:
                print('Starting Iteration '+str(l+1))
            for j in range(X.shape[1]):
                
                if self.verbose == True and j in my_iter_cutoffs:
                    print(str(round(100*j/X.shape[1],1))+'% of iteration completed')
                if np.all(M[: , j] == False):
                    continue
                my_imp = MixedPredictor(random_pred=self.random_pred,
                                        C_LR = self.C_LR, fuzzy_pred_bin=self.fuzzy_pred_bin,
                                       scale=self.scale).fit(X_0[: , self.pred_mat[j , :] == True][M[: , j] == False , :] , X_0[M[: , j] == False , j])
                X_1[M[: , j] == True, j] = my_imp.predict(X_0[M[: , j] == True ,:][:, self.pred_mat[j , :] == True])
            #If there is any missingness left for any reason, we impute it using median imputation
            X_0 = copy(X_1)
        return X_1
