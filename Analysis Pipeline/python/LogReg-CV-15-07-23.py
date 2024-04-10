import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import leaves_list
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import pickle
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import plotly.express as px
from scipy.stats import t
from statsmodels.stats.multitest import fdrcorrection
from scipy.stats import chi2_contingency
import time
from scipy.spatial.distance import squareform

np.random.seed(42)


base_directory = './miss-based-HC-15-03-23'

current_directory = base_directory

pickle_in = open('nIDPs.pickle','rb')

df = pickle.load(pickle_in)

X = df[1]

binary_feats = np.empty((X.shape[1],))

binary_feats[:] = False

for j in range(X.shape[1]):
    if np.all(np.logical_or(np.isnan(X[:,j]) , np.logical_or(X[:,j]==0 , X[:,j]==1))):
        binary_feats[j] = True
        
pickle_in = open('nIDPs_dist_mat.pickle','rb')

df = pickle.load(pickle_in)

miss_based_dist = 1- df[1]/df[1].max()

N_clus = 4

N_feats = miss_based_dist.shape[0]

clus_miss_based = AgglomerativeClustering(n_clusters = N_clus , affinity = 'precomputed' , linkage = 'complete')

clus_miss_based.fit(miss_based_dist)

no_miss_clus = 3

X_no_missing = X[:, clus_miss_based.labels_== no_miss_clus]

X_no_missing =  X_no_missing[: , X_no_missing.std(axis = 0) != 0]


binary_feats_no_missing = np.empty((X_no_missing.shape[1],))

binary_feats_no_missing[:] = False

for j in range(X_no_missing.shape[1]):
    if np.all(np.logical_or(np.isnan(X_no_missing[:,j]) , np.logical_or(X_no_missing[:,j]==0 , X_no_missing[:,j]==1))):
        binary_feats_no_missing[j] = True


X_no_missing[: , binary_feats_no_missing == False] = SimpleImputer(strategy = 'mean').fit_transform(X_no_missing[: , binary_feats_no_missing == False])

X_no_missing[: , binary_feats_no_missing == True] = SimpleImputer(strategy = 'most_frequent').fit_transform(X_no_missing[: , binary_feats_no_missing == True])

X_no_missing[: , binary_feats_no_missing == False] = (X_no_missing[: , binary_feats_no_missing == False]- X_no_missing[: , binary_feats_no_missing == False].mean(axis = 0))/X_no_missing[: , binary_feats_no_missing == False].std(axis = 0)

clus_block_dict = {}

for k in [1,2]:
    if k != no_miss_clus:
        clus_block_dict[k] = np.isnan(X[:, clus_miss_based.labels_== k]).mean(axis = 1) > 0.90
        
        
N_vals = 13

log_lambda_min = 1.0

log_lambda_max = 7.0

for clus in [1,2]:

    log_lambda_vals = log_lambda_min + (log_lambda_max-log_lambda_min)*np.array(range(N_vals))/(N_vals-1)
    print('Cluster '+str(clus)+':')
    auc_scores = np.empty((N_vals,))
    num_vars = np.empty((N_vals,))
    for i in range(N_vals):
        log_lambda = log_lambda_vals[i]
        print('log-lambda = '+str(log_lambda))
        my_logreg = LogisticRegression(C = np.exp(-log_lambda) , verbose = 1 , penalty = 'l1' , solver = 'liblinear')
        auc_scores[i] = cross_val_score(estimator = my_logreg , X=X_no_missing , y = clus_block_dict[clus] , cv = 5 , scoring = 'roc_auc').mean()
        num_vars[i] = (LogisticRegression(C = np.exp(-log_lambda) , verbose = 0 , penalty = 'l1' , solver = 'liblinear').fit(X_no_missing, clus_block_dict[clus]).coef_ > 0).sum()
    print('Max AUC is: ' + str(auc_scores.max()))
    img = plt.plot(log_lambda_vals , auc_scores , 'bo')
    plt.xlabel('log-lambda')
    plt.ylabel('5-Fold CV AUC Score')
    plt.axvline(x = 6 , color = 'red' , ls = '--')
    plt.title('Plot of CV AUC Scores for LASSO Logistic Regression, Cluster c = '+str(clus), fontweight ="bold")
    fig = plt.gcf()
    fig.savefig(base_directory + '/scatter-plots/CV-AUC-Score-clus'+ str(clus)+'.png' , bbox_inches = 'tight', dpi = 300)
    plt.show()
    plt.clf()
    plt.cla()
    img = plt.plot(log_lambda_vals , num_vars , 'b-')
    plt.xlabel('log-lambda')
    plt.ylabel('Non-Zero Parameters')
    plt.axvline(x = 6 , color = 'red' , ls = '--')
    plt.title('Plot of Number of Non-Zero Parameters for LASSO Logistic Regression, Cluster c = '+str(clus), fontweight ="bold")
    fig = plt.gcf()
    fig.savefig(base_directory + '/scatter-plots/num-non-zero-clus'+ str(clus)+'.png' , bbox_inches = 'tight', dpi = 300)
    plt.show()
    plt.clf()
    plt.cla()
    pd.DataFrame(log_lambda_vals , columns= ['log_lambda']).to_csv(base_directory + '/scatter-plots/log_lambda_vals_clus'+str(clus)+'.csv')
    pd.DataFrame(auc_scores , columns= ['auc']).to_csv(base_directory + '/scatter-plots/auc_scores_clus'+str(clus)+'.csv')
    pd.DataFrame(num_vars , columns= ['num_vars']).to_csv(base_directory + '/scatter-plots/num_vars_clus'+str(clus)+'.csv')
