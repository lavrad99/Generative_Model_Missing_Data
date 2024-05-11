import numpy as np
import mat73
import scipy
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
import torch
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet

my_directory = './miss-based-HC-10-10-23/'

df = mat73.loadmat('workspace13d.mat')

X_nIDPs = np.concatenate((np.concatenate((df['vars'] , df['sex12'].reshape(df['sex12'].shape[0],1)) , axis = 1) ,
                          df['age1'].reshape(df['age1'].shape[0],1)**2) , axis = 1)

#We make up names for our age and sex variables

nIDP_names =np.concatenate((np.array(df['varsHeader']) , np.array(['sex','age_squared'])))

#We make up names for our age and sex variables

nIDP_numbers = np.concatenate((np.array(df['varsVARS']) , np.array(['0-0.0','1-0.0'])))

miss_thresh = 0.4

vars_analysis_indic =  np.logical_and(np.isnan(X_nIDPs).mean(axis = 0) <= miss_thresh , np.nanstd(X_nIDPs , axis = 0) > 0)

X_nIDPs_analysis = X_nIDPs[: , vars_analysis_indic]

nIDP_names_analysis = nIDP_names[vars_analysis_indic]

nIDP_numbers_analysis = nIDP_numbers[vars_analysis_indic]

pd.DataFrame(nIDP_names_analysis).to_csv(my_directory +'nIDP_names_analysis.csv')

pd.DataFrame(nIDP_numbers_analysis).to_csv(my_directory +'nIDP_numbers_analysis.csv')

y_IDP = np.log(np.array(df['IDPs1'])[: , (np.array(df['IDP_names']) == 'IDP_T1_SIENAX_grey_normalised_volume').reshape(-1)])

y_missing_inds = np.isnan(y_IDP)

y_IDP_comp = y_IDP[y_missing_inds == False]

X_nIDPs_analysis_comp = X_nIDPs_analysis[y_missing_inds.reshape(-1) == False , :]

my_dict = {None:X_nIDPs_analysis_comp}
pickle.dump(my_dict , open(my_directory + 'X_nIDPs_analysis_comp'+'.pkl', 'wb') , protocol = 4)

pd.DataFrame(X_nIDPs_analysis_comp).to_csv(my_directory +'X_nIDPs_analysis_comp.gz')

pd.DataFrame(y_IDP_comp).to_csv(my_directory +'y_IDP_comp.csv')

raw_corr_mat = pd.DataFrame(X_nIDPs_analysis_comp).corr().to_numpy()

raw_corr_mat = SimpleImputer(strategy = 'constant' , fill_value = 0).fit_transform(raw_corr_mat)

my_dict = {None:raw_corr_mat}

pickle.dump(my_dict , open(my_directory + 'raw_corr_mat'+'.pkl', 'wb') , protocol = 4)
