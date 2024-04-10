import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from pyreadr import read_r
import pickle
from sklearn.preprocessing import StandardScaler

my_directory = ''

nIDP_names_analysis = pd.read_csv(my_directory +'nIDP_names_analysis.csv').drop('Unnamed: 0' , axis = 1).to_numpy().reshape(-1)

nIDP_numbers_analysis = pd.read_csv(my_directory +'nIDP_numbers_analysis.csv').drop('Unnamed: 0' , axis = 1).to_numpy().reshape(-1)

y_IDP_comp = pd.read_csv(my_directory +'y_IDP_comp.csv').drop('Unnamed: 0' , axis = 1).to_numpy()

X_imp_IterativeImputer_scaled = StandardScaler().fit_transform(pickle.load(open(my_directory+'X_imp_IterativeImputer.pkl','rb'))[None])

n_vars = 15

low_lim = np.log(0.001)

high_lim = np.log(0.1)

while True:
    print(np.exp((low_lim + high_lim)/2))
    my_net = Lasso(alpha=np.exp((low_lim + high_lim)/2) , max_iter=5000)
    my_net.fit(X_imp_IterativeImputer_scaled , y_IDP_comp)
    n_select_vars = np.unique([my_str.split('-')[0] for my_str in nIDP_numbers_analysis[my_net.coef_ != 0]]).shape[0]
    if n_select_vars == n_vars:
        break
    if n_select_vars > n_vars:
        low_lim = (low_lim + high_lim)/2
        continue
    if n_select_vars < n_vars:
        high_lim = (low_lim + high_lim)/2
        continue  
    

pd.DataFrame(nIDP_names_analysis[my_net.coef_ != 0]).to_csv(my_directory + 'selected_var_names_IterativeImputer_LASSO_'+str(n_vars)+'.csv')

pd.DataFrame(nIDP_numbers_analysis[my_net.coef_ != 0]).to_csv(my_directory + 'selected_var_numbers_IterativeImputer_LASSO_'+str(n_vars)+'.csv')

my_vars = np.unique([my_str.split('-')[0] for my_str in nIDP_numbers_analysis[my_net.coef_ != 0]])

pd.DataFrame(my_vars).to_csv(my_directory + 'selected_var_numbers_unique_IterativeImputer_LASSO_'+str(n_vars)+'.csv')
