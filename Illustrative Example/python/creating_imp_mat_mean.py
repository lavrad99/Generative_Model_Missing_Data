import numpy as np
import pandas as pd
import pickle
from sklearn.impute import SimpleImputer

my_directory = ''

my_df = pickle.load(open(my_directory+'X_nIDPs_analysis_comp.pkl','rb'))[None]

my_df_imp = SimpleImputer(strategy = 'mean').fit_transform(my_df)

my_dict = {None:my_df_imp}

pickle.dump(my_dict , open(my_directory + 'X_imp_mean.pkl', 'wb') , protocol = 4)
