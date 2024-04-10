import numpy as np
import pandas as pd
import pickle


my_directory = ''

my_df = pickle.load(open(my_directory+'X_nIDPs_analysis_comp.pkl','rb'))[None]

no_nans_rows = np.isnan(my_df).sum(axis = 1) == 0

my_df_imp = my_df[no_nans_rows == True  , :]

print(my_df_imp.shape[0])

my_dict = {None:my_df_imp}

pickle.dump(my_dict , open(my_directory + 'X_imp_CompCase.pkl', 'wb') , protocol = 4)
