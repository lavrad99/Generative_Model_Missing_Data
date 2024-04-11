# Generative_Model_Missing_Data
Code repository accompanying the paper "A Generative Model For Evaluating Missing Data Methods in Large Epidemiological Cohorts" by Radosavljevic et. al. The repository is split into three parts corresponding to the three subsections of the result section in the paper.

* Analysis Pipeline
  * python: miss-based-HC-10-10-23.ipynb is a jupyter notebook that contains all the results in the Analysis Pipeline section as well as additional results which did not make it into the paper. LogReg-CV-15-07-23.py is code used for tuning the penalty term in the LASSO-LR model predicting structured missingness.
* Simulation Study
  * python: FastIterativeImputation.py is our own implementation of iterative imputation that can handle three different methods of selecting a subset of imputation predictors for each variable. IterativeImputer.py is code for evaluating the performance of iterative imputation.
  * R: sim-data-11-10-23.R is code for simulating synthetic data using our generative model. binary_search_tools.R is code for tuning informativeness of missingness and sampling_rho_from_hist.R is for sampling correlation from a given pair of clusters. mean_impute.R and SoftImpute.R are evaluation code for mean imputation and SoftImpute on simulated data.
* Illustrative Example
  * python: creating_and_pickling_data.ipynb is a jupyter notebook where the 15000 nIDPs for the illustrative example are selected. Files with the prefix "creating_imp_mat_" contain code for handling missing data. Files with the prefix "feature_selection_" contain code for selecting the 15 features for each method. creating_final_data_sets-new.ipynb is a data set for creating the final data sets containing only the selected variables and confounds.
  * R: creating_imp_mat_SoftImpute.R is code for creating the imputed data matrix of real data using SoftImpute. evaluating_methods_illustrative_example.R contains code for evaluating the methods on the illustrative example.
