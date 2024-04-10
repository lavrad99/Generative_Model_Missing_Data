library('BinNor')
library('glmnet')
library('caret')
library('pracma')
library('stats')
library('tidyverse')

#Loading functions for the binary search procedure and sampling correlations from histograms

source('binary_search_tools.R')
source('sample_rho_from_hist.R')

set_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))

#Determining the number of rows and the clusters chosen for analysis

n_obs <- 46471

n_clus_analysis <- 2

analysis_clus <- c(2 , 3)

my_directory <- ''

#Generating important variables

corr_mat_imp_vars <- as.matrix(read_csv(
  paste(my_directory , 'corr_mat_important_vars.csv' , sep = '' )))[ , -c(1)]

X_imp_vars <- rmvnorm(n = n_obs , sigma = corr_mat_imp_vars)

n_imp_vars <- dim(X_imp_vars)[2]

#Generating sizes of clusters and number of binary variables

n_vars_tot <- n_imp_vars

n_vars_clus <- list()

for(clus in analysis_clus){
  n_vars_clus[[clus]] <- read.csv(paste(my_directory , 'n_feats_clus',
                                        as.character(clus),'.csv', sep = ''),
                                  header = F)[[1]]
  n_vars_tot <- n_vars_tot + n_vars_clus[[clus]]
}


n_bin_vars_clus <- list()

for(clus in analysis_clus){
  n_bin_vars_clus[[clus]] <- read.csv(paste(my_directory , 'n_bin_feats_clus',
                                        as.character(clus),'.csv', sep = ''),
                                  header = F)[[1]]
}


#reading in indicatior vectors for important variables for each cluster

imp_vars_list <- list()

for(clus in analysis_clus){
  imp_vars_list[[clus]] <- read.csv(paste(my_directory ,
                                        'indic_imp_vars_clus',
                                        as.character(clus), '.csv',
                                        sep = '') , header = F)
}

#Generating structured missingness for each cluster

struct_miss_list <- list()

for(clus in analysis_clus){
  prop_struct_miss <- read.csv(paste(my_directory ,
                                     'prop_block_clus' ,
                                     as.character(clus),
                                     '.csv', sep = '') , header = F)[[1]]
  my_auc <- read.csv(paste(my_directory ,
                           'AUC_clus' ,
                           as.character(clus),
                           '.csv', sep = '') , header = F)[[2,2]]
  X_imp_vars_clus <- X_imp_vars[ ,  imp_vars_list[[clus]]== 1]
  
  params_clus <- find_param(X_imp_vars_clus, prop_struct_miss,
                            my_auc)
  
  my_intercept <- find_intercept(X_imp_vars_clus, prop_struct_miss,
                                 params_clus)
  
  struct_miss_list[[clus]] <- rbinom(n = n_obs , size = 1,
                        p = inv_logit(my_intercept+X_imp_vars_clus%*%matrix(data = params_clus,
                                                                                nrow = dim(X_imp_vars_clus)[2],
                                                                                ncol = 1)))
}

#Generating unstructured missingness proportions

unstruct_miss_vecs <- list()

for(clus in analysis_clus){
  my_alpha <- read.csv(paste(my_directory ,
                             'alpha_clus' ,
                             as.character(clus),
                             '.csv', sep = '') , header = F)[[1]]
  my_beta <- read.csv(paste(my_directory ,
                            'beta_clus' ,
                            as.character(clus),
                            '.csv', sep = '') , header = F)[[1]]
  unstruct_miss_vecs[[clus]] <- rbeta(n = n_vars_clus[[clus]] , shape1 = my_alpha, shape2 = my_beta)
}

#Generating rest of correlation matrix

my_corr_mat <- matrix(nrow = n_vars_tot , ncol = n_vars_tot)

my_corr_mat[1:n_imp_vars , 1:n_imp_vars] <- corr_mat_imp_vars

#First filling out correlation between important variables and 

ind_pos <- n_imp_vars

clus_positions <- list()

for(clus in analysis_clus){
  clus_positions[[clus]] <- ind_pos + 1:n_vars_clus[[clus]]
  my_hist <- as.vector(read.csv(paste(my_directory ,
                                      'histogram_corr_between_imp_and_clus_',as.character(clus)
                                      , '.csv' ,
                                      sep = '') , header = F)[ , -c(1)])
  sub_corr_mat <- matrix(data = sample_rho_from_hist(n = n_imp_vars*n_vars_clus[[clus]], my_hist = my_hist),
                         nrow = n_imp_vars, ncol = n_vars_clus[[clus]]
                         )
  
  my_corr_mat[1:n_imp_vars , clus_positions[[clus]]] <- sub_corr_mat
  my_corr_mat[ind_pos + 1:n_vars_clus[[clus]] , 1:n_imp_vars] <- t(sub_corr_mat)
  ind_pos <- ind_pos+n_vars_clus[[clus]]
}

#Saving cluster indices

my_bin_inds <- c()

my_cont_inds <- c()

for(clus in analysis_clus){
  my_bin_inds <- c(my_bin_inds , clus_positions[[clus]][1:n_bin_vars_clus[[clus]]])
  
  saveRDS(clus_positions[[clus]][1:n_bin_vars_clus[[clus]]] ,
          paste(my_directory  , 'bin_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
  my_cont_inds <- c(my_cont_inds ,
                    clus_positions[[clus]][(n_bin_vars_clus[[clus]]+1):length(clus_positions[[clus]])])
  
  saveRDS(clus_positions[[clus]][(n_bin_vars_clus[[clus]]+1):length(clus_positions[[clus]])] ,
          paste(my_directory  , 'cont_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
}

saveRDS(my_bin_inds , paste(my_directory , 'bin_inds.rds' , sep = ''))

saveRDS(my_cont_inds , paste(my_directory , 'cont_inds.rds' , sep = ''))

#Then filling out correlations between clusters

for(i in 1:(length(analysis_clus)-1)){
  for(j in (i+1):length(analysis_clus)){
    clus1 <- analysis_clus[i]
    clus2 <- analysis_clus[j]
    my_hist <- as.vector(read.csv(paste(my_directory ,
                                        'histogram_corr_between_clus_',as.character(clus1),
                                        '_', as.character(clus2)
                                        , '.csv' ,
                                        sep = '') , header = F)[ , -c(1)])
    sub_corr_mat <- matrix(data = sample_rho_from_hist(n = n_vars_clus[[clus1]]*n_vars_clus[[clus2]], my_hist = my_hist),
                           nrow = n_vars_clus[[clus1]], ncol = n_vars_clus[[clus2]]
    )
    my_corr_mat[clus_positions[[clus1]] , clus_positions[[clus2]]] <- sub_corr_mat
    my_corr_mat[clus_positions[[clus2]] , clus_positions[[clus1]]] <- t(sub_corr_mat)
  }
}


#Finally filling out the between cluster correlations


for(clus in analysis_clus){
  my_hist <- as.vector(read.csv(paste(my_directory ,
                                      'histogram_corr_within_clus',as.character(clus)
                                      , '.csv' ,
                                      sep = '') , header = F)[ , -c(1)])
  lower_tri_mat <- sample_rho_from_hist(n = n_vars_clus[[clus]]*(n_vars_clus[[clus]]-1)/2, my_hist = my_hist)
  
  my_corr_mat[clus_positions[[clus]] , clus_positions[[clus]]] <- lower.tri.to.corr.mat(lower_tri_mat , 
                                                                                        d = n_vars_clus[[clus]])
  
}

#Making our correlation matrix positive definite

my_corr_mat_PD <- nearPD(my_corr_mat , corr = T)$mat

my_corr_mat_PD <- matrix(data = my_corr_mat_PD ,
                         nrow = dim(my_corr_mat_PD)[1],
                         ncol = dim(my_corr_mat_PD)[2])

#Generating rest of the data
my_corr_mat_PD_clus_imp <- my_corr_mat_PD[(n_imp_vars+1):n_vars_tot , 1:n_imp_vars]

my_corr_mat_PD_clus <- my_corr_mat_PD[(n_imp_vars+1):n_vars_tot , (n_imp_vars+1):n_vars_tot]

my_corr_mat_PD_imp <- my_corr_mat_PD[1:n_imp_vars , 1:n_imp_vars]

mean_adjuster <- my_corr_mat_PD_clus_imp %*% solve(my_corr_mat_PD_imp)

adjusted_means <- X_imp_vars %*% t(mean_adjuster)

adjusted_vcov_mat_clus <- my_corr_mat_PD_clus -  mean_adjuster %*% t(my_corr_mat_PD_clus_imp)

X_rest <- adjusted_means + rmvnorm(n = n_obs, sigma = as.matrix(adjusted_vcov_mat_clus))

X_full <- cbind(X_imp_vars , X_rest)

#Binarising binary variables in X_full

for(clus in analysis_clus){
  bin_inds <- clus_positions[[clus]][1:n_bin_vars_clus[[clus]]]
  thresh_mat <- t(matrix(data = rnorm(n = n_bin_vars_clus[[clus]]) ,
                         nrow = n_bin_vars_clus[[clus]] , ncol = n_obs))
  X_full[ , bin_inds] <- X_full[ , bin_inds]  > thresh_mat
}

#Adding missingness to matrix

miss_mat <- matrix(data = F , nrow = n_obs , ncol = n_vars_tot)

for(clus in analysis_clus){
  #Adding structured missingness
  
  miss_mat[struct_miss_list[[clus]] == 1 , clus_positions[[clus]]] <- T
  
  #Adding unstructured missingness
  unstruct_miss_mat <- sapply(unstruct_miss_vecs[[clus]] ,
                function(p){return(rbinom(n = n_obs , size = 1 ,
                                          prob = p))})
  miss_mat[ , clus_positions[[clus]]][unstruct_miss_mat == 1] <- T
}

X_miss <- X_full

X_miss[miss_mat] <- NA


#Creating equivalent matrix with MCAR-B missingness

X_miss_MCAR_B <- X_full

for(clus in analysis_clus){
  #Finding rate of missingness for cluster
  
  p_miss <- mean(miss_mat[ , clus_positions[[clus]]])
  
  MCAR_B_miss_mat <- matrix(data = rbinom(n = n_obs*n_vars_clus[[clus]] , size = 1, prob = p_miss),
                            nrow = n_obs, ncol = n_vars_clus[[clus]])
  X_miss_MCAR_B[ , clus_positions[[clus]]][MCAR_B_miss_mat == 1] <- NA
  
}

#Saving data sets

saveRDS(X_full , paste(my_directory , 'X_full_', as.character(set_id) ,
                       '.rds' , sep = ''))

saveRDS(X_miss , paste(my_directory , 'X_miss_', as.character(set_id) ,
                       '.rds' , sep = ''))


saveRDS(X_miss_MCAR_B , paste(my_directory , 'X_miss_MCAR_B_', as.character(set_id) ,
                       '.rds' , sep = ''))
