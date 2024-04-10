library('softImpute')
library('mice')
library('DescTools')
library('pROC')
library('missMethods')

my_directory <- ''

my_output_directory <- './results/'

n_sets_per_dim <- 20

perc_low_rank_vec <- c(5 , 15, 30)

env_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))

set_id <- env_id - ((env_id-1) %/% n_sets_per_dim)*n_sets_per_dim

perc_low_rank <- perc_low_rank_vec[1+(env_id-1) %/% n_sets_per_dim]


X_miss <- readRDS(paste(my_directory , 'X_miss_',as.character(set_id), '.rds' , sep = ''))

X_miss_MCAR_B <- readRDS(paste(my_directory , 'X_miss_MCAR_B_',as.character(set_id), '.rds' , sep = ''))

X_full <- readRDS(paste(my_directory , 'X_full_',as.character(set_id), '.rds' , sep = ''))

analysis_clus <- c(2,3)

bin_inds <- readRDS(paste(my_directory , 'bin_inds.rds' , sep = ''))

dim_low_rank <- round(perc_low_rank/100*dim(X_miss)[2])

my_lambda <- lambda0(X_miss , trace.it = T)/4

my_result <- softImpute(x = X_miss , rank.max = dim_low_rank,
                        lambda = my_lambda, trace.it = T , type = 'svd')

X_new <- softImpute::complete(X_miss , my_result)

X_star_mean_bin <- X_new

X_star_mean_bin[ , bin_inds][0.5 <  X_new[ , bin_inds]] <- 1
X_star_mean_bin[ , bin_inds][0.5 >=  X_new[ , bin_inds]] <- 0


for(clus in analysis_clus){
  bias_imp_FW <- c()
  
  sd_imp_FW <- c()
  
  RMSE_imp_FW <- c()
  
  auc_imp_FW <- c()
  
  cont_inds_clus <- readRDS(paste(my_directory  , 'cont_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
  bin_inds_clus <- readRDS(paste(my_directory  , 'bin_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
  for(i in cont_inds_clus){
    
    my_X_imp <- X_star_mean_bin[ , i]
    my_X_full <- X_full[ , i]
    bias_imp_FW <- c(bias_imp_FW ,
                     mean((my_X_imp - my_X_full)[is.na(X_miss[ , i])]))
    sd_imp_FW <- c(sd_imp_FW ,
                   sd(my_X_imp[is.na(X_miss[ , i])]))
    RMSE_imp_FW <- c(RMSE_imp_FW ,
                  sqrt(mean(((my_X_imp - my_X_full)^2)[is.na(X_miss[ , i])])))
    
  }
  
  
  for(i in bin_inds_clus){
    my_X_imp <- X_star_mean_bin[ , i]
    my_X_full <- X_full[ , i]
    if(length(unique(my_X_imp[is.na(X_miss[ , i])])) == 1){
      auc_imp_FW <- c(auc_imp_FW , 0.5)
    }else{
      auc_imp_FW <- c(auc_imp_FW ,
                      as.numeric(pROC::auc(predictor = my_X_full[is.na(X_miss[ , i])] ,
                                           response = my_X_imp[is.na(X_miss[ , i])])))
      
    }
  }
  
  
  saveRDS(bias_imp_FW , paste(my_output_directory , 'bias_imp_clus', as.character(clus)
                              , '_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(sd_imp_FW , paste(my_output_directory , 'sd_imp_clus', as.character(clus)
                            ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(RMSE_imp_FW , paste(my_output_directory , 'RMSE_imp_clus', as.character(clus)
                              ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(auc_imp_FW , paste(my_output_directory , 'auc_imp_clus', as.character(clus)
                             ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  
}

dim_low_rank <- round(perc_low_rank/100*dim(X_miss_MCAR_B)[2])

my_lambda <- lambda0(X_miss_MCAR_B , trace.it = T)/4

my_result <- softImpute(x = X_miss_MCAR_B , rank.max = dim_low_rank,
                        lambda = my_lambda, trace.it = T , type = 'svd')

X_new <- softImpute::complete(X_miss_MCAR_B , my_result)

X_star_mean_bin <- X_new

X_star_mean_bin[ , bin_inds][0.5 <  X_new[ , bin_inds]] <- 1
X_star_mean_bin[ , bin_inds][0.5 >=  X_new[ , bin_inds]] <- 0


for(clus in analysis_clus){
  bias_imp_FW <- c()
  
  sd_imp_FW <- c()
  
  RMSE_imp_FW <- c()
  
  auc_imp_FW <- c()
  
  cont_inds_clus <- readRDS(paste(my_directory  , 'cont_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
  bin_inds_clus <- readRDS(paste(my_directory  , 'bin_inds_clus', as.character(clus) , '.rds' , sep = ''))
  
  for(i in cont_inds_clus){
    my_X_imp <- X_star_mean_bin[ , i]
    my_X_full <- X_full[ , i]
    bias_imp_FW <- c(bias_imp_FW ,
                     mean((my_X_imp - my_X_full)[is.na(X_miss_MCAR_B[ , i])]))
    sd_imp_FW <- c(sd_imp_FW ,
                   sd(my_X_imp[is.na(X_miss_MCAR_B[ , i])]))
    RMSE_imp_FW <- c(RMSE_imp_FW ,
                  sqrt(mean(((my_X_imp - my_X_full)^2)[is.na(X_miss_MCAR_B[ , i])])))
    
  }
  
  
  for(i in bin_inds_clus){
    my_X_imp <- X_star_mean_bin[ , i]
    my_X_full <- X_full[ , i]
    if(length(unique(my_X_imp[is.na(X_miss_MCAR_B[ , i])])) == 1){
      auc_imp_FW <- c(auc_imp_FW , 0.5)
    }else{
      auc_imp_FW <- c(auc_imp_FW ,
                      as.numeric(pROC::auc(predictor = my_X_full[is.na(X_miss_MCAR_B[ , i])] ,
                                           response = my_X_imp[is.na(X_miss_MCAR_B[ , i])])))
      
    }
  }
  
  
  saveRDS(bias_imp_FW , paste(my_output_directory , 'MCAR_B_bias_imp_clus', as.character(clus)
                              ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(sd_imp_FW , paste(my_output_directory , 'MCAR_B_sd_imp_clus', as.character(clus)
                            ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(RMSE_imp_FW , paste(my_output_directory , 'MCAR_B_RMSE_imp_clus', as.character(clus)
                              ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  saveRDS(auc_imp_FW , paste(my_output_directory , 'MCAR_B_auc_imp_clus', as.character(clus)
                             ,'_FW_SoftImpute_',as.character(set_id), '_perc', as.character(perc_low_rank),'.rds' , sep = ''))
  
}
