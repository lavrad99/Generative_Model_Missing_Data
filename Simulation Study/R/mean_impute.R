library('softImpute')
library('mice')
library('DescTools')
library('pROC')
library('missMethods')

my_directory <- ''

my_output_directory <- './results/'

set_id <- as.integer(Sys.getenv("SLURM_ARRAY_TASK_ID"))


X_miss <- readRDS(paste(my_directory , 'X_miss_',as.character(set_id), '.rds' , sep = ''))

X_miss_MCAR_B <- readRDS(paste(my_directory , 'X_miss_MCAR_B_',as.character(set_id), '.rds' , sep = ''))

X_full <- readRDS(paste(my_directory , 'X_full_',as.character(set_id), '.rds' , sep = ''))

analysis_clus <- c(2,3)

bin_inds <- readRDS(paste(my_directory , 'bin_inds.rds' , sep = ''))

X_new <- impute_mean(X_miss)

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
                              , '_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(sd_imp_FW , paste(my_output_directory , 'sd_imp_clus', as.character(clus)
                            ,'_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(RMSE_imp_FW , paste(my_output_directory , 'RMSE_imp_clus', as.character(clus)
                              , '_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(auc_imp_FW , paste(my_output_directory , 'auc_imp_clus', as.character(clus)
                             ,'_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  
}

X_new <- impute_mean(X_miss_MCAR_B)

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
                              , '_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(sd_imp_FW , paste(my_output_directory , 'MCAR_B_sd_imp_clus', as.character(clus)
                            ,'_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(RMSE_imp_FW , paste(my_output_directory , 'MCAR_B_RMSE_imp_clus', as.character(clus)
                              ,'_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  saveRDS(auc_imp_FW , paste(my_output_directory , 'MCAR_B_auc_imp_clus', as.character(clus)
                             ,'_FW_mean_',as.character(set_id),'.rds' , sep = ''))
  
}
