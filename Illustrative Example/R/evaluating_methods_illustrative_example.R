library('tidyverse')
library('mice')
library('miceadds')
library('lmtest')
library('DescTools')

my_directory <- ''


my_df_CompVars <- read_csv(paste(my_directory ,
                             paste('final_df_CompVars_15.csv', sep = ''),
                             sep = ''))[ , -c(1)]

my_df_mean <- read_csv(paste(my_directory ,
                        paste('final_df_mean_15.csv', sep = ''),
                        sep = ''))[ , -c(1)]

my_df_SoftImpute <- read_csv(paste(my_directory ,
                             paste('final_df_SoftImpute_15.csv', sep = ''),
                             sep = ''))[ , -c(1)]

my_df_IterativeImputer <- read_csv(paste(my_directory ,
                             paste('final_df_IterativeImputer_15.csv', sep = ''),
                             sep = ''))[ , -c(1)]


n_binary_CompVars <- my_df_CompVars %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  apply(MARGIN = 2 , function(x){return(sum(!is.na(unique(x))) == 2)}) %>%
  sum

n_binary_mean <- my_df_mean %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  apply(MARGIN = 2 , function(x){return(sum(!is.na(unique(x))) == 2)}) %>%
  sum

n_binary_SoftImpute <- my_df_SoftImpute %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  apply(MARGIN = 2 , function(x){return(sum(!is.na(unique(x))) == 2)}) %>%
  sum

n_binary_IterativeImputer <- my_df_IterativeImputer %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  apply(MARGIN = 2 , function(x){return(sum(!is.na(unique(x))) == 2)}) %>%
  sum

print(c(n_binary_CompVars,n_binary_mean,n_binary_SoftImpute,n_binary_IterativeImputer))


p_miss_CompVars <- my_df_CompVars %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  is.na %>%
  mean

p_miss_mean <- my_df_mean %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  is.na %>%
  mean

p_miss_SoftImpute <- my_df_SoftImpute %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  is.na %>%
  mean

p_miss_IterativeImputer <- my_df_IterativeImputer %>%
  select(!c(normalised_grey_matter_log , townsend_indx)) %>%
  is.na %>%
  mean

print(c(p_miss_CompVars,p_miss_mean,p_miss_SoftImpute,p_miss_IterativeImputer))


combined_df <- my_df_CompVars %>%
  merge(my_df_mean) %>%
  merge(my_df_SoftImpute) %>%
  merge(my_df_IterativeImputer)

combined_df$age <- sqrt(combined_df$age_sq)

combined_df$age <- (combined_df$age - mean(combined_df$age))/sd(combined_df$age)

combined_df$age_sq <- combined_df$age^2


formula_list <- c()

method_list <- c('CompVars','mean', 'SoftImpute','IterativeImputer')

m <- 100

n_methods <- 4

for(n_vars in c(15)){
  for(method in method_list){
    my_df <- read_csv(paste(my_directory ,
                            paste('final_df_',method,'_',as.character(n_vars),'.csv', sep = ''),
                            sep = ''))[ , -c(1)]
    my_formula <- ''
    
    for(i in 1:dim(my_df)[2]){
      if(i == 1){
        my_formula <- paste(colnames(my_df)[i] , '~ age +' , sep = ' ')
        next
      }
      if(i == dim(my_df)[2]){
        my_formula <- paste(my_formula , colnames(my_df)[i], sep = ' ')
        next
      }
      my_formula <- paste(my_formula , colnames(my_df)[i], '+', sep = ' ')
    }
    
    formula_list <- c(formula_list , my_formula)
    
  }
}

# MI_data <- mice(combined_df, m = m , method = 'pmm', printFlag = T)
# 
# saveRDS(MI_data , paste(my_directory ,
#                        paste('MI_data_comb_',as.character(n_vars),'.rds', sep = ''), sep = ''))

MI_data <- readRDS(paste(my_directory ,paste('MI_data_comb_',as.character(n_vars),'.rds', sep = ''), sep = ''))

n <- dim(my_df_CompVars)[1]

k <- dim(my_df_CompVars)[2]

OF_SE_R2 <- function(R2,n,k){
  return(sqrt(4*R2*(1-R2)^2*(n-k-1)^2/(n^2-1)/(3+n)))
}



for(i in 1:n_methods){

  print(method_list[i])
  
  MI_model <- with(MI_data , lm(as.formula(formula_list[i])))
  
  print(summary(pool(MI_model)))
  
  R2_list <- c()
  V_within <- 0
  
  for(j in 1:m){
    my_OLS <- lm(formula = as.formula(formula_list[i]) , data = mice::complete(MI_data , action = j))
    
    R2 <- summary(my_OLS)$r.squared
    
    R2_list <- c(R2_list, R2)
    
    V_within <- V_within + OF_SE_R2(R2=R2,n=n,k=k)^2
    
  }
  
  V_within <- V_within/m
  
  V_between <- var(R2_list)
  
  V_total <- V_within + (1+1/m)*V_between
  
  print(paste('Average R^2 ' , method_list[i],': ',as.character(mean(R2_list)), sep = ''))
  
  print(paste('SE R^2 ' , method_list[i],': ',as.character(sqrt(V_total)), sep = ''))

}


# for(i in 1:n_methods){
#   
#   print(method_list[i])
#   
#   MI_model <- with(MI_data , lm(as.formula(formula_list[i])))
#   
#   print(summary(pool(MI_model)))
#   
#   R_list_Fisher <- c()
#   V_within_Fisher <- 0
#   
#   for(j in 1:m){
#     my_OLS <- lm(formula = as.formula(formula_list[i]) , data = mice::complete(MI_data , action = j))
#     
#     R2 <- summary(my_OLS)$r.squared
#     
#     R_list_Fisher <- c(R_list_Fisher, FisherZ(sqrt(R2)))
#   }
#   
#   V_within_Fisher <- 1/n
#   
#   V_between_Fisher <- var(R2_list)
#   
#   V_total_Fisher <- V_within_Fisher + (1+1/m)*V_between_Fisher
#   
#   print(paste('Averaged R^2 ' , method_list[i],': ',as.character(FisherZInv(mean(R_list_Fisher))^2), sep = ''))
#   
#   print(paste('95% CI R^2 ' , method_list[i],': [',as.character((FisherZInv(mean(R_list_Fisher)+qnorm(0.025)*sqrt(V_total_Fisher)))^2),
#               ' , ',as.character((FisherZInv(mean(R_list_Fisher)+qnorm(1-0.025)*sqrt(V_total_Fisher)))^2),']', sep = ''))
#   
# }
# 
# 
# 
