library('softImpute')
library('tidyverse')
library('missMethods')

my_directory <- ''

X_miss <- as.matrix(read_csv(paste(my_directory , 'X_nIDPs_analysis_comp.gz',
                                   sep = '')))[ , -c(1)]

X_miss <- t(t(X_miss)-apply(X_miss , MARGIN = 2 , function(x){return(mean(x,na.rm = T))}))

X_miss <- t(t(X_miss)/apply(X_miss , MARGIN = 2 , function(x){return(sd(x,na.rm = T))}))

perc_full_rank <- 0.3

dim_low_rank <- round(dim(X_miss)[2]*perc_full_rank)

data_SVD <- X_miss

data_SVD[is.na(X_miss)] <- impute_mean(X_miss)[is.na(X_miss)]

singular_vals <- svd(data_SVD)$d

my_lambda <- singular_vals[dim_low_rank]

my_result <- softImpute(x = X_miss , rank.max = dim_low_rank,
                        lambda = my_lambda, trace.it = T , type = 'svd')

X_new <- softImpute::complete(X_miss , my_result)

write_tsv(as.data.frame(X_new) , file=paste(my_directory, 'X_imp_SoftImpute.tsv.gz', sep = ''))
