logit <- function(p){
  #logit function
  return(log(p/(1-p)))
}

inv_logit <- function(x){
  #inverse logit function
  return(1/(1+exp(-x)))
}

bin_search_cont <- function(my_function , y , a , b , n_iter, verbose = F){
  #binary search function
  for(i in 1:n_iter){
    if(verbose){
      print(paste('Iteration :' , as.character(i)))
    }
    if(my_function((a+b)/2) < y){
      a <- (a+b)/2
    }else{
      b <- (a+b)/2
    }
  }
  return(a)
}

find_intercept <- function(X ,prop , param){
  #finding intercept in logit glm given parameters, proportion and data
  beta <- matrix(data = param , nrow = dim(X)[2] , ncol = 1)
  return(bin_search_cont(my_function = function(intercept){mean(inv_logit(intercept + X %*% beta))} ,
                         y = prop, a = -100 , b = 100 , n_iter = 30))
}

get_auc_score <- function(X, prop, param){
  #getting AUC score given parameters, proportion and data
  intercept <- find_intercept(X, prop, param)
  beta <- matrix(data = param , nrow = dim(X)[2] , ncol = 1)
  y <- rbinom(n = dim(X)[1] , size = 1 , p = inv_logit(intercept+X%*%beta))
  
  my_df <- data.frame(cbind(X , y))
  
  my_df$y <- as.factor(make.names(my_df$y))
  
  my_ctrl <- trainControl(method = "cv", number = 5, summaryFunction = twoClassSummary, classProbs = TRUE)
  
  my_model <- train(y ~., data = my_df, method = "glm", family = 'binomial',
                    trControl = my_ctrl)
  return(my_model$results[[2]])
}

find_param <- function(X , prop, auc_score){
  #finding parameters given proportion and AUC score
  return(bin_search_cont(function(param){return(get_auc_score(X , prop , param))} 
                         , y = auc_score ,a = 0 , b = 10 , n_iter = 14 , verbose = T))
}
