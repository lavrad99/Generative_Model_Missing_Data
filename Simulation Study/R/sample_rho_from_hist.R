sample_rho_from_hist <- function(n , my_hist){
  #function for taking n samples of rho from a histogram
  n_bins_tail <- (length(my_hist)-1)/2
  rho_samps <- replicate(n = n , expr = sample(x = 0:(length(my_hist)-1) , size = 1 , prob = my_hist))/n_bins_tail-1
  return(rho_samps)
}
