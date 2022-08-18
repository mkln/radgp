response.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, unif_bounds=NULL){
  
  if(is.null(theta_start)){
    theta_start <- c(5,  1, 1)  
  }
  
  if(is.null(unif_bounds)){
    unif_bounds <- matrix(nrow=3, ncol=2)
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
  }
  
  #unif_bounds[2,] <- c(1.001, 2-.01)
  
  metrop_sd <- 0.15
  altdag_time <- system.time({
    altdag_model <- altdaggp_response(y, coords_train, rho=rho, mcmc, n_threads,
                                      theta_start, metrop_sd, unif_bounds) })
  
  result <- c(altdag_model, list(time=altdag_time))
  
  class(result) <- "altdag"
  return(result)
}

