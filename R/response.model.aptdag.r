response.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, unif_bounds=NULL, printn=10){
  
  if(is.null(theta_start)){
    theta_start <- c(5, 1, 1.5, 1)  
  }
  
  if(is.null(unif_bounds)){
    unif_bounds <- matrix(nrow=4, ncol=2)
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
    # nu powerexp
    unif_bounds[3,] <- c(1.001, 2-.01)
  }
  
  metrop_sd <- 0.15
  aptdag_time <- system.time({
    aptdag_model <- aptdaggp_response(y, coords_train, rho=rho, mcmc, n_threads,
                                      theta_start, metrop_sd, unif_bounds, printn) })
  
  result <- c(aptdag_model, list(time=aptdag_time))
  
  class(result) <- "response.aptdag"
  return(result)
}

