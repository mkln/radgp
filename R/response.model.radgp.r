response.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, 
                           theta_prior=NULL, 
                           nugg_start=NULL,
                           nugg_prior=NULL,
                           nugg_bounds=NULL,
                           printn=10){
  
  if(is.null(theta_start)){
    theta_start <- c(5, 1, 1.5)  
  }
  
  if(is.null(nugg_start)){
    nugg_start <- .5
  }
  
  param_start <- c(theta_start, nugg_start)
  
  unif_bounds <- matrix(nrow=4, ncol=2)
  if(is.null(theta_prior)){
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
    # nu powerexp
    unif_bounds[3,] <- c(1.001, 2-.01)
  } else {
    unif_bounds <- theta_prior[1:3,]
  }
  
  if(is.null(nugg_prior)){
    nugg_prior <- c(2, 1)
  }
  if(is.null(nugg_bounds)){
    nugg_bounds = c(1e-7, Inf)
  } 
  unif_bounds <- rbind(unif_bounds, nugg_bounds)
  
  metrop_sd <- 0.15
  radgp_time <- system.time({
    radgp_model <- aptdaggp_response(y, coords_train, rho=rho, mcmc, n_threads,
                                      param_start, metrop_sd, unif_bounds, 
                                      nugg_prior, printn) })
  
  result <- c(radgp_model, list(time=radgp_time))
  
  result$nugg <- result$theta[4,]
  result$theta <- result$theta[1:3,]
  
  class(result) <- "response.radgp"
  return(result)
}

