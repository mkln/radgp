response.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, 
                           theta_prior=NULL, 
                           nugg_start=NULL,
                           nugg_prior=NULL,
                           nugg_bounds=NULL,
                           covariance_model="pexp",
                           printn=10){
  
  unif_bounds <- matrix(nrow=3, ncol=2)
  if(is.null(theta_prior)){
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
    # nu matern
    unif_bounds[3,] <- c(0.5001, 2)
  } else {
    unif_bounds <- theta_prior[1:3,]
  }
  
  if(is.null(theta_start)){
    theta_start <- apply(unif_bounds, 1, mean)
    theta_start[2] <- 1
  }
  
  if(is.null(nugg_start)){
    nugg_start <- .5
  }
  if(is.null(nugg_prior)){
    nugg_prior <- c(2, 1)
  }
  if(is.null(nugg_bounds)){
    nugg_bounds = c(1e-7, Inf)
  } 
  
  unif_bounds <- rbind(unif_bounds, nugg_bounds)
  param_start <- c(theta_start, nugg_start)
  
  if(covariance_model == "pexp"){ 
    covar <- 0
  } else {
    covar <- 1
  }
  
  metrop_sd <- 0.15
  radgp_time <- system.time({
    radgp_model <- radgp_response(y, coords, rho=rho, mcmc, n_threads,
                                      param_start, metrop_sd, unif_bounds, 
                                      nugg_prior, covar, printn) })
  
  result <- c(radgp_model, list(time=radgp_time))
  
  result$nugg <- result$theta[4,]
  result$theta <- result$theta[1:3,]
  
  class(result) <- "response.radgp"
  return(result)
}

