latent.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, 
                           theta_prior=NULL, 
                           nugg_start=NULL,
                           nugg_prior=NULL,
                           covariance_model="pexp",
                           printn=10){
  
  
  
  
  
  if(is.null(theta_prior)){
    unif_bounds <- matrix(nrow=3, ncol=2)
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
    # nu matern
    unif_bounds[3,] <- c(0.5001, 2)
  } else {
    unif_bounds <- theta_prior[1:3, ]
  }
  if(is.null(theta_start)){
    theta_start <- c(apply(unif_bounds[1:3,], 1, mean), 1e-8)  
  } else {
    theta_start[4] <- 1e-8
  }
  
  
  if(is.null(nugg_start)){
    nugg_start <- 1
  }
  if(is.null(nugg_prior)){
    nugg_prior <- c(2, 1)
  }
  
  unif_bounds <- rbind(unif_bounds, theta_start[4] + c(-1e-9, +1e-9))
  
  if(covariance_model == "pexp"){ 
    covar <- 0
  } else {
    covar <- 1
  }
  
  metrop_sd <- 0.15
  radgp_time <- system.time({
    radgp_model <- radgp_latent(y, coords, rho=rho, mcmc, n_threads,
                                    theta_start, nugg_start, 
                                    metrop_sd, unif_bounds, 
                                    nugg_prior, covar, printn) })
  
  radgp_model$theta <- radgp_model$theta[1:3,]
  
  result <- c(radgp_model, list(time=radgp_time))
  
  class(result) <- "latent.radgp"
  return(result)
}

