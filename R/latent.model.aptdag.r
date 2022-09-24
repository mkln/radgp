latent.model <- function(y, coords, rho, mcmc, n_threads,
                           theta_start=NULL, 
                           theta_prior=NULL, 
                           tausq_start=NULL,
                           printn=10){
  
  if(is.null(theta_start)){
    theta_start <- c(5, 1, 1.5, 1e-8)  
  } else {
    theta_start[4] <- 1e-8
  }
  
  if(is.null(tausq_start)){
    tausq_start <- 1
  }
  
  if(is.null(theta_prior)){
    unif_bounds <- matrix(nrow=4, ncol=2)
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
    # nu powerexp
    unif_bounds[3,] <- c(1.001, 2-.01)
  } else {
    unif_bounds <- theta_prior[1:3, ]
  }
  
  unif_bounds <- rbind(unif_bounds, theta_start[4] + c(-1e-9, +1e-9))
  
  metrop_sd <- 0.15
  aptdag_time <- system.time({
    aptdag_model <- aptdaggp_latent(y, coords_train, rho=rho, mcmc, n_threads,
                                    theta_start, tausq_start, 
                                    metrop_sd, unif_bounds, printn) })
  
  aptdag_model$theta <- aptdag_model$theta[1:3,]
  
  result <- c(aptdag_model, list(time=aptdag_time))
  
  class(result) <- "latent.aptdag"
  return(result)
}

