latent.model.vecchia <- function(y, coords, m, mcmc, n_threads,
                                 theta_start=NULL, 
                                 theta_prior=NULL,
                                 nugg_start=NULL,
                                 nugg_prior=NULL,
                                 unif_bounds=NULL, 
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
  
  metrop_sd <- 0.15
  
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  y_mm <- y[ixmm]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  
  if(covariance_model == "pexp"){ 
    covar <- 0
  } else {
    covar <- 1
  }
  
  maxmin_time <- system.time({    
    latent_model <- daggp_custom_latent(y_mm, coords_mm, nn_dag, mcmc, n_threads,
                                      theta_start, nugg_start,
                                      metrop_sd, unif_bounds, 
                                      nugg_prior,
                                      covar, printn) })
  
  latent_model$theta <- latent_model$theta[1:3,]
  
  result <- c(latent_model, list(
    dag=nn_dag,
    ord=ixmm,
    m=m,
    y=y_mm,
    coords=coords_mm,
    message="'dag' uses max-min ordering stored in 'ord'. Original order is given by order(ord)."))
  class(result) <- "latent.vecchia"
  return(result)
}
