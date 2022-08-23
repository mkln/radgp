response.model.vecchia <- function(y, coords, m, mcmc, n_threads,
                                   theta_start=NULL, unif_bounds=NULL){
  
  if(is.null(theta_start)){
    theta_start <- c(5,  1, 1)  
  }
  
  if(is.null(unif_bounds)){
    unif_bounds <- matrix(nrow=3, ncol=2)
    unif_bounds[,1] <- 1e-3
    unif_bounds[,2] <- 30
  }
  
  metrop_sd <- 0.15
  
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  y_mm <- y[ixmm]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  
  maxmin_time <- system.time({    
    response_model <- altdaggp_custom(y_mm, coords_mm, nn_dag, mcmc, n_threads,
                                      theta_start, metrop_sd, unif_bounds) })
  
  return(c(response_model, list(
    dag=nn_dag,
    ord=ixmm
  )))
}
