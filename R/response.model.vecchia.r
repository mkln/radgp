response.model.vecchia <- function(y, coords, m, mcmc, n_threads,
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
  
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  y_mm <- y[ixmm]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  
  maxmin_time <- system.time({    
    response_model <- aptdaggp_custom(y_mm, coords_mm, nn_dag, mcmc, n_threads,
                                      theta_start, metrop_sd, unif_bounds, printn) })
  
  result <- c(response_model, list(
    dag=nn_dag,
    ord=ixmm,
    m=m,
    y=y_mm,
    coords=coords_mm))
  class(result) <- "response.vecchia"
  return(result)
}
