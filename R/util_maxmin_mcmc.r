maxmin_mcmc <- function(y, coords, m, mcmc, n_threads,
                        theta_start, metrop_sd, unif_bounds){
  
  ixmm <- GPvecchia::order_maxmin_exact(coords)
  coords_mm <- coords[ixmm,]
  y_mm <- y[ixmm]
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_mm, m)
  nn_dag <- nn_dag_mat %>% apply(1, \(x) x[!is.na(x)][-1]-1)
  
  maxmin_time <- system.time({    
    response_model <- altdaggp_custom(y_mm, coords_mm, nn_dag, mcmc, n_threads,
                                      theta_start, metrop_sd, unif_bounds) })
  
  return(list(
    theta=response_model$theta,
    ldens=response_model$ldens,
    dag=nn_dag,
    ord=ixmm,
    timing=maxmin_time["elapsed"]
  ))
}
