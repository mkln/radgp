predict.response.vecchia <- function(obj, newcoords, n_threads){
  ixmm_pred <- GPvecchia::order_maxmin_exact_obs_pred(obj$coords, newcoords)$ord_pred
  
  newcoords_mm <- newcoords[ixmm_pred,]
  coords_all_mm <- rbind(obj$coords, newcoords_mm)
  nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_all_mm, obj$m)
  nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
  
  result <- vecchiagp_response_predict(newcoords_mm, obj$y, obj$coords, nn_dag,
                                      obj$theta, n_threads)
  
  return(list(yout = result[order(ixmm_pred),]))
}

