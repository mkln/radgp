predict.response.vecchia <- function(obj, newcoords, mcmc_keep=NULL, n_threads=1, independent=FALSE){
  if(is.null(mcmc_keep)){
    mcmc_keep <- ncol(obj$theta)
  }
  
  mcmc_burn <- ncol(obj$theta) - mcmc_keep
  theta <- obj$theta[,-(1:mcmc_burn)]
  if(!independent){
    ixmm_pred <- GPvecchia::order_maxmin_exact_obs_pred(obj$coords, newcoords)$ord_pred
  
    newcoords_mm <- newcoords[ixmm_pred,]
    coords_all_mm <- rbind(obj$coords, newcoords_mm)
    nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_all_mm, obj$m)
    nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
    
    result <- vecchiagp_response_predict(newcoords_mm, obj$y, obj$coords, nn_dag,
                                        theta, n_threads)
    
    return(list(yout = result[order(ixmm_pred),]))
  } else {
    
    # find nearest neighbor in reference set
    nn.found <- FNN::get.knnx(obj$coords, newcoords, k=obj$m)
    
    pred_dag <- as.list(as.data.frame(t(nn.found$nn.index-1)))
    nn_dag <- c(obj$dag, pred_dag)
    
    result <- vecchiagp_response_predict(newcoords, obj$y, obj$coords, nn_dag,
                                         theta, n_threads)
    
    return(list(yout = result))
  }
}

