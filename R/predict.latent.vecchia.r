predict.latent.vecchia <- function(obj, newcoords, mcmc_keep=NULL, n_threads=1, independent=FALSE){
  if(is.null(mcmc_keep)){
    mcmc_keep <- ncol(obj$theta)
  }
  
  mcmc_burn <- ncol(obj$theta) - mcmc_keep
  if(mcmc_burn > 0){
    theta <- obj$theta[,-(1:mcmc_burn)]
    w <- obj$w[,-(1:mcmc_burn)]
    nugg <- obj$nugg[-(1:mcmc_burn)]
  } else {
    theta <- obj$theta
    w <- obj$w
    nugg <- obj$nugg
  }
  theta <- rbind(theta, rep(1e-8, ncol(theta)))
  
  covar <- obj$covar
  
  if(!independent){
    ixmm_pred <- GPvecchia::order_maxmin_exact_obs_pred(obj$coords, newcoords)$ord_pred
  
    newcoords_mm <- newcoords[ixmm_pred,]
    coords_all_mm <- rbind(obj$coords, newcoords_mm)
    nn_dag_mat <- GPvecchia:::findOrderedNN_kdtree2(coords_all_mm, obj$m)
    nn_dag <- apply(nn_dag_mat, 1, function(x){ x[!is.na(x)][-1]-1 })
    
    result <- vecchiagp_latent_predict(newcoords_mm, w, obj$coords, nn_dag,
                                        theta, covar, n_threads)
    
    wout <- result[order(ixmm_pred),]
    dag <- nn_dag
    ord <- ixmm_pred
  } else {
    
    # find nearest neighbor in reference set
    nn.found <- FNN::get.knnx(obj$coords, newcoords, k=obj$m)
    
    pred_dag <- as.list(as.data.frame(t(nn.found$nn.index-1)))
    nn_dag <- c(obj$dag, pred_dag)
    
    result <- vecchiagp_latent_predict(newcoords, w, obj$coords, nn_dag,
                                         theta, covar, n_threads)
    
    wout <- result
    dag <- nn_dag
    ord <- NULL
  }

  nout <- nrow(wout)
  tau_mcmc <- matrix(1, nrow=nout, ncol=1) %*% matrix(nugg^.5, nrow=1, ncol=mcmc_keep)
  yout <- wout + tau_mcmc * matrix(rnorm(nout * mcmc_keep), ncol=mcmc_keep)

  return(list(yout = yout,
              wout = wout,
              dag = dag,
              ord = ord))
}

