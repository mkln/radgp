predict.response.radgp <- function(obj, newcoords, rho=NULL, mcmc_keep=NULL, n_threads=1){
  
  if(is.null(mcmc_keep)){
    mcmc_keep <- ncol(obj$theta)
  }
  
  if(is.null(rho)){
    rho <- obj$rho
  }
  
  covar <- obj$covar
  
  mcmc_burn <- ncol(obj$theta) - mcmc_keep
  if(mcmc_burn > 0){
    param <- rbind(obj$theta[,-(1:mcmc_burn)],
                   obj$nugg[-(1:mcmc_burn)])
  } else {
    param <- rbind(obj$theta, obj$nugg)
  }
  
  result <- radgp_response_predict(newcoords, obj$y, 
                                      obj$coords, rho, 
                                      param, obj$M, covar, n_threads)
  
  return(result)
}