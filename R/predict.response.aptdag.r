predict.response.aptdag <- function(obj, newcoords, rho=NULL, mcmc_keep=NULL, n_threads=1){
  
  if(is.null(mcmc_keep)){
    mcmc_keep <- ncol(obj$theta)
  }
  
  if(is.null(rho)){
    rho <- obj$rho
  }
  
  mcmc_burn <- ncol(obj$theta) - mcmc_keep
  theta <- obj$theta[,-(1:mcmc_burn)]
  result <- aptdaggp_response_predict(newcoords, obj$y, 
                                      obj$coords, rho, 
                                      theta, obj$M, n_threads)
  
  return(result)
}