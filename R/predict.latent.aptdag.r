predict.latent.aptdag <- function(obj, newcoords, rho=NULL, mcmc_keep=NULL, n_threads=1){
  
  if(is.null(mcmc_keep)){
    mcmc_keep <- ncol(obj$theta)
  }
  
  if(is.null(rho)){
    rho <- obj$rho
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
  
  result <- aptdaggp_latent_predict(newcoords, w, 
                                      obj$coords, rho, 
                                      theta, obj$M, n_threads)

  nout <- nrow(result$wout)
  tau_mcmc <- matrix(1, nrow=nout, ncol=1) %*% matrix(nugg^.5, nrow=1, ncol=mcmc_keep)
  yout <- result$wout + tau_mcmc * matrix(rnorm(nout * mcmc_keep), ncol=mcmc_keep)

  return(list(yout = yout,
              wout = result$wout,
              dag = result$dag))
}

