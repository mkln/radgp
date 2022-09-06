predict.response.altdag <- function(obj, newcoords, rho, mc_keep, n_threads){
  result <- altdaggp_response_predict(newcoords, obj$y, 
                                      obj$coords, rho, 
                                      obj$theta, obj$M, mc_keep, n_threads)
  
  return(result)
}