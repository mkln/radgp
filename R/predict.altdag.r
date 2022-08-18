predict.altdag <- function(obj, newcoords, n_threads){
  result <- altdaggp_response_predict(coords_test, obj$y, 
                                      obj$coords, obj$rho, 
                                      obj$theta, obj$M, n_threads)
  
  return(result)
}