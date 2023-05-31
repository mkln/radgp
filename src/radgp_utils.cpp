#include "radgp.h"

//[[Rcpp::export]]
Rcpp::List radgp_build(const arma::mat& coords,
                    const arma::vec& theta,
                    double rho, int covar){

  arma::vec y = arma::randn(coords.n_rows);
  DagGP adag(y, coords, rho, 0, covar, 1);
  
  adag.make_precision_ahci(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("A") = adag.A,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("layers") = adag.layers
  );
}


//[[Rcpp::export]]
Rcpp::List vecchiagp_build(const arma::mat& coords,
                     const arma::vec& theta,
                     const arma::field<arma::uvec>& dag, int covar){

  arma::vec y = arma::randn(coords.n_rows);
  DagGP adag(y, coords, dag, 0, covar, 1);
  
  adag.make_precision_ahci(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("A") = adag.A,
    Rcpp::Named("H") = adag.H
  );
}


//[[Rcpp::export]]
double daggp_negdens(const arma::vec& y,
                     const arma::mat& coords,
                     const arma::field<arma::uvec>& dag,
                     const arma::vec& theta,
                     int covar,
                     int num_threads){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  DagGP adag(y, coords, dag, 0, covar, num_threads);
  return adag.logdens(theta);
}