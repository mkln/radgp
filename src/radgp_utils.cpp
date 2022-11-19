#include "radgp.h"

//[[Rcpp::export]]
Rcpp::List aptdaggp(const arma::mat& coords,
                    const arma::vec& theta,
                    double rho){
  
  arma::vec y = arma::randn(coords.n_rows);
  AptDAG adag(y, coords, rho);
  
  adag.make_precision_ahci(theta);
  
  double ldens = adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = ldens,
    Rcpp::Named("A") = adag.A,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("layers") = adag.layers
  );
}


//[[Rcpp::export]]
Rcpp::List vecchiagp(const arma::mat& coords,
                     const arma::vec& theta,
                     const arma::field<arma::uvec>& dag){
  
  arma::vec y = arma::randn(coords.n_rows);
  AptDAG adag(y, coords, dag);
  adag.make_precision_ahci(theta);
  double ldens = adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = ldens,
    Rcpp::Named("A") = adag.A,
    Rcpp::Named("H") = adag.H
  );
}


//[[Rcpp::export]]
double daggp_negdens(const arma::vec& y,
                     const arma::mat& coords,
                     const arma::field<arma::uvec>& dag,
                     const arma::vec& theta,
                     int num_threads){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  AptDAG adag(y, coords, dag, 0, num_threads);
  return adag.logdens(theta);
}