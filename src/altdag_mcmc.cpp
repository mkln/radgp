#include <RcppArmadillo.h>
#include "altdag.h"


using namespace std;

//[[Rcpp::export]]
Rcpp::List altdaggp(const arma::mat& coords,
                    const arma::vec& theta,
                    double rho){
  
  AltDAG adag(coords, rho);
  
  adag.make_precision(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("H") = adag.H
  );
}