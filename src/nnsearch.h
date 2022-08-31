
#include <RcppArmadillo.h>
#include "MaxMin.h"
using namespace std;

arma::umat make_candidates(const arma::mat& w, const arma::uvec& indsort, 
                           unsigned int col,
                           double rho);

arma::field<arma::uvec> neighbor_search(const arma::mat& w, double rho);
  
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset, int& M, const arma::mat& w, double rho);

arma::field<arma::uvec> altdagbuild(const arma::mat& w, double rho, arma::uvec& layers, int& M);

arma::umat sparse_struct(const arma::field<arma::uvec>& dag, int nr);