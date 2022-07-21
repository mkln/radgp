#ifndef ALTDAG 
#define ALTDAG

// uncomment to disable openmp on compilation
//#undef _OPENMP

#include "RcppArmadillo.h"
#include <RcppEigen.h>
//#include <Eigen/CholmodSupport>
#include "nnsearch.h"
#include "covariance_lmc.h"

using namespace std;

class AltDAG {
public:
  int nr;
  arma::vec w;
  arma::mat coords;
  double rho;
  
  Eigen::SparseMatrix<double> H;
  
  void make_precision(const arma::vec& theta);
  
  arma::field<arma::uvec> dag;
  
  AltDAG(const arma::mat& coords, double rho);
  
  // utils
  arma::uvec oneuv;
};




#endif