#ifndef ALTDAG 
#define ALTDAG

// uncomment to disable openmp on compilation
#include "RcppArmadillo.h"
#include <RcppEigen.h>
//#include <Eigen/CholmodSupport>
#include "nnsearch.h"
#include "covariance.h"
#include "interrupt.h"


#ifdef _OPENMP
#include <omp.h>
#endif

//#undef _OPENMP

using namespace std;

class AltDAG {
public:
  int nr;
  arma::vec y;
  arma::vec w;
  arma::mat coords;
  
  arma::uvec layers;
  
  int type; // 1=altdag, 2=nn maxmin order
  int M;
  double rho;
  
  Eigen::SparseMatrix<double> A, H;
  
  double logdens(const arma::vec& theta);
  void make_precision(const arma::vec& theta);
  
  arma::field<arma::uvec> dag;
  
  //double ldens;
  
  AltDAG(const arma::vec& y,
    const arma::mat& coords, double rho);
  AltDAG(const arma::vec& y,
         const arma::mat& coords, 
         const arma::field<arma::uvec>& custom_dag);
  
  // utils
  arma::uvec oneuv;
};




#endif