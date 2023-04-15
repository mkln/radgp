#ifndef DAGGP 
#define DAGGP

#include "RcppArmadillo.h"
#include <RcppEigen.h>
//#include <Eigen/CholmodSupport>

#include "nnsearch.h"
#include "covariance.h"
#include "interrupt.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// uncomment to disable openmp on compilation
//#undef _OPENMP

using namespace std;

class DagGP {
public:
  int nr;
  arma::vec y;
  arma::mat coords;
  
  arma::uvec layers;
  
  int type; // 1=altdag, 2=nn maxmin order
  int M;
  double rho;
  
  Eigen::SparseMatrix<double> A, H, Ci;
  
  double * bessel_ws;
  
  double logdens(const arma::vec& theta);
  
  void make_precision_ahci(const arma::vec& theta);
  void make_precision_hci(const arma::vec& theta);
  void make_precision_hci_core(
      Eigen::SparseMatrix<double> & H,
      Eigen::SparseMatrix<double> & Ci,
      double & prec_det,
      const arma::vec& theta);
  
  arma::field<arma::uvec> dag;
  
  // info about model: 
  // 0 = response
  // 1 = latent CG
  int model_type;
  
  // stuff for latent model
  arma::vec w;
  double prec_det;
  void new_w_logdens(); // update logdens after refreshing w
  bool prec_inited;
  
  //double ldens;
  DagGP(const arma::vec& y,
    const arma::mat& coords, double rho, int model=0,
    int nthread=0);
  DagGP(const arma::vec& y,
         const arma::mat& coords, 
         const arma::field<arma::uvec>& custom_dag,
         int model=0,
         int nthread=0);
  
  // utils
  arma::uvec oneuv;
  int n_threads;
};




#endif