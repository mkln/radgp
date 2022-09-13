#include "aptdag.h"

AptDAG::AptDAG(
    const arma::vec& y_in,
    const arma::mat& coords_in, 
    double rho_in){
  y = y_in;
  coords = coords_in;
  nr = coords.n_rows;
  rho = rho_in;
  
  layers = arma::zeros<arma::uvec>(nr);
  dag = aptdagbuild(coords, rho, layers, M);
  oneuv = arma::ones<arma::uvec>(1);
}

AptDAG::AptDAG(
  const arma::vec& y_in,
  const arma::mat& coords_in, 
  const arma::field<arma::uvec>& custom_dag){
  y = y_in;
  coords = coords_in;
  nr = coords.n_rows;
  rho = -1;
  
  dag = custom_dag;
  oneuv = arma::ones<arma::uvec>(1);
}

double AptDAG::logdens(const arma::vec& theta){
  arma::vec logdetvec = arma::zeros(nr);
  arma::vec logdensvec = arma::zeros(nr);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, false, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, false, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, false, true) );
    
    arma::vec ht = PPi * CPt;
    double sqrtR = sqrt( arma::conv_to<double>::from(
      CC - CPt.t() * ht ));
    
    double ycore = arma::conv_to<double>::from((y(i) - ht.t() * y(px))/sqrtR);
    
    logdetvec(i) = -log(sqrtR);
    logdensvec(i) = -.5 * ycore*ycore;
    
  }
  
  return arma::accu(logdetvec) + arma::accu(logdensvec);
}


void AptDAG::make_precision(const arma::vec& theta){
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H, tripletList_A;
  
  int nnz_h = 0;
  for(int i=0; i<nr; i++){
    nnz_h += dag(i).n_elem;
  }
  tripletList_A.reserve(nnz_h);
  nnz_h += nr; // plus nr for eye
  tripletList_H.reserve(nnz_h);
  
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
  
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, false, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, false, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, false, true) );
    
    arma::vec ht = PPi * CPt;
    double sqrtR = sqrt( arma::conv_to<double>::from(
      CC - CPt.t() * ht ));
    
    tripletList_H.push_back( T(i, i, 1.0/sqrtR) );
    for(unsigned int j=0; j<dag(i).n_elem; j++){
      tripletList_H.push_back( T(i, dag(i)(j), -ht(j)/sqrtR) );
      tripletList_A.push_back( T(i, dag(i)(j), ht(j)) );
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  H = Eigen::SparseMatrix<double>(nr, nr);
  H.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  A = Eigen::SparseMatrix<double>(nr, nr);
  A.setFromTriplets(tripletList_A.begin(), tripletList_A.end());
}


//[[Rcpp::export]]
Eigen::SparseMatrix<double> hmat_from_dag(
    const arma::mat& coords,
    const arma::field<arma::uvec>& dag, 
    const arma::vec& theta){
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  int nr = dag.n_elem;
  int nnz_h = 0;
  for(int i=0; i<nr; i++){
    nnz_h += dag(i).n_elem + 1; // plus one for eye
  }
  tripletList_H.reserve(nnz_h);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, false, true);
    arma::mat CPt = Correlationf(coords, px, ix, theta, false, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, false, true) );
    
    arma::vec ht = PPi * CPt;
    double sqrtR = sqrt( arma::conv_to<double>::from(
      CC - CPt.t() * ht ));
    
    tripletList_H.push_back( T(i, i, 1.0/sqrtR) );
    for(unsigned int j=0; j<dag(i).n_elem; j++){
      tripletList_H.push_back( T(i, dag(i)(j), -ht(j)/sqrtR) );
    }
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  Eigen::SparseMatrix<double> H(nr, nr);
  H.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  return H;
}


//[[Rcpp::export]]
arma::vec pred_from_dag(
    const arma::mat& coords,
    const arma::field<arma::uvec>& dag, 
    const arma::vec& theta,
    const arma::vec& urng){

  int nr = coords.n_rows;
  if(nr != urng.n_elem){
    Rcpp::stop("Wrong dimensions in coords and input rng vector");
  }
  arma::vec w = arma::zeros(nr);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int i=0; i<nr; i++){
    arma::uvec ix = oneuv * i;
    arma::uvec px = dag(i);
    
    arma::mat CC = Correlationf(coords, ix, ix, 
                                theta, false, true);
    arma::mat CP = Correlationf(coords, ix, px, theta, false, false);
    arma::mat PPi = 
      arma::inv_sympd( Correlationf(coords, px, px, theta, false, true) );
    
    arma::mat H = CP * PPi;
    double sqrtR = sqrt( arma::conv_to<double>::from(
      CC - H * CP.t() ));
    
    w(i) = arma::conv_to<double>::from(H * w.elem(px)) + sqrtR * urng(i);
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
  }
  
  return w;
}