#include "altdag.h"

AltDAG::AltDAG(
    const arma::vec& y_in,
    const arma::mat& coords_in, 
    double rho_in){
  y = y_in;
  coords = coords_in;
  nr = coords.n_rows;
  rho = rho_in;
  dag = altdagbuild(coords, rho);
  oneuv = arma::ones<arma::uvec>(1);
  
}


void AltDAG::logdens(const arma::vec& theta){
  
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
    logdetvec(i) = -sqrtR;
    logdensvec(i) = -.5 * ycore*ycore;
  }
  
  ldens = arma::accu(logdetvec) + arma::accu(logdensvec);
}


void AltDAG::make_precision(const arma::vec& theta){
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  
  int nnz_h = 0;
  for(int i=0; i<nr; i++){
    nnz_h += dag(i).n_elem + 1; // plus one for eye
  }
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
    }
  }
  
  H = Eigen::SparseMatrix<double>(nr, nr);
  H.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
}
