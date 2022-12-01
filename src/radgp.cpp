#include "radgp.h"
#include <thread>

DagGP::DagGP(
    const arma::vec& y_in,
    const arma::mat& coords_in, 
    double rho_in,
    int model,
    int nthread){
  y = y_in;
  coords = coords_in;
  nr = coords.n_rows;
  rho = rho_in;
  
  layers = arma::zeros<arma::uvec>(nr);
  dag = radialndag(coords, rho, layers, M);
  oneuv = arma::ones<arma::uvec>(1);
  
  model_type = model;
  prec_inited = false;
  
  n_threads = nthread;
}

DagGP::DagGP(
  const arma::vec& y_in,
  const arma::mat& coords_in, 
  const arma::field<arma::uvec>& custom_dag,
  int model, int nthread){
  y = y_in;
  coords = coords_in;
  nr = coords.n_rows;
  rho = -1;
  
  dag = custom_dag;
  oneuv = arma::ones<arma::uvec>(1);
  
  model_type = model;
  prec_inited = false;
  
  n_threads = nthread;
}

double DagGP::logdens(const arma::vec& theta){
  arma::vec logdetvec = arma::zeros(nr);
  arma::vec logdensvec = arma::zeros(nr);
  
  arma::vec* target;
  if(model_type == 0){
    // response
    target = &y;
  } else if(model_type == 1){
    // latent
    target = &w;
  }
  
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
    
    double core = arma::conv_to<double>::from(((*target)(i) - ht.t() * (*target)(px))/sqrtR);
    
    logdetvec(i) = -log(sqrtR);
    logdensvec(i) = -.5 * core*core;
    
  }
  
  //Rcpp::Rcout << "logdens: " << arma::accu(logdetvec) << " " << arma::accu(logdensvec) << endl;
  
  return arma::accu(logdetvec) + arma::accu(logdensvec);
}


void DagGP::make_precision_ahci(const arma::vec& theta){
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


void DagGP::make_precision_hci(const arma::vec& theta){
  make_precision_hci_core(H, Ci, prec_det, theta);
}

void DagGP::make_precision_hci_core(
  Eigen::SparseMatrix<double> & H,
  Eigen::SparseMatrix<double> & Ci,
  double & prec_det,
  const arma::vec& theta){
 
  std::chrono::steady_clock::time_point tstart = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
  
  tstart = std::chrono::steady_clock::now();
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  
  int nnz_h = 0;
  for(int i=0; i<nr; i++){
    nnz_h += dag(i).n_elem;
  }
  nnz_h += nr; // plus nr for eye
  tripletList_H.reserve(nnz_h);
  
  arma::field<arma::vec> ht(nr);
  arma::vec sqrtR(nr);
  arma::vec logdetvec(nr);
  
  tend = std::chrono::steady_clock::now();
  int timer = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
  //Rcpp::Rcout << "part 1 : " << timer << endl;
  tstart = std::chrono::steady_clock::now();
  
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
    
    ht(i) = PPi * CPt;
    sqrtR(i) = sqrt( arma::conv_to<double>::from(
                                          CC - CPt.t() * ht(i) ));
    
    logdetvec(i) = -log(sqrtR(i));
  }
  
  tend = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
  //Rcpp::Rcout << "part 2 : " << timer << endl;
  
  tstart = std::chrono::steady_clock::now();
  
  for(int i=0; i<nr; i++){
    tripletList_H.push_back( T(i, i, 1.0/sqrtR(i)) );
    for(unsigned int j=0; j<dag(i).n_elem; j++){
      tripletList_H.push_back( T(i, dag(i)(j), -ht(i)(j)/sqrtR(i)) );
    }
  }
  
  H = Eigen::SparseMatrix<double>(nr, nr);
  H.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  tend = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
  //Rcpp::Rcout << "part 3 : " << timer << endl;
  
  tstart = std::chrono::steady_clock::now();
  
  int nchunks = 1; 
#ifdef _OPENMP
  nchunks = n_threads; //std::thread::hardware_concurrency()/2;
#endif
  //Rcpp::Rcout << "num threads " << nchunks << endl;
  int chunksize = H.cols() / nchunks;
  
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
  static const bool windows=true;
#else
  static const bool windows=false;
#endif
  
  if(!prec_inited | windows){
    Ci = H.transpose() * H;
    prec_inited = true;
  } else {
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<nchunks; i++){
      if(i == nchunks-1){
        Ci.rightCols(H.cols() - (nchunks-1)*chunksize) = H.transpose() * H.rightCols(H.cols() - (nchunks-1)*chunksize);
      } else {
        Ci.middleCols(i*chunksize, chunksize) = H.transpose() * H.middleCols(i*chunksize, chunksize);
      }
    }
  }
  prec_det = 2 * arma::accu(logdetvec);
  
  tend = std::chrono::steady_clock::now();
  timer = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
  //Rcpp::Rcout << "part 4 : " << timer << endl;
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

