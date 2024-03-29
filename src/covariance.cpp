#include "covariance.h"

using namespace std;

// matern
void matern_inplace(arma::mat& res, 
                             const arma::mat& coords,
                             const arma::uvec& ix, const arma::uvec& iy, 
                             const double& phi, const double& nu, 
                             const double& sigmasq, 
                             double * bessel_ws,
                             const double& nugginside=0,  bool same=false){
  
  int threadid = 0;
#ifdef _OPENMP
  threadid = omp_get_thread_num();
#endif
  
  double pow2_nu1_gammanu_sigmasq = sigmasq * pow(2.0, 1.0-nu) / R::gammafn(nu);
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));//x.row(i);
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));//y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          res(i, j) = pow(hphi, nu) * pow2_nu1_gammanu_sigmasq *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadid*3]);
        } else {
          res(i, j) = sigmasq + nugginside;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));//x.row(i);
      for(unsigned int j=0; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));//y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          res(i, j) = pow(hphi, nu) * pow2_nu1_gammanu_sigmasq *
            R::bessel_k_ex(hphi, nu, 1.0, &bessel_ws[threadid*3]);
        } else {
          res(i, j) = sigmasq + nugginside;
        }
      }
    }
  }
  //return res;
}


// powered exponential nu<2
void powerexp_inplace(arma::mat& res, 
                      const arma::mat& coords,
                      const arma::uvec& ix, const arma::uvec& iy, 
                      const double& phi, const double& nu, 
                      const double& sigmasq, const double& nugg,
                      bool same){
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hh = arma::norm(delta);
        if(hh==0){
          res(i, j) = sigmasq + nugg;
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=0; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hh = arma::norm(delta);
        if(hh==0){
          res(i, j) = sigmasq + nugg;
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq;
        }
      }
    }
  }
}

arma::mat Correlationf(
    const arma::mat& coords,
    const arma::uvec& ix, const arma::uvec& iy,
    const arma::vec& theta,
    double * bessel_ws,
    int covar,
    bool same){
  // these are not actually correlation functions because they are reparametrized to have 
  // C(0) = 1/reparam
  arma::mat res = arma::zeros(ix.n_rows, iy.n_rows);
  
  double phi = theta(0);
  double sigmasq = theta(1);
  double nu = theta(2);
  double nugg = theta(3);
  
  if(covar==0){
    powerexp_inplace(res, coords, ix, iy, phi, nu, sigmasq, nugg, same); 
  } else {
    matern_inplace(res, coords, ix, iy, phi, nu, sigmasq, bessel_ws, nugg, same);    
  }

  return res;
  

}

//[[Rcpp::export]]
arma::mat Correlationc(
    const arma::mat& coordsx,
    const arma::mat& coordsy,
    const arma::vec& theta,
    int covar,
    bool same){

  int nthreads = 0;
#ifdef _OPENMP
  nthreads = omp_get_num_threads();
#endif
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(nthreads*bessel_ws_inc, sizeof(double));
  
  
  if(same){
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    
    return Correlationf(coordsx, ix, ix, theta, bessel_ws, covar, same);
  } else {
    arma::mat coords = arma::join_vert(coordsx, coordsy);
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    arma::uvec iy = arma::regspace<arma::uvec>(coordsx.n_rows, coords.n_rows-1);
    
    return Correlationf(coords, ix, iy, theta, bessel_ws, covar, same);
  }
  
}


//[[Rcpp::export]]
arma::mat gpkernel(
    const arma::mat& coordsx,
    const arma::vec& theta,
    int covar){
  
  int nthreads = 0;
#ifdef _OPENMP
  nthreads = omp_get_num_threads();
#endif
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(nthreads*bessel_ws_inc, sizeof(double));
  
  
  arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
  return Correlationf(coordsx, ix, ix, theta, bessel_ws, covar, true);
}

