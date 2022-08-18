#include "covariance.h"

using namespace std;



// powered exponential nu<2
void powerexp_inplace(arma::mat& res, 
                      const arma::mat& coords,
                      const arma::uvec& ix, const arma::uvec& iy, 
                      const double& phi, const double& nu, 
                      const double& sigmasq, const double& nugg, const double& reparam,
                      bool same){
  
  double sigmasq_reparam = sigmasq / reparam;
  
  if(same){
    for(unsigned int i=0; i<ix.n_rows; i++){
      arma::rowvec cri = coords.row(ix(i));
      for(unsigned int j=i; j<iy.n_rows; j++){
        arma::rowvec delta = cri - coords.row(iy(j));
        double hh = arma::norm(delta);
        if(hh==0){
          res(i, j) = sigmasq_reparam + nugg;
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq_reparam;
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
          res(i, j) = sigmasq_reparam + nugg;
        } else {
          double hnuphi = pow(hh, nu) * phi;
          res(i, j) = exp(-hnuphi) * sigmasq_reparam;
        }
      }
    }
  }
}

arma::mat Correlationf(
    const arma::mat& coords,
    const arma::uvec& ix, const arma::uvec& iy,
    const arma::vec& theta,
    bool ps, bool same){
  // these are not actually correlation functions because they are reparametrized to have 
  // C(0) = 1/reparam
  arma::mat res = arma::zeros(ix.n_rows, iy.n_rows);
  
  double phi = theta(0);
  double nu = 1.0;//theta(1);
  double sigmasq = theta(1);
  double nugg = theta(2);
  
  double reparam = 1.0; 
  
  powerexp_inplace(res, coords, ix, iy, phi, nu, sigmasq, nugg, reparam, same);
  return res;
  

}

//[[Rcpp::export]]
arma::mat Correlationc(
    const arma::mat& coordsx,
    const arma::mat& coordsy,
    const arma::vec& theta,
    bool same){
  // inefficient
  bool ps = false;
  
  if(same){
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    
    return Correlationf(coordsx, ix, ix, theta, ps, same);
  } else {
    arma::mat coords = arma::join_vert(coordsx, coordsy);
    arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
    arma::uvec iy = arma::regspace<arma::uvec>(coordsx.n_rows, coords.n_rows-1);
    
    return Correlationf(coords, ix, iy, theta, ps, same);
  }
  
}


//[[Rcpp::export]]
arma::mat gpkernel(
    const arma::mat& coordsx,
    const arma::vec& theta){
  arma::uvec ix = arma::regspace<arma::uvec>(0, coordsx.n_rows-1);
  return Correlationf(coordsx, ix, ix, theta, false, true);
}

