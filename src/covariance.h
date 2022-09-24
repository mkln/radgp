
#ifndef XCOV_LMC 
#define XCOV_LMC

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

using namespace std;


arma::mat Correlationf(const arma::mat& coords, const arma::uvec& ix, const arma::uvec& iy, 
                       const arma::vec& theta, bool ps, bool same);

arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, 
                       const arma::vec& theta, bool ps, bool same);

#endif
