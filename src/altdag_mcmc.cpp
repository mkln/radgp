#define ARMA_DONT_PRINT_ERRORS

#include <RcppArmadillo.h>

#include "altdag.h"
#include "rama.h"

using namespace std;

//[[Rcpp::export]]
Rcpp::List altdaggp(const arma::vec& y,
                    const arma::mat& coords,
                    const arma::vec& theta,
                    double rho){
  
  AltDAG adag(y, coords, rho);
  
  Rcpp::Rcout << "make precision " << endl;
  adag.make_precision(theta);
  
  Rcpp::Rcout << "logdens " << endl;
  adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("ldens") = adag.ldens,
    Rcpp::Named("H") = adag.H
  );
}


//[[Rcpp::export]]
Rcpp::List altdaggp_response(const arma::vec& y,
                    const arma::mat& coords,
                    double rho, 
                    int mcmc,
                    int num_threads,
                    const arma::vec& theta_init,
                    double metrop_sd,
                    const arma::mat& theta_unif_bounds ){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  
  arma::vec theta = theta_init;
  
  int nt = theta.n_elem;
  arma::mat metrop_theta_sd = metrop_sd * arma::eye(nt, nt);
  

  // RAMA for theta
  int theta_mcmc_counter = 0;

  RAMAdapt theta_adapt(nt, metrop_theta_sd, 0.24); // +1 for tausq
  bool theta_adapt_active = true;

  AltDAG adag(y, coords, rho);
  
  adag.logdens(theta);
  double current_logdens = adag.ldens;
  
  arma::mat theta_mcmc = arma::zeros(nt, mcmc);
  arma::vec logdens_save = arma::zeros(mcmc);
  
  for(int m=0; m<mcmc; m++){
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(nt);
    
    theta_adapt.count_proposal();
    
    arma::vec new_param = par_huvtransf_back(par_huvtransf_fwd(theta, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
    
    adag.logdens(new_param);
    double proposal_logdens = adag.ldens;
  
    double prior_logratio = 0;
    double jacobian  = calc_jacobian(new_param, theta, theta_unif_bounds);
    
    double logaccept = proposal_logdens - current_logdens + 
      prior_logratio + jacobian;
    
    bool accepted = do_I_accept(logaccept);
    if(accepted){
      theta = new_param;
      current_logdens = proposal_logdens;
      theta_adapt.count_accepted();
    }
    
    theta_adapt.update_ratios();
    if(theta_adapt_active){
      theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
    }
    theta_mcmc_counter++;
    
    logdens_save(m) = current_logdens;
    theta_mcmc.col(m) = theta;
    
    Rcpp::Rcout << "logdens " << current_logdens << endl;
  
  }
  
  return Rcpp::List::create(
    Rcpp::Named("ldens") = logdens_save,
    Rcpp::Named("theta") = theta_mcmc
  );
}