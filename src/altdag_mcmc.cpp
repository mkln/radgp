#define ARMA_DONT_PRINT_ERRORS
#include "R.h"
#include <RcppArmadillo.h>

#include "altdag.h"
#include "rama.h"

using namespace std;

inline static void chkIntFn(void *dummy) {
  R_CheckUserInterrupt();
}

// this will call the above in a top-level context so it won't longjmp-out of your context
inline bool checkInterrupt() {
  return (R_ToplevelExec(chkIntFn, NULL) == FALSE);
}


//[[Rcpp::export]]
Rcpp::List altdaggp(const arma::mat& coords,
                    const arma::vec& theta,
                    double rho){
  
  arma::vec y = arma::randn(coords.n_rows);
  AltDAG adag(y, coords, rho);
  
  adag.make_precision(theta);

  adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = adag.ldens,
    Rcpp::Named("H") = adag.H
  );
}

void response_mcmc(AltDAG& adag, 
                   arma::mat& theta_mcmc,
                   arma::vec& logdens_mcmc,
                   int mcmc, 
                   const arma::vec& theta_init,
                   double metrop_sd,
                   const arma::mat& theta_unif_bounds){
  
  // timers
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  // ------
  
  int print_every = 100;
  
  arma::vec theta = theta_init;
  
  int nt = theta.n_elem;
  arma::mat metrop_theta_sd = metrop_sd * arma::eye(nt, nt);
  
  // RAMA for theta
  int theta_mcmc_counter = 0;
  RAMAdapt theta_adapt(nt, metrop_theta_sd, 0.24); // +1 for tausq
  bool theta_adapt_active = true;
  
  adag.logdens(theta);
  double current_logdens = adag.ldens;
  
  theta_mcmc = arma::zeros(nt, mcmc);
  logdens_mcmc = arma::zeros(mcmc);
  
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
    
    logdens_mcmc(m) = current_logdens;
    theta_mcmc.col(m) = theta;
    
    
    bool interrupted = checkInterrupt();
    if(interrupted){
      Rcpp::stop("Interrupted by the user.");
    }
    
    if((m>0) & (mcmc > 100)){
      bool print_condition = (print_every>0);
      if(print_condition){
        print_condition = print_condition & (!(m % print_every));
      };
      
      if(print_condition){
        end_mcmc = std::chrono::steady_clock::now();
        
        int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
        int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
        theta_adapt.print_summary(time_tick, time_mcmc, m, mcmc);
        
        tick_mcmc = std::chrono::steady_clock::now();
        
        unsigned int printlimit = 10;
        
        theta_adapt.print_acceptance();
        //Rprintf("\n\n");
      } 
    } else {
      tick_mcmc = std::chrono::steady_clock::now();
    }
    
  }
  
  
}

//[[Rcpp::export]]
Rcpp::List altdaggp_response(const arma::vec& y,
                    const arma::mat& coords,
                    double rho,
                    int mcmc,
                    int num_threads,
                    const arma::vec& theta_init,
                    double metrop_sd,
                    const arma::mat& theta_unif_bounds){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  AltDAG adag(y, coords, rho);
  
  arma::mat theta_mcmc;
  arma::vec logdens_mcmc;
  
  response_mcmc(adag, theta_mcmc, logdens_mcmc, mcmc, theta_init, metrop_sd, theta_unif_bounds);

  return Rcpp::List::create(
    Rcpp::Named("M") = adag.M,
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = logdens_mcmc,
    Rcpp::Named("theta") = theta_mcmc
  );
}


//[[Rcpp::export]]
Rcpp::List altdaggp_custom(const arma::vec& y,
                             const arma::mat& coords,
                             const arma::field<arma::uvec>& dag,
                             int mcmc,
                             int num_threads,
                             const arma::vec& theta_init,
                             double metrop_sd,
                             const arma::mat& theta_unif_bounds){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  AltDAG adag(y, coords, dag);
  
  arma::mat theta_mcmc;
  arma::vec logdens_mcmc;
  
  response_mcmc(adag, theta_mcmc, logdens_mcmc, mcmc, theta_init, metrop_sd, theta_unif_bounds);
    
    return Rcpp::List::create(
      Rcpp::Named("M") = adag.M,
      Rcpp::Named("dag") = adag.dag,
      Rcpp::Named("ldens") = logdens_mcmc,
      Rcpp::Named("theta") = theta_mcmc
    );
}