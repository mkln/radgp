#define ARMA_DONT_PRINT_ERRORS
#include <RcppArmadillo.h>

#include "altdag.h"
#include "predict.h"
#include "rama.h"

using namespace std;


//[[Rcpp::export]]
Rcpp::List altdaggp(const arma::mat& coords,
                    const arma::vec& theta,
                    double rho){
  
  arma::vec y = arma::randn(coords.n_rows);
  AltDAG adag(y, coords, rho);
  
  adag.make_precision(theta);

  double ldens = adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = ldens,
    Rcpp::Named("H") = adag.H,
    Rcpp::Named("layers") = adag.layers
  );
}


//[[Rcpp::export]]
Rcpp::List vecchiagp(const arma::mat& coords,
                    const arma::vec& theta,
                    const arma::field<arma::uvec>& dag){
  
  arma::vec y = arma::randn(coords.n_rows);
  AltDAG adag(y, coords, dag);
  adag.make_precision(theta);
  double ldens = adag.logdens(theta);
  
  return Rcpp::List::create(
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("ldens") = ldens,
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
  
  double current_logdens = adag.logdens(theta);
  
  theta_mcmc = arma::zeros(nt, mcmc);
  logdens_mcmc = arma::zeros(mcmc);
  
  for(int m=0; m<mcmc; m++){
    Rcpp::RNGScope scope;
    arma::vec U_update = arma::randn(nt);
    
    theta_adapt.count_proposal();
    
    arma::vec new_param = par_huvtransf_back(par_huvtransf_fwd(theta, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
    
    double proposal_logdens = adag.logdens(new_param);
    
    double prior_logratio = 
      calc_prior_logratio(new_param.subvec(1,2), theta.subvec(1,2)); // ig
    double jacobian = calc_jacobian(new_param, theta, theta_unif_bounds);
    
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
    Rcpp::Named("rho") = rho,
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("y") = y,
    Rcpp::Named("coords") = coords,
    Rcpp::Named("ldens") = logdens_mcmc,
    Rcpp::Named("theta") = theta_mcmc
  );
}

//[[Rcpp::export]]
Rcpp::List altdaggp_response_predict(const arma::mat& cout,
                                     const arma::vec& y, 
                                     const arma::mat& coords, double rho,
                                     const arma::mat& theta_mcmc, 
                                     int M,
                                     int num_threads){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int ntrain = coords.n_rows;
  int ntest = cout.n_rows;
  int mcmc = theta_mcmc.n_cols;
  arma::mat cxall = arma::join_vert(coords, cout);
  
  arma::uvec layers;
  arma::field<arma::uvec> predict_dag = 
    altdagbuild_testset(coords, cout, rho, layers, M);
  arma::uvec pred_order = arma::sort_index(layers);
  
  arma::mat yout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  
  Rcpp::Rcout << "predicting " << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    
    arma::vec ytemp = arma::zeros(ntrain+ntest);
    ytemp.subvec(0, ntrain-1) = y;
    for(int i=0; i<ntest; i++){
      
      int itarget = pred_order(ntrain+i);
      int idagtarget = itarget - ntrain;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = predict_dag(idagtarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, false, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, false, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, false, true) );
      
      arma::vec ht = PPi * CPt;
      double sqrtR = sqrt( arma::conv_to<double>::from(
        CC - CPt.t() * ht ) );
      
      ytemp(itarget) = arma::conv_to<double>::from(
        ht.t() * ytemp(px) + random_stdnormal(m, i) * sqrtR );
      
      yout_mcmc(itarget-ntrain, m) = ytemp(itarget);
    }

  }
  
  return Rcpp::List::create(
    Rcpp::Named("yout") = yout_mcmc,
    Rcpp::Named("predict_dag") = predict_dag,
    Rcpp::Named("layers") = layers,
    Rcpp::Named("pred_order") = pred_order
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



//[[Rcpp::export]]
double daggp_negdens(const arma::vec& y,
                           const arma::mat& coords,
                           const arma::field<arma::uvec>& dag,
                           const arma::vec& theta,
                           int num_threads){
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  AltDAG adag(y, coords, dag);
  
  return adag.logdens(theta);
}


