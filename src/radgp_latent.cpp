
#include <RcppArmadillo.h>

#include "radgp.h"
#include "rama.h"

using namespace std;

inline Eigen::MatrixXd armamat_to_matrixxd(arma::mat arma_A){
  
  Eigen::MatrixXd eigen_B = Eigen::Map<Eigen::MatrixXd>(arma_A.memptr(),
                                                        arma_A.n_rows,
                                                        arma_A.n_cols);
  
  return eigen_B;
}
inline arma::mat matrixxd_to_armamat(Eigen::MatrixXd eigen_A){
  
  arma::mat arma_B = arma::mat(eigen_A.data(), eigen_A.rows(), eigen_A.cols(),
                               true, false);
  return arma_B;
}


void latent_mcmc(DagGP& adag, 
                   arma::mat& theta_mcmc,
                   arma::mat& w_mcmc,
                   arma::vec& tausq_mcmc,
                   arma::vec& logdens_mcmc,
                   int mcmc, 
                   const arma::vec& theta_init,
                   double tausq_init,
                   double metrop_sd,
                   const arma::mat& theta_unif_bounds,
                   const arma::vec& tausq_prior,
                   int print_every=10){
  
  // timers
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point tstart = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point tend = std::chrono::steady_clock::now();
  // ------
  
  arma::vec theta = theta_init;
  double tausq = tausq_init;
  
  int n = adag.y.n_elem;
  int nt = theta.n_elem;
  arma::mat metrop_theta_sd = metrop_sd * arma::eye(nt, nt);

  // RAMA for theta
  int theta_mcmc_counter = 0;
  RAMAdapt theta_adapt(nt, metrop_theta_sd, 0.24); // +1 for tausq
  bool theta_adapt_active = true;
  
  adag.make_precision_hci(theta);
  
  // proposal data
  Eigen::SparseMatrix<double> H_prop = adag.H;
  Eigen::SparseMatrix<double> Ci_prop = adag.Ci;
  double prec_det_prop;
  
  // mcmc storing
  theta_mcmc = arma::zeros(nt, mcmc);
  tausq_mcmc = arma::zeros(mcmc);
  w_mcmc = arma::zeros(n, mcmc);
  logdens_mcmc = arma::zeros(mcmc);
  
  // solver
  Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
  cg.setTolerance(1e-6);
  Eigen::MatrixXd ye = armamat_to_matrixxd(adag.y);
  
  // timing
  arma::vec timing(4);
  
  bool sample_tausq = arma::any(tausq_prior != 0);
  
  for(int m=0; m<mcmc; m++){
    // w
    //Rcpp::Rcout << "sample w CG " << endl;
    tstart = std::chrono::steady_clock::now();
    
    arma::vec rn1 = arma::randn(n); // arma uses same seed as R
    Eigen::VectorXd rn1_e = armamat_to_matrixxd(rn1);
    arma::vec rn2 = arma::randn(n); 
    
    // pseudocode for sampling w
    Eigen::MatrixXd Htrn1_e = adag.H.transpose() * rn1_e;
    arma::vec Htrn1 = matrixxd_to_armamat(Htrn1_e);
    arma::vec wtemp = 1.0/tausq * adag.y + Htrn1 + rn2/pow(tausq, .5); // add xb stuff
    Eigen::MatrixXd wtemp_e = armamat_to_matrixxd(wtemp);
    
    Eigen::SparseMatrix<double> Phihat = adag.Ci;
    Phihat.diagonal().array() += 1.0/tausq;
    tend = std::chrono::steady_clock::now();
    int time_firstpart = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
    //Rcpp::Rcout << "first part : " << time_firstpart << endl;
    timing(0) += time_firstpart;
    
    tstart = std::chrono::steady_clock::now();
    cg.compute(Phihat);
    
    Eigen::MatrixXd we = cg.solveWithGuess(wtemp_e, ye/tausq);
    tend = std::chrono::steady_clock::now();
    int time_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
    //Rcpp::Rcout << "solve : " << time_solve << endl;
    timing(1) += time_solve;
    
    
    tstart = std::chrono::steady_clock::now();
    adag.w = matrixxd_to_armamat(we);
    Eigen::VectorXd wcore = we.transpose() * adag.Ci * we;
    
    double current_logdens = 0.5 * adag.prec_det - 0.5 * wcore(0);
    
    tend = std::chrono::steady_clock::now();
    int time_product = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
    //Rcpp::Rcout << "product : " << time_product << endl;
    timing(2) += time_product;
    
    // theta
    //Rcpp::Rcout << "update theta " << endl;
    arma::vec U_update = arma::randn(nt);
    
    theta_adapt.count_proposal();
    
    arma::vec new_param = par_huvtransf_back(par_huvtransf_fwd(theta, theta_unif_bounds) + 
      theta_adapt.paramsd * U_update, theta_unif_bounds);
    
    tstart = std::chrono::steady_clock::now();
    adag.make_precision_hci_core(H_prop, Ci_prop, prec_det_prop, new_param);
    tend = std::chrono::steady_clock::now();
    int time_secondpart = std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart).count();
    //Rcpp::Rcout << "second part : " << time_secondpart << endl;
    timing(3) += time_secondpart;
    
    
    Eigen::VectorXd wcore_prop = we.transpose() * Ci_prop * we;
    double proposal_logdens = 0.5 * prec_det_prop - 0.5 * wcore_prop(0);
    
    double prior_logratio = 
      calc_prior_logratio(new_param.subvec(1,2), theta.subvec(1,2)); // ig
    double jacobian = calc_jacobian(new_param, theta, theta_unif_bounds);
    
    double logaccept = proposal_logdens - current_logdens + 
      prior_logratio + jacobian;
    
    bool accepted = do_I_accept(logaccept);
    if(accepted){
      theta = new_param;
      current_logdens = proposal_logdens;
      std::swap(adag.Ci, Ci_prop);
      std::swap(adag.H, H_prop);
      adag.prec_det = prec_det_prop;
      theta_adapt.count_accepted();
    } 
    
    theta_adapt.update_ratios();
    if(theta_adapt_active){
      theta_adapt.adapt(U_update, exp(logaccept), theta_mcmc_counter); 
    }
    theta_mcmc_counter++;

    if(sample_tausq){
      // tausq
      double aprior = tausq_prior(0);
      double bprior = tausq_prior(1);
      
      arma::mat yrr = adag.y - 
        //XB
        adag.w;
      
      double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
      double aparam = aprior + n/2.0;
      double bparam = 1.0/( bprior + .5 * bcore );
      
      Rcpp::RNGScope scope;
      tausq = 1.0/R::rgamma(aparam, bparam);
      //logpost += 0.5 * (nr + .0) * log(1.0/tausq) - 0.5/tausq*bcore;
    }

    // store
    
    logdens_mcmc(m) = current_logdens;
    theta_mcmc.col(m) = theta;
    w_mcmc.col(m) = adag.w;
    tausq_mcmc(m) = tausq;
    
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
  //Rcpp::Rcout << timing.t() << endl;
  //Rcpp::Rcout << arma::accu(timing)/1e9 << endl;
}

//[[Rcpp::export]]
Rcpp::List radgp_latent(const arma::vec& y,
                    const arma::mat& coords,
                    double rho,
                    int mcmc,
                    int num_threads,
                    const arma::vec& theta_init,
                    double tausq_init,
                    double metrop_sd,
                    const arma::mat& theta_unif_bounds,
                    const arma::vec& tausq_prior,
                    int covar,
                    int num_prints = 10){
  
  int print_every = num_prints>0? round(mcmc/num_prints) : 0;
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  DagGP adag(y, coords, rho, 1, covar, num_threads); // 1 for latent
  
  arma::mat theta_mcmc;
  arma::mat w_mcmc;
  arma::vec tausq_mcmc;
  arma::vec logdens_mcmc;
  
  latent_mcmc(adag, theta_mcmc, w_mcmc, tausq_mcmc, 
              logdens_mcmc, 
              mcmc, theta_init, tausq_init,
              metrop_sd, theta_unif_bounds, tausq_prior, print_every);

  return Rcpp::List::create(
    Rcpp::Named("M") = adag.M,
    Rcpp::Named("rho") = rho,
    Rcpp::Named("dag") = adag.dag,
    Rcpp::Named("y") = y,
    Rcpp::Named("coords") = coords,
    Rcpp::Named("ldens") = logdens_mcmc,
    Rcpp::Named("theta") = theta_mcmc,
    Rcpp::Named("w") = w_mcmc,
    Rcpp::Named("nugg") = tausq_mcmc,
    Rcpp::Named("covar") = covar
  );
}


//[[Rcpp::export]]
Rcpp::List daggp_custom_latent(const arma::vec& y,
                             const arma::mat& coords,
                             const arma::field<arma::uvec>& dag,
                             int mcmc,
                             int num_threads,
                             const arma::vec& theta_init,
                             double tausq_init,
                             double metrop_sd,
                             const arma::mat& theta_unif_bounds,
                             const arma::vec& tausq_prior,
                             int covar,
                             int num_prints = 10){
  
  int print_every = num_prints>0? round(mcmc/num_prints) : 0;
  
  
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  
  DagGP adag(y, coords, dag, 1, covar, num_threads);
  
  arma::mat theta_mcmc;
  arma::mat w_mcmc;
  arma::vec tausq_mcmc;
  arma::vec logdens_mcmc;
  
  latent_mcmc(adag, theta_mcmc, w_mcmc, tausq_mcmc, 
              logdens_mcmc, 
              mcmc, theta_init, tausq_init, 
              metrop_sd, theta_unif_bounds, 
              tausq_prior, print_every);
    
    return Rcpp::List::create(
      Rcpp::Named("M") = adag.M,
      Rcpp::Named("dag") = adag.dag,
      Rcpp::Named("ldens") = logdens_mcmc,
      Rcpp::Named("theta") = theta_mcmc,
      Rcpp::Named("w") = w_mcmc,
      Rcpp::Named("nugg") = tausq_mcmc,
      Rcpp::Named("covar") = covar
    );
}


