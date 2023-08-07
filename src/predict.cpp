#include "predict.h"


arma::umat make_candidates_testset(const arma::mat& w, 
                                   const arma::uvec& indsort, arma::uvec& testindsort,
                                   unsigned int col,
                                   double rho){
  arma::uvec colsel = col + arma::zeros<arma::uvec>(1);
  //arma::vec wsort = w.submat(indsort, colsel);
  
  int nr = indsort.n_elem;
  int ntest = testindsort.n_elem;
  arma::umat candidates = arma::zeros<arma::umat>(ntest, 2);
  int left = 0;
  int right = 0;
  
  for(unsigned int loc = 0; loc<ntest; loc++){
    while((w(testindsort(loc), col) - w(indsort(left), col)) > rho){
      left ++;
    }
    if(right < nr - 1){
      while(w(indsort(right+1)) - w(testindsort(loc)) <= rho){
        right ++;
        if(right == nr - 1){
          break;
        }
      }
    }
    candidates(loc, 0) = left;
    candidates(loc, 1) = right;
  }
  return candidates;
}

//[[Rcpp::export]]
arma::field<arma::uvec> neighbor_search_testset(const arma::mat& wtrain, 
                                                const arma::mat& wtest, double rho){
  int ntest = wtest.n_rows;
  int ntrain = wtrain.n_rows;
  int nr = ntest+ntrain;
  
  double rho2 = rho*rho;
  
  arma::mat w = arma::join_vert(wtrain, wtest);
  
  arma::uvec indsort = arma::sort_index(w.col(0));
  arma::uvec testindsort = indsort(arma::find(indsort >= ntrain));
  arma::umat candidates = arma::zeros<arma::umat>(ntest, 2);
  candidates = make_candidates_testset(w, indsort, testindsort, 0, rho);
  
  arma::field<arma::uvec> Nset(ntest);
  for(unsigned int i=0; i<ntest; i++){
    int left = candidates(i, 0);
    int right = candidates(i, 1);
    int rightx = right==(ntest-1)? ntest-1 : right;
    arma::uvec try_ids = indsort.rows(left, rightx);
    try_ids = try_ids(arma::find(try_ids != testindsort(i)));
    
    int ni = try_ids.n_elem;
    arma::vec try_dist2 = arma::zeros(ni);
    for(unsigned int j=0; j<ni; j++){
      arma::rowvec rdiff = w.row(try_ids(j)) - w.row(testindsort(i));
      try_dist2(j) = arma::accu(rdiff % rdiff);
    }
    Nset(testindsort(i)-ntrain) = try_ids.rows(arma::find(try_dist2<=rho2));
  }
  return Nset;
}

arma::uvec sort_test(const arma::mat wtrain, const arma::mat wtest){
  int ntrain = wtrain.n_rows;
  int ntest = wtest.n_rows;
  arma::rowvec center = arma::mean(wtrain);
  arma::mat diffs_test = wtest - arma::ones<arma::colvec>(ntest) * center;
  arma::colvec c_dist2_test(ntest);
  for(int i=0; i<ntest; i++){
    c_dist2_test(i) = arma::dot(diffs_test.row(i),diffs_test.row(i));
  }
  arma::uvec sorted_id = arma::sort_index(c_dist2_test);
  return sorted_id;
}

arma::field<arma::uvec> dagbuild_from_nn_testset(const arma::field<arma::uvec>& Rset, 
                                                 const arma::mat& wtrain, const arma::mat& wtest){
  int ntrain = wtrain.n_rows;
  int ntest = wtest.n_rows;
  int nall = ntrain + ntest;
  arma::mat wall = arma::join_cols(wtrain, wtest);
  
  arma::uvec sorted_id = sort_test(wtrain, wtest);
  arma::uvec v1ntrain = arma::linspace<arma::uvec>(0,ntrain-1,ntrain);
  arma::uvec sorted_id_all = arma::join_cols(v1ntrain, sorted_id+ntrain);
  arma::uvec sorted_order_all(nall);
  for(int i=0; i<nall; i++){
    sorted_order_all(sorted_id_all(i)) = i;
  }
  arma::field<arma::uvec> Nset(ntest);
  for(int i=0; i<ntest; i++){
    Nset(i) = Rset(i)(arma::find(sorted_order_all(Rset(i))<sorted_order_all(i+ntrain)));
  }
  for(int i=0; i<ntest; i++){
    if(Nset(sorted_id(i)).n_elem == 0){
      arma::uvec inds_b = sorted_id_all.subvec(0,i+ntrain-1);
      arma::mat diffs_b = wall.rows(inds_b) - arma::ones<arma::colvec>(i+ntrain) * wtest.row(sorted_id(i));
      arma::colvec c_dist2_b(i+ntrain);
      for(int j=0; j<i; j++){
        c_dist2_b(j) = arma::dot(diffs_b.row(j),diffs_b.row(j));
      }
      Nset(i) = c_dist2_b.index_min();
    }
  }
  return Nset;
}

//[[Rcpp::export]]
arma::field<arma::uvec> radgpbuild_testset(const arma::mat& wtrain,
                                           const arma::mat& wtest, 
                                           double rho){
  arma::field<arma::uvec> Rset = neighbor_search_testset(wtrain, wtest, rho);
  arma::field<arma::uvec> dag = dagbuild_from_nn_testset(Rset, wtrain, wtest);
  return dag;
}


//[[Rcpp::export]]
Rcpp::List radial_neighbors_dag_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho){
  arma::field<arma::uvec> dag = radgpbuild_testset(wtrain, wtest, rho);
  arma::uvec sorted_id = sort_test(wtrain, wtest);
  return Rcpp::List::create(
    Rcpp::Named("dag") = dag,
    Rcpp::Named("sorted_id") = sorted_id
  );
}

//[[Rcpp::export]]
Rcpp::List radgp_response_predict(const arma::mat& cout,
                                     const arma::vec& y, 
                                     const arma::mat& coords, double rho,
                                     const arma::mat& theta_mcmc, 
                                     int covar,
                                     int num_threads){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  //thread safe stuff
  //int nthreads = 0;
  //#ifdef _OPENMP
  //nthreads = omp_get_num_threads();
  //#endif
  
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int ntrain = coords.n_rows;
  int ntest = cout.n_rows;
  int mcmc = theta_mcmc.n_cols;
  arma::mat cxall = arma::join_vert(coords, cout);
  
  arma::uvec layers;
  arma::field<arma::uvec> predict_dag = 
    radgpbuild_testset(coords, cout, rho);
  arma::uvec pred_order = sort_test(coords, cout);
  
  arma::mat yout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  
  Rcpp::Rcout << "RadGP predicting (response model)" << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    
    arma::vec ytemp = arma::zeros(ntrain+ntest);
    ytemp.subvec(0, ntrain-1) = y;
    for(int i=0; i<ntest; i++){
      
      int itarget = pred_order(i) + ntrain;
      int idagtarget = itarget - ntrain;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = predict_dag(idagtarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, bessel_ws, covar, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, covar, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, covar, true) );
      
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
    Rcpp::Named("dag") = predict_dag,
    Rcpp::Named("pred_order") = pred_order
  );
}

//[[Rcpp::export]]
arma::mat vecchiagp_response_predict(const arma::mat& cout,
                                     const arma::vec& y, 
                                     const arma::mat& coords, 
                                     const arma::field<arma::uvec>& dag,
                                     const arma::mat& theta_mcmc, 
                                     int covar,
                                     int num_threads){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  //thread safe stuff
  //int nthreads = 0;
  //#ifdef _OPENMP
  //nthreads = omp_get_num_threads();
  //#endif
  
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int ntrain = coords.n_rows;
  int ntest = cout.n_rows;
  int mcmc = theta_mcmc.n_cols;
  arma::mat cxall = arma::join_vert(coords, cout);
  
  arma::mat yout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  
  arma::vec ytemp = arma::zeros(ntrain+ntest);
  ytemp.subvec(0, ntrain-1) = y;
  
  Rcpp::Rcout << "VecchiaGP predicting (response model)" << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    
    for(int i=0; i<ntest; i++){
      
      int itarget = ntrain+i;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = dag(itarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, bessel_ws, covar, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, covar, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, covar, true) );
      
      arma::vec ht = PPi * CPt;
      double sqrtR = sqrt( arma::conv_to<double>::from(
        CC - CPt.t() * ht ) );
      
      ytemp(itarget) = arma::conv_to<double>::from(
        ht.t() * ytemp(px) + random_stdnormal(m, i) * sqrtR );
      
      yout_mcmc(itarget-ntrain, m) = ytemp(itarget);
    }
    
  }
  
  return yout_mcmc;
}


//[[Rcpp::export]]
Rcpp::List radgp_latent_predict(const arma::mat& cout,
                                     const arma::mat& w, 
                                     const arma::mat& coords, double rho,
                                     const arma::mat& theta_mcmc, 
                                     int covar,
                                     int num_threads){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  //thread safe stuff
  //int nthreads = 0;
  //#ifdef _OPENMP
  //nthreads = omp_get_num_threads();
  //#endif
  
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int ntrain = coords.n_rows;
  int ntest = cout.n_rows;
  int mcmc = theta_mcmc.n_cols;
  arma::mat cxall = arma::join_vert(coords, cout);
  
  arma::uvec layers;
  arma::field<arma::uvec> predict_dag = 
    radgpbuild_testset(coords, cout, rho);
  arma::uvec pred_order = sort_test(coords, cout);
  
  arma::mat wout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  arma::vec wtemp = arma::zeros(ntrain+ntest);
  
  Rcpp::Rcout << "RadGP predicting (latent model)" << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    wtemp.subvec(0, ntrain-1) = w.col(m);
    
    for(int i=0; i<ntest; i++){
      
      int itarget = pred_order(i) + ntrain;
      int idagtarget = itarget - ntrain;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = predict_dag(idagtarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, bessel_ws, covar, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, covar, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, covar, true) );
      
      arma::vec ht = PPi * CPt;
      double sqrtR = sqrt( arma::conv_to<double>::from(
        CC - CPt.t() * ht ) );
      
      wtemp(itarget) = arma::conv_to<double>::from(
        ht.t() * wtemp(px) + random_stdnormal(m, i) * sqrtR );
      
      wout_mcmc(itarget-ntrain, m) = wtemp(itarget);
    }
    
  }
  
  return Rcpp::List::create(
    Rcpp::Named("wout") = wout_mcmc,
    Rcpp::Named("dag") = predict_dag,
    Rcpp::Named("pred_order") = pred_order
  );
}

//[[Rcpp::export]]
arma::mat vecchiagp_latent_predict(const arma::mat& cout,
                                     const arma::mat& w, 
                                     const arma::mat& coords, 
                                     const arma::field<arma::uvec>& dag,
                                     const arma::mat& theta_mcmc, 
                                     int covar,
                                     int num_threads){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  //thread safe stuff
  //int nthreads = 0;
  //#ifdef _OPENMP
  //nthreads = omp_get_num_threads();
  //#endif
  
  int bessel_ws_inc = 5;//see bessel_k.c for working space needs
  double *bessel_ws = (double *) R_alloc(num_threads*bessel_ws_inc, sizeof(double));
  
  int ntrain = coords.n_rows;
  int ntest = cout.n_rows;
  int mcmc = theta_mcmc.n_cols;
  arma::mat cxall = arma::join_vert(coords, cout);
  
  arma::mat wout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  
  arma::vec wtemp = arma::zeros(ntrain+ntest);

  Rcpp::Rcout << "VecchiaGP predicting (latent model)" << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    wtemp.subvec(0, ntrain-1) = w.col(m);
    
    for(int i=0; i<ntest; i++){
      
      int itarget = ntrain+i;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = dag(itarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, bessel_ws, covar, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, covar, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, covar, true) );
      
      arma::vec ht = PPi * CPt;
      double sqrtR = sqrt( arma::conv_to<double>::from(
        CC - CPt.t() * ht ) );
      
      wtemp(itarget) = arma::conv_to<double>::from(
        ht.t() * wtemp(px) + random_stdnormal(m, i) * sqrtR );
      
      wout_mcmc(itarget-ntrain, m) = wtemp(itarget);
    }
    
  }
  
  return wout_mcmc;
}
