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

//[[Rcpp::export]]
arma::field<arma::uvec> dagbuild_from_nn_testset(const arma::field<arma::uvec>& Rset, 
                                                 int ntrain, arma::uvec& layers,
                                                 int Mmin, const arma::mat& wtrain, const arma::mat& wtest){
  int ntest = Rset.n_elem;
  int nr = ntrain + ntest;
  int M = Mmin;
  
  layers = arma::zeros<arma::uvec>(nr);
  arma::field<arma::uvec> R_layers(nr);
  
  std::deque<int> queue;
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int i=ntrain; i<nr; i++){
    if(!(layers(i) == 0)){
      continue;
    } else {
      int l = i;
      for(int k=Mmin; k<M+2; k++){
        arma::uvec found_elem = arma::find(R_layers(l) == k);
        if(found_elem.n_elem == 0){ 
          layers(l) = k;
          break;
        }
      }
      M = layers(l) > M ? layers(l) : M;
      for(int jx=0; jx < Rset(l-ntrain).n_elem; jx++){
        int j = Rset(l-ntrain)(jx);
        if(j > ntrain){
          R_layers(j) = arma::join_vert(R_layers(j), oneuv*layers(l));
          if(layers(j) == 0){
            queue.push_back(j); 
          }
        }
      } 
      
      while(queue.size()>0){
        l = queue.front();
        queue.pop_front();
        if(layers(l) > 0){
          continue;
        }
        for(int k=Mmin; k<M+2; k++){
          arma::uvec found_elem = arma::find(R_layers(l) == k);
          if(found_elem.n_elem == 0){ //**
            layers(l) = k;
            break;
          }
        }
        M = layers(l) > M ? layers(l) : M;
        for(int jx=0; jx<Rset(l-ntrain).n_elem; jx++){
          int j = Rset(l-ntrain)(jx);
          if(j > ntrain){
            R_layers(j) = arma::join_vert(R_layers(j), oneuv*layers(l));
            if(layers(j) == 0){
              queue.push_back(j); 
            }
          }
        } 
      }
    }
  }
  arma::field<arma::uvec> Nset(ntest);
  for(int i=0; i<ntest; i++){
    arma::uvec layers_ne(Rset(i).n_elem);
    for(int jx=0; jx<Rset(i).n_elem; jx++){
      int x = Rset(i)(jx);
      layers_ne(jx) = layers(x);
    }
    Nset(i) = Rset(i)(arma::find(layers_ne<layers(i+ntrain)));
  }

  // add training parents for testing locations with no parents
  arma::uvec indsort0 = arma::sort_index(wtrain.col(0));
  for (int ni=1; ni<ntest; ni++){
    if (Nset(ni).n_elem == 0){

      // first pass: get closest training parent
      int left=0;
      int right=ntrain-1;
      int mid=(int)round((left+right)/2);
      while(right-left>1){
        if (wtrain(indsort0(mid),0)<=wtest(ni,0)){
          left = mid;
          mid=(int)round((left+right)/2);
        } else {
          right= mid;
          mid=(int)round((left+right)/2);
        }
      }
      int left0 = left;
      int right0 = right;
      arma::rowvec rdiff = wtest.row(ni) - wtrain.row(indsort0(left));
      double rdist = sqrt(arma::accu(rdiff % rdiff));
      int s0 = left;
      double dist0 = rdist;
      rdiff = wtest.row(ni) - wtrain.row(indsort0(right));
      rdist = sqrt(arma::accu(rdiff % rdiff));
      if (rdist < dist0){
        s0 = right;
        dist0 = rdist;
      }
      double adist_l = wtest(ni,0) - wtrain(indsort0(left),0);
      double adist_r = wtrain(indsort0(right),0) - wtest(ni,0);
      double adist = std::min(adist_l, adist_r);
      while (adist < dist0){
        if (adist_l < adist_r){
          if (left==0){
            adist_l = dist0*2;
          } else{
            left--;
            arma::rowvec rdiff = wtest.row(ni) - wtrain.row(indsort0(left));
            double rdist = sqrt(arma::accu(rdiff % rdiff));
            if (rdist < dist0){
              s0 = left;
              dist0 = rdist;
            }
            adist_l = wtest(ni,0) - wtrain(indsort0(left),0);
          }
        } else{
          if (right==ntrain-1){
            adist_r = dist0*2;
          } else{
            right++;
            arma::rowvec rdiff = wtest.row(ni) - wtrain.row(indsort0(right));
            double rdist = sqrt(arma::accu(rdiff % rdiff));
            if (rdist < dist0){
              s0 = right;
              dist0 = rdist;
            }
            adist_r = wtrain(indsort0(right),0) - wtest(ni,0);            
          }
        }
        adist = std::min(adist_l, adist_r);
      }

      // second pass: get closest training location suject to angle constraints 
      double cos_thre = sqrt(2)/2;
      arma::rowvec nv = (wtrain.row(indsort0(s0)) - wtest.row(ni)) / dist0;
      left = left0;
      right = right0;
      double dist1 = std::numeric_limits<double>::infinity();
      int s1 = -1;
      rdiff = wtrain.row(indsort0(left)) - wtest.row(ni);
      rdist = sqrt(arma::accu(rdiff % rdiff));
      if (arma::accu(rdiff % nv) < rdist*cos_thre and rdist < dist1){
        s1 = left;
        dist1 = rdist;       
      }
      rdiff = wtrain.row(indsort0(right)) - wtest.row(ni);
      rdist = sqrt(arma::accu(rdiff % rdiff));
      if (arma::accu(rdiff % nv) < rdist*cos_thre and rdist < dist1){
        s1 = right;
        dist1 = rdist;       
      }
      adist_l = wtest(ni,0) - wtrain(indsort0(left),0);
      adist_r = wtrain(indsort0(right),0) - wtest(ni,0);
      adist = std::min(adist_l, adist_r);
      while (adist < dist1){
        if (adist_l < adist_r){
          if (left==0){
            adist_l = dist1*2;
          } else{
            left--;
            arma::rowvec rdiff = wtrain.row(indsort0(left)) - wtest.row(ni);
            rdist = sqrt(arma::accu(rdiff % rdiff));
            if (arma::accu(rdiff % nv) < rdist*cos_thre and rdist < dist1){
              s1 = left;
              dist1 = rdist;
            }
            adist_l = wtest(ni,0) - wtrain(indsort0(left),0);
          }
        } else{
          if (right==ntrain-1){
            adist_r = dist1*2;
          } else{
            right++;
            arma::rowvec rdiff = wtrain.row(indsort0(right)) - wtest.row(ni);
            rdist = sqrt(arma::accu(rdiff % rdiff));
            if (arma::accu(rdiff % nv) < rdist*cos_thre and rdist < dist1){
              s1 = right;
              dist1 = rdist;  
            }
            adist_r = wtrain(indsort0(right),0) - wtest(ni,0);            
          }
        }
        adist = std::min(adist_l, adist_r);
      }

      // add elements to parent set
      if (s1 == -1){
        Nset(ni) = arma::uvec(1,arma::fill::value(indsort0(s0)));
      } else{
        std::vector<int> addset = {indsort0(s0), indsort0(s1)};
        Nset(ni) = arma::conv_to<arma::uvec>::from(addset);
      }
    }
  }
  
  return Nset;
}

arma::field<arma::uvec> radgpbuild_testset(const arma::mat& wtrain,
                                            const arma::mat& wtest, 
                                            double rho,
                                            arma::uvec& layers, int M){
  arma::field<arma::uvec> Rset = neighbor_search_testset(wtrain, wtest, rho);
  int ntrain = wtrain.n_rows;
  arma::field<arma::uvec> dag = dagbuild_from_nn_testset(Rset, ntrain, layers, M, wtrain, wtest);
  return dag;
}


//[[Rcpp::export]]
Rcpp::List radial_neighbors_dag_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho, int M){
  arma::field<arma::uvec> Rset = neighbor_search_testset(wtrain, wtest, rho);
  int ntrain = wtrain.n_rows;
  arma::uvec layers;
  arma::field<arma::uvec> dag = dagbuild_from_nn_testset(Rset, ntrain, layers, M, wtrain, wtest);
  return Rcpp::List::create(
    Rcpp::Named("dag") = dag,
    Rcpp::Named("layers") = layers,
    Rcpp::Named("M") = M
  );
}

//[[Rcpp::export]]
Rcpp::List radgp_response_predict(const arma::mat& cout,
                                     const arma::vec& y, 
                                     const arma::mat& coords, double rho,
                                     const arma::mat& theta_mcmc, 
                                     int M,
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
    radgpbuild_testset(coords, cout, rho, layers, M);
  arma::uvec pred_order = arma::sort_index(layers);
  
  arma::mat yout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  
  Rcpp::Rcout << "AptDAG-GP predicting (response model)" << endl;
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
                                  theta, bessel_ws, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, true) );
      
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
    Rcpp::Named("layers") = layers,
    Rcpp::Named("pred_order") = pred_order
  );
}

//[[Rcpp::export]]
arma::mat vecchiagp_response_predict(const arma::mat& cout,
                                     const arma::vec& y, 
                                     const arma::mat& coords, 
                                     const arma::field<arma::uvec>& dag,
                                     const arma::mat& theta_mcmc, 
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
                                  theta, bessel_ws, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, true) );
      
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
                                     int M,
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
    radgpbuild_testset(coords, cout, rho, layers, M);
  arma::uvec pred_order = arma::sort_index(layers);
  
  arma::mat wout_mcmc = arma::zeros(ntest, mcmc);
  arma::mat random_stdnormal = arma::randn(mcmc, ntest);
  arma::vec wtemp = arma::zeros(ntrain+ntest);
  
  Rcpp::Rcout << "AptDAG-GP predicting (latent model)" << endl;
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int m=0; m<mcmc; m++){
    arma::vec theta = theta_mcmc.col(m);
    wtemp.subvec(0, ntrain-1) = w.col(m);
    
    for(int i=0; i<ntest; i++){
      
      int itarget = pred_order(ntrain+i);
      int idagtarget = itarget - ntrain;
      
      arma::uvec ix = oneuv * (itarget);
      arma::uvec px = predict_dag(idagtarget);
      
      arma::mat CC = Correlationf(cxall, ix, ix, 
                                  theta, bessel_ws, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, true) );
      
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
    Rcpp::Named("layers") = layers,
    Rcpp::Named("pred_order") = pred_order
  );
}

//[[Rcpp::export]]
arma::mat vecchiagp_latent_predict(const arma::mat& cout,
                                     const arma::mat& w, 
                                     const arma::mat& coords, 
                                     const arma::field<arma::uvec>& dag,
                                     const arma::mat& theta_mcmc, 
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
                                  theta, bessel_ws, true);
      arma::mat CPt = Correlationf(cxall, px, ix, theta, bessel_ws, false);
      arma::mat PPi = 
        arma::inv_sympd( Correlationf(cxall, px, px, theta, bessel_ws, true) );
      
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
