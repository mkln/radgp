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
                                                 int Mmin){
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
  return Nset;
}

arma::field<arma::uvec> altdagbuild_testset(const arma::mat& wtrain,
                                            const arma::mat& wtest, 
                                            double rho,
                                            arma::uvec& layers, int M){
  arma::field<arma::uvec> Rset = neighbor_search_testset(wtrain, wtest, rho);
  int ntrain = wtrain.n_rows;
  arma::field<arma::uvec> dag = dagbuild_from_nn_testset(Rset, ntrain, layers, M);
  return dag;
}


//[[Rcpp::export]]
Rcpp::List Raltdagbuild_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho, int M){
  arma::field<arma::uvec> Rset = neighbor_search_testset(wtrain, wtest, rho);
  int ntrain = wtrain.n_rows;
  arma::uvec layers;
  arma::field<arma::uvec> dag = dagbuild_from_nn_testset(Rset, ntrain, layers, M);
  return Rcpp::List::create(
    Rcpp::Named("dag") = dag,
    Rcpp::Named("layers") = layers,
    Rcpp::Named("M") = M
  );
}
