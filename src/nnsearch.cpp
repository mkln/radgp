#include "nnsearch.h"

//[[Rcpp::export]]
arma::umat make_candidates(const arma::mat& w, 
                           const arma::uvec& indsort,
                           unsigned int col,
                           double rho){
  arma::uvec colsel = col + arma::zeros<arma::uvec>(1);
  //arma::vec wsort = w.submat(indsort, colsel);
  
  int nr = indsort.n_elem;
  arma::umat candidates = arma::zeros<arma::umat>(nr, 2);
  int left = 0;
  int right = 0;
  
  for(unsigned int loc = 0; loc<nr; loc++){
    while((w(indsort(loc), col) - w(indsort(left), col)) > rho){
      left ++;
    }
    if(right < nr - 1){
      while(w(indsort(right+1)) - w(indsort(loc)) <= rho){
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
arma::field<arma::uvec> neighbor_search(const arma::mat& w, double rho){
  unsigned int nr = w.n_rows;
  double rho2 = rho*rho;
  
  arma::umat candidates = arma::zeros<arma::umat>(nr, 2);
  arma::uvec indsort = arma::sort_index(w.col(0));
  candidates = make_candidates(w, indsort, 0, rho);
  
  arma::field<arma::uvec> Nset(nr);
  for(unsigned int i=0; i<nr; i++){
    int left = candidates(i, 0);
    int right = candidates(i, 1);
    arma::uvec try_ids;
    int rightx = right==(nr-1)? nr-1 : right;
  
    if((i == 0)|(i==left)){
      try_ids = indsort.rows(i+1, rightx);
    } else if((i == nr - 1)|(i==rightx)){
      try_ids = indsort.rows(left, i-1);
    } else {
      try_ids = arma::join_vert(indsort.rows(left, i-1),
                                indsort.rows(i+1, rightx));
    }
    
    int ni = try_ids.n_elem;
    arma::vec try_dist2 = arma::zeros(ni);
    for(unsigned int j=0; j<ni; j++){
      arma::rowvec rdiff = w.row(try_ids(j)) - w.row(indsort(i));
      try_dist2(j) = arma::accu(rdiff % rdiff);
    }
    Nset(indsort(i)) = try_ids.rows(arma::find(try_dist2<=rho2));
  
    
  }
  return Nset;
}

//[[Rcpp::export]]
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset, 
                                         arma::uvec& layers, int& M){
  int nr = Rset.n_elem;
  
  M = 1;
  //arma::uvec layers = arma::zeros<arma::uvec>(nr);
  arma::field<arma::uvec> R_layers(nr);
  
  std::deque<int> queue;
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  for(int i=0; i<nr; i++){
    if(!(layers(i) == 0)){
      continue;
    } else {
      int l = i;
      for(int k=1; k<M+2; k++){
        arma::uvec found_elem = arma::find(R_layers(l) == k);
        if(found_elem.n_elem == 0){ 
          layers(l) = k;
          break;
        }
      }
      //M = layers(l) > M ? layers(l) : M;
      if(layers(l) > M){
        M += 1;
      }
        
      for(int jx=0; jx < Rset(l).n_elem; jx++){
        int j = Rset(l)(jx);
        R_layers(j) = arma::join_vert(R_layers(j), oneuv*layers(l));
        if(layers(j) == 0){
          queue.push_back(j); 
        }
      } 
      
      while(queue.size()>0){
        l = queue.front();
        queue.pop_front();
        if(layers(l) > 0){
          continue;
        }
        for(int k=1; k<M+2; k++){
          arma::uvec found_elem = arma::find(R_layers(l) == k);
          if(found_elem.n_elem == 0){ //**
            layers(l) = k;
            break;
          }
        }
        //M = layers(l) > M ? layers(l) : M;
        if(layers(l) > M){
          M += 1;
        }
        
        for(int jx=0; jx<Rset(l).n_elem; jx++){
          int j = Rset(l)(jx);
          R_layers(j) = arma::join_vert(R_layers(j), oneuv*layers(l));
          if(layers(j) == 0){
            queue.push_back(j); 
          }
        } 
      }
    }
  }
  arma::field<arma::uvec> Nset(nr);
  for(int i=0; i<nr; i++){
    
    arma::uvec layers_ne(Rset(i).n_elem);
    for(int jx=0; jx<Rset(i).n_elem; jx++){
      int x = Rset(i)(jx);
      layers_ne(jx) = layers(x);
    }
    
    Nset(i) = Rset(i)(arma::find(layers_ne<layers(i)));
  }
  return Nset;
}

arma::field<arma::uvec> altdagbuild(const arma::mat& w, double rho, arma::uvec& layers, int& M){
  int nr = w.n_rows;
  
  arma::field<arma::uvec> Rset = neighbor_search(w, rho);
  arma::field<arma::uvec> dag = dagbuild_from_nn(Rset, layers, M);
  return dag;
}

//[[Rcpp::export]]
Rcpp::List Raltdagbuild(const arma::mat& w, double rho){
  int nr = w.n_rows;
  arma::uvec layers = arma::zeros<arma::uvec>(nr);
  
  Rcpp::Rcout << "neighbor search "<< endl;
  arma::field<arma::uvec> Rset = neighbor_search(w, rho);
  Rcpp::Rcout << "dag build"<< endl;
  int M=0;
  arma::field<arma::uvec> dag = dagbuild_from_nn(Rset, layers, M);
  return Rcpp::List::create(
    Rcpp::Named("dag") = dag,
    Rcpp::Named("layers") = layers,
    Rcpp::Named("M") = M
  );
}

//[[Rcpp::export]]
arma::umat sparse_struct(const arma::field<arma::uvec>& dag, int nr){
  arma::umat result = arma::zeros<arma::umat>(nr, nr);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  for(int i=0; i<nr; i++){
    result.submat(i*oneuv, dag(i)).fill(1);
  }
  return result;
}