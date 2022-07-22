
#include <RcppArmadillo.h>
using namespace std;

//[[Rcpp::export]]
arma::umat make_candidates(const arma::mat& w, const arma::uvec& indsort, 
                           unsigned int col,
                           double rho){
  arma::uvec colsel = col + arma::zeros<arma::uvec>(1);
  arma::vec wsort = w.submat(indsort, colsel);
  
  int nr = wsort.n_elem;
  arma::umat candidates = arma::zeros<arma::umat>(nr, 2);
  int left = 0;
  int right = 0;
  
  //Rcpp::Rcout << "hey " << nr << endl;
  
  for(unsigned int loc = 0; loc<nr; loc++){
    while(wsort(loc) - wsort(left) > rho){
      left ++;
      //Rcpp::Rcout << "left: " << left << endl;
    }
    //Rcpp::Rcout << "right loop " << endl;
    if(right < nr - 1){
      while(wsort(right+1) - wsort(loc) <= rho){
        right ++;
        //Rcpp::Rcout << "right: " << right << " " << nr-1 << endl;
        if(right == nr - 1){
          break;
        }
      }
    }
    //Rcpp::Rcout << "candidates " << loc << endl;
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
    
    if(i == 0){
      try_ids = indsort.rows(i+1, rightx);
    } else if(i == nr - 1){
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
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset, int& M){
  int nr = Rset.n_elem;
  
  M = 1;
  arma::uvec layers = arma::zeros<arma::uvec>(nr);
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
      M = layers(l) > M ? layers(l) : M;
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
        M = layers(l) > M ? layers(l) : M;
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

//[[Rcpp::export]]
arma::field<arma::uvec> altdagbuild(const arma::mat& w, double rho, int& M){
  arma::field<arma::uvec> Rset = neighbor_search(w, rho);
  return dagbuild_from_nn(Rset, M);
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