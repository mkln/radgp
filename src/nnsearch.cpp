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
                                         const arma::mat& w){
  int nr = Rset.n_elem;
  arma::rowvec center = arma::mean(w);
  // arma::rowvec center = arma::min(w);
  arma::mat diffs = w - arma::ones<arma::colvec>(nr) * center;
  arma::colvec c_dist2(nr);
  for(int i=0; i<nr; i++){
    c_dist2(i) = arma::dot(diffs.row(i),diffs.row(i));
  }
  arma::uvec sorted_id = arma::sort_index(c_dist2);
  arma::uvec sorted_order(nr);
  for(int i=0; i<nr; i++){
    sorted_order(sorted_id(i)) = i;
  }
  arma::field<arma::uvec> Nset(nr);
  for(int i=0; i<nr; i++){
    Nset(i) = Rset(i)(arma::find(sorted_order(Rset(i))<sorted_order(i)));
  }
  for(int i=1; i<nr; i++){
    if(Nset(sorted_id(i)).n_elem == 0){
      arma::mat diffs_b = w.rows(sorted_id.subvec(0,i-1)) - arma::ones<arma::colvec>(i) * w.row(sorted_id(i));
      arma::colvec c_dist2_b(i);
      for(int j=0; j<i; j++){
        c_dist2_b(j) = arma::dot(diffs_b.row(j),diffs_b.row(j)); 
      }
      Nset(i) = c_dist2_b.index_min();
    }
  }
  return Nset;
}

//[[Rcpp::export]]
arma::field<arma::uvec> radialndag(const arma::mat& w, double rho){
  int nr = w.n_rows;
  arma::field<arma::uvec> Rset = neighbor_search(w, rho);
  return dagbuild_from_nn(Rset, w);
}

//[[Rcpp::export]]
Rcpp::List radial_neighbors_dag(const arma::mat& w, double rho){
  int nr = w.n_rows;;
  arma::field<arma::uvec> dag = radialndag(w, rho);
  arma::rowvec center = arma::mean(w);
  arma::mat diffs = w - arma::ones<arma::colvec>(nr) * center;
  arma::colvec c_dist2(nr);
  for(int i=0; i<nr; i++){
    c_dist2(i) = arma::dot(diffs.row(i),diffs.row(i));
  }
  arma::uvec sorted_id = arma::sort_index(c_dist2);
  return Rcpp::List::create(
    Rcpp::Named("dag") = dag,
    Rcpp::Named("sorted_id") = sorted_id
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