
#include <RcppArmadillo.h>
using namespace std;

arma::umat make_candidates_testset(const arma::mat& w, 
                                   const arma::uvec& indsort, arma::uvec& testindsort,
                                   unsigned int col,
                                   double rho);
arma::field<arma::uvec> neighbor_search_testset(const arma::mat& wtrain, 
                                                const arma::mat& wtest, double rho);
arma::field<arma::uvec> dagbuild_from_nn_testset(const arma::field<arma::uvec>& Rset, 
                                                 int ntrain,
                                                 int Mmin);
arma::field<arma::uvec> altdagbuild_testset(const arma::mat& wtrain,
                                            const arma::mat& wtest, 
                                            double rho,
                                            arma::uvec& layers, int M);