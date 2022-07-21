// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// altdaggp
Rcpp::List altdaggp(const arma::mat& coords, const arma::vec& theta, double rho);
RcppExport SEXP _altdag_altdaggp(SEXP coordsSEXP, SEXP thetaSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(altdaggp(coords, theta, rho));
    return rcpp_result_gen;
END_RCPP
}
// Correlationc
arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, const arma::vec& theta, bool ps, bool same);
RcppExport SEXP _altdag_Correlationc(SEXP coordsxSEXP, SEXP coordsySEXP, SEXP thetaSEXP, SEXP psSEXP, SEXP sameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsx(coordsxSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsy(coordsySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type ps(psSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    rcpp_result_gen = Rcpp::wrap(Correlationc(coordsx, coordsy, theta, ps, same));
    return rcpp_result_gen;
END_RCPP
}
// gpkernel
arma::mat gpkernel(const arma::mat& coordsx, const arma::vec& theta);
RcppExport SEXP _altdag_gpkernel(SEXP coordsxSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsx(coordsxSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(gpkernel(coordsx, theta));
    return rcpp_result_gen;
END_RCPP
}
// make_candidates
arma::umat make_candidates(const arma::mat& w, const arma::uvec& indsort, unsigned int col, double rho);
RcppExport SEXP _altdag_make_candidates(SEXP wSEXP, SEXP indsortSEXP, SEXP colSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::uvec& >::type indsort(indsortSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type col(colSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(make_candidates(w, indsort, col, rho));
    return rcpp_result_gen;
END_RCPP
}
// neighbor_search
arma::field<arma::uvec> neighbor_search(const arma::mat& w, double rho);
RcppExport SEXP _altdag_neighbor_search(SEXP wSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(neighbor_search(w, rho));
    return rcpp_result_gen;
END_RCPP
}
// dagbuild_from_nn
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset);
RcppExport SEXP _altdag_dagbuild_from_nn(SEXP RsetSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type Rset(RsetSEXP);
    rcpp_result_gen = Rcpp::wrap(dagbuild_from_nn(Rset));
    return rcpp_result_gen;
END_RCPP
}
// altdagbuild
arma::field<arma::uvec> altdagbuild(const arma::mat& w, double rho);
RcppExport SEXP _altdag_altdagbuild(SEXP wSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(altdagbuild(w, rho));
    return rcpp_result_gen;
END_RCPP
}
// sparse_struct
arma::umat sparse_struct(const arma::field<arma::uvec>& dag, int nr);
RcppExport SEXP _altdag_sparse_struct(SEXP dagSEXP, SEXP nrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< int >::type nr(nrSEXP);
    rcpp_result_gen = Rcpp::wrap(sparse_struct(dag, nr));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_altdag_altdaggp", (DL_FUNC) &_altdag_altdaggp, 3},
    {"_altdag_Correlationc", (DL_FUNC) &_altdag_Correlationc, 5},
    {"_altdag_gpkernel", (DL_FUNC) &_altdag_gpkernel, 2},
    {"_altdag_make_candidates", (DL_FUNC) &_altdag_make_candidates, 4},
    {"_altdag_neighbor_search", (DL_FUNC) &_altdag_neighbor_search, 2},
    {"_altdag_dagbuild_from_nn", (DL_FUNC) &_altdag_dagbuild_from_nn, 1},
    {"_altdag_altdagbuild", (DL_FUNC) &_altdag_altdagbuild, 2},
    {"_altdag_sparse_struct", (DL_FUNC) &_altdag_sparse_struct, 2},
    {NULL, NULL, 0}
};

RcppExport void R_init_altdag(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
