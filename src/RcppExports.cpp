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

// hmat_from_dag
Eigen::SparseMatrix<double> hmat_from_dag(const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::vec& theta);
RcppExport SEXP _aptdag_hmat_from_dag(SEXP coordsSEXP, SEXP dagSEXP, SEXP thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(hmat_from_dag(coords, dag, theta));
    return rcpp_result_gen;
END_RCPP
}
// pred_from_dag
arma::vec pred_from_dag(const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::vec& theta, const arma::vec& urng);
RcppExport SEXP _aptdag_pred_from_dag(SEXP coordsSEXP, SEXP dagSEXP, SEXP thetaSEXP, SEXP urngSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type urng(urngSEXP);
    rcpp_result_gen = Rcpp::wrap(pred_from_dag(coords, dag, theta, urng));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_latent
Rcpp::List aptdaggp_latent(const arma::vec& y, const arma::mat& coords, double rho, int mcmc, int num_threads, const arma::vec& theta_init, double tausq_init, double metrop_sd, const arma::mat& theta_unif_bounds, bool sample_tausq, int num_prints);
RcppExport SEXP _aptdag_aptdaggp_latent(SEXP ySEXP, SEXP coordsSEXP, SEXP rhoSEXP, SEXP mcmcSEXP, SEXP num_threadsSEXP, SEXP theta_initSEXP, SEXP tausq_initSEXP, SEXP metrop_sdSEXP, SEXP theta_unif_boundsSEXP, SEXP sample_tausqSEXP, SEXP num_printsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< double >::type tausq_init(tausq_initSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_sd(metrop_sdSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_unif_bounds(theta_unif_boundsSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_tausq(sample_tausqSEXP);
    Rcpp::traits::input_parameter< int >::type num_prints(num_printsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_latent(y, coords, rho, mcmc, num_threads, theta_init, tausq_init, metrop_sd, theta_unif_bounds, sample_tausq, num_prints));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_custom_latent
Rcpp::List aptdaggp_custom_latent(const arma::vec& y, const arma::mat& coords, const arma::field<arma::uvec>& dag, int mcmc, int num_threads, const arma::vec& theta_init, double tausq_init, double metrop_sd, const arma::mat& theta_unif_bounds, bool sample_tausq, int num_prints);
RcppExport SEXP _aptdag_aptdaggp_custom_latent(SEXP ySEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP mcmcSEXP, SEXP num_threadsSEXP, SEXP theta_initSEXP, SEXP tausq_initSEXP, SEXP metrop_sdSEXP, SEXP theta_unif_boundsSEXP, SEXP sample_tausqSEXP, SEXP num_printsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< double >::type tausq_init(tausq_initSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_sd(metrop_sdSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_unif_bounds(theta_unif_boundsSEXP);
    Rcpp::traits::input_parameter< bool >::type sample_tausq(sample_tausqSEXP);
    Rcpp::traits::input_parameter< int >::type num_prints(num_printsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_custom_latent(y, coords, dag, mcmc, num_threads, theta_init, tausq_init, metrop_sd, theta_unif_bounds, sample_tausq, num_prints));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_response
Rcpp::List aptdaggp_response(const arma::vec& y, const arma::mat& coords, double rho, int mcmc, int num_threads, const arma::vec& theta_init, double metrop_sd, const arma::mat& theta_unif_bounds, int num_prints);
RcppExport SEXP _aptdag_aptdaggp_response(SEXP ySEXP, SEXP coordsSEXP, SEXP rhoSEXP, SEXP mcmcSEXP, SEXP num_threadsSEXP, SEXP theta_initSEXP, SEXP metrop_sdSEXP, SEXP theta_unif_boundsSEXP, SEXP num_printsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_sd(metrop_sdSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_unif_bounds(theta_unif_boundsSEXP);
    Rcpp::traits::input_parameter< int >::type num_prints(num_printsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_response(y, coords, rho, mcmc, num_threads, theta_init, metrop_sd, theta_unif_bounds, num_prints));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_custom
Rcpp::List aptdaggp_custom(const arma::vec& y, const arma::mat& coords, const arma::field<arma::uvec>& dag, int mcmc, int num_threads, const arma::vec& theta_init, double metrop_sd, const arma::mat& theta_unif_bounds, int num_prints);
RcppExport SEXP _aptdag_aptdaggp_custom(SEXP ySEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP mcmcSEXP, SEXP num_threadsSEXP, SEXP theta_initSEXP, SEXP metrop_sdSEXP, SEXP theta_unif_boundsSEXP, SEXP num_printsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< int >::type mcmc(mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta_init(theta_initSEXP);
    Rcpp::traits::input_parameter< double >::type metrop_sd(metrop_sdSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_unif_bounds(theta_unif_boundsSEXP);
    Rcpp::traits::input_parameter< int >::type num_prints(num_printsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_custom(y, coords, dag, mcmc, num_threads, theta_init, metrop_sd, theta_unif_bounds, num_prints));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp
Rcpp::List aptdaggp(const arma::mat& coords, const arma::vec& theta, double rho);
RcppExport SEXP _aptdag_aptdaggp(SEXP coordsSEXP, SEXP thetaSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp(coords, theta, rho));
    return rcpp_result_gen;
END_RCPP
}
// vecchiagp
Rcpp::List vecchiagp(const arma::mat& coords, const arma::vec& theta, const arma::field<arma::uvec>& dag);
RcppExport SEXP _aptdag_vecchiagp(SEXP coordsSEXP, SEXP thetaSEXP, SEXP dagSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchiagp(coords, theta, dag));
    return rcpp_result_gen;
END_RCPP
}
// daggp_negdens
double daggp_negdens(const arma::vec& y, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::vec& theta, int num_threads);
RcppExport SEXP _aptdag_daggp_negdens(SEXP ySEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP thetaSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(daggp_negdens(y, coords, dag, theta, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// Correlationc
arma::mat Correlationc(const arma::mat& coordsx, const arma::mat& coordsy, const arma::vec& theta, bool same);
RcppExport SEXP _aptdag_Correlationc(SEXP coordsxSEXP, SEXP coordsySEXP, SEXP thetaSEXP, SEXP sameSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsx(coordsxSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coordsy(coordsySEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type theta(thetaSEXP);
    Rcpp::traits::input_parameter< bool >::type same(sameSEXP);
    rcpp_result_gen = Rcpp::wrap(Correlationc(coordsx, coordsy, theta, same));
    return rcpp_result_gen;
END_RCPP
}
// gpkernel
arma::mat gpkernel(const arma::mat& coordsx, const arma::vec& theta);
RcppExport SEXP _aptdag_gpkernel(SEXP coordsxSEXP, SEXP thetaSEXP) {
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
RcppExport SEXP _aptdag_make_candidates(SEXP wSEXP, SEXP indsortSEXP, SEXP colSEXP, SEXP rhoSEXP) {
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
RcppExport SEXP _aptdag_neighbor_search(SEXP wSEXP, SEXP rhoSEXP) {
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
arma::field<arma::uvec> dagbuild_from_nn(const arma::field<arma::uvec>& Rset, arma::uvec& layers, int& M, const arma::mat& w, double rho);
RcppExport SEXP _aptdag_dagbuild_from_nn(SEXP RsetSEXP, SEXP layersSEXP, SEXP MSEXP, SEXP wSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type Rset(RsetSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type layers(layersSEXP);
    Rcpp::traits::input_parameter< int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(dagbuild_from_nn(Rset, layers, M, w, rho));
    return rcpp_result_gen;
END_RCPP
}
// Raptdagbuild
Rcpp::List Raptdagbuild(const arma::mat& w, double rho);
RcppExport SEXP _aptdag_Raptdagbuild(SEXP wSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(Raptdagbuild(w, rho));
    return rcpp_result_gen;
END_RCPP
}
// sparse_struct
arma::umat sparse_struct(const arma::field<arma::uvec>& dag, int nr);
RcppExport SEXP _aptdag_sparse_struct(SEXP dagSEXP, SEXP nrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< int >::type nr(nrSEXP);
    rcpp_result_gen = Rcpp::wrap(sparse_struct(dag, nr));
    return rcpp_result_gen;
END_RCPP
}
// neighbor_search_testset
arma::field<arma::uvec> neighbor_search_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho);
RcppExport SEXP _aptdag_neighbor_search_testset(SEXP wtrainSEXP, SEXP wtestSEXP, SEXP rhoSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type wtrain(wtrainSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type wtest(wtestSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    rcpp_result_gen = Rcpp::wrap(neighbor_search_testset(wtrain, wtest, rho));
    return rcpp_result_gen;
END_RCPP
}
// dagbuild_from_nn_testset
arma::field<arma::uvec> dagbuild_from_nn_testset(const arma::field<arma::uvec>& Rset, int ntrain, arma::uvec& layers, int Mmin);
RcppExport SEXP _aptdag_dagbuild_from_nn_testset(SEXP RsetSEXP, SEXP ntrainSEXP, SEXP layersSEXP, SEXP MminSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type Rset(RsetSEXP);
    Rcpp::traits::input_parameter< int >::type ntrain(ntrainSEXP);
    Rcpp::traits::input_parameter< arma::uvec& >::type layers(layersSEXP);
    Rcpp::traits::input_parameter< int >::type Mmin(MminSEXP);
    rcpp_result_gen = Rcpp::wrap(dagbuild_from_nn_testset(Rset, ntrain, layers, Mmin));
    return rcpp_result_gen;
END_RCPP
}
// Raptdagbuild_testset
Rcpp::List Raptdagbuild_testset(const arma::mat& wtrain, const arma::mat& wtest, double rho, int M);
RcppExport SEXP _aptdag_Raptdagbuild_testset(SEXP wtrainSEXP, SEXP wtestSEXP, SEXP rhoSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type wtrain(wtrainSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type wtest(wtestSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(Raptdagbuild_testset(wtrain, wtest, rho, M));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_response_predict
Rcpp::List aptdaggp_response_predict(const arma::mat& cout, const arma::vec& y, const arma::mat& coords, double rho, const arma::mat& theta_mcmc, int M, int num_threads);
RcppExport SEXP _aptdag_aptdaggp_response_predict(SEXP coutSEXP, SEXP ySEXP, SEXP coordsSEXP, SEXP rhoSEXP, SEXP theta_mcmcSEXP, SEXP MSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type cout(coutSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mcmc(theta_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_response_predict(cout, y, coords, rho, theta_mcmc, M, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// vecchiagp_response_predict
arma::mat vecchiagp_response_predict(const arma::mat& cout, const arma::vec& y, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::mat& theta_mcmc, int num_threads);
RcppExport SEXP _aptdag_vecchiagp_response_predict(SEXP coutSEXP, SEXP ySEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP theta_mcmcSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type cout(coutSEXP);
    Rcpp::traits::input_parameter< const arma::vec& >::type y(ySEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mcmc(theta_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchiagp_response_predict(cout, y, coords, dag, theta_mcmc, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// aptdaggp_latent_predict
Rcpp::List aptdaggp_latent_predict(const arma::mat& cout, const arma::mat& w, const arma::mat& coords, double rho, const arma::mat& theta_mcmc, int M, int num_threads);
RcppExport SEXP _aptdag_aptdaggp_latent_predict(SEXP coutSEXP, SEXP wSEXP, SEXP coordsSEXP, SEXP rhoSEXP, SEXP theta_mcmcSEXP, SEXP MSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type cout(coutSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< double >::type rho(rhoSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mcmc(theta_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type M(MSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(aptdaggp_latent_predict(cout, w, coords, rho, theta_mcmc, M, num_threads));
    return rcpp_result_gen;
END_RCPP
}
// vecchiagp_latent_predict
arma::mat vecchiagp_latent_predict(const arma::mat& cout, const arma::mat& w, const arma::mat& coords, const arma::field<arma::uvec>& dag, const arma::mat& theta_mcmc, int num_threads);
RcppExport SEXP _aptdag_vecchiagp_latent_predict(SEXP coutSEXP, SEXP wSEXP, SEXP coordsSEXP, SEXP dagSEXP, SEXP theta_mcmcSEXP, SEXP num_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::mat& >::type cout(coutSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type w(wSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type coords(coordsSEXP);
    Rcpp::traits::input_parameter< const arma::field<arma::uvec>& >::type dag(dagSEXP);
    Rcpp::traits::input_parameter< const arma::mat& >::type theta_mcmc(theta_mcmcSEXP);
    Rcpp::traits::input_parameter< int >::type num_threads(num_threadsSEXP);
    rcpp_result_gen = Rcpp::wrap(vecchiagp_latent_predict(cout, w, coords, dag, theta_mcmc, num_threads));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_aptdag_hmat_from_dag", (DL_FUNC) &_aptdag_hmat_from_dag, 3},
    {"_aptdag_pred_from_dag", (DL_FUNC) &_aptdag_pred_from_dag, 4},
    {"_aptdag_aptdaggp_latent", (DL_FUNC) &_aptdag_aptdaggp_latent, 11},
    {"_aptdag_aptdaggp_custom_latent", (DL_FUNC) &_aptdag_aptdaggp_custom_latent, 11},
    {"_aptdag_aptdaggp_response", (DL_FUNC) &_aptdag_aptdaggp_response, 9},
    {"_aptdag_aptdaggp_custom", (DL_FUNC) &_aptdag_aptdaggp_custom, 9},
    {"_aptdag_aptdaggp", (DL_FUNC) &_aptdag_aptdaggp, 3},
    {"_aptdag_vecchiagp", (DL_FUNC) &_aptdag_vecchiagp, 3},
    {"_aptdag_daggp_negdens", (DL_FUNC) &_aptdag_daggp_negdens, 5},
    {"_aptdag_Correlationc", (DL_FUNC) &_aptdag_Correlationc, 4},
    {"_aptdag_gpkernel", (DL_FUNC) &_aptdag_gpkernel, 2},
    {"_aptdag_make_candidates", (DL_FUNC) &_aptdag_make_candidates, 4},
    {"_aptdag_neighbor_search", (DL_FUNC) &_aptdag_neighbor_search, 2},
    {"_aptdag_dagbuild_from_nn", (DL_FUNC) &_aptdag_dagbuild_from_nn, 5},
    {"_aptdag_Raptdagbuild", (DL_FUNC) &_aptdag_Raptdagbuild, 2},
    {"_aptdag_sparse_struct", (DL_FUNC) &_aptdag_sparse_struct, 2},
    {"_aptdag_neighbor_search_testset", (DL_FUNC) &_aptdag_neighbor_search_testset, 3},
    {"_aptdag_dagbuild_from_nn_testset", (DL_FUNC) &_aptdag_dagbuild_from_nn_testset, 4},
    {"_aptdag_Raptdagbuild_testset", (DL_FUNC) &_aptdag_Raptdagbuild_testset, 4},
    {"_aptdag_aptdaggp_response_predict", (DL_FUNC) &_aptdag_aptdaggp_response_predict, 7},
    {"_aptdag_vecchiagp_response_predict", (DL_FUNC) &_aptdag_vecchiagp_response_predict, 6},
    {"_aptdag_aptdaggp_latent_predict", (DL_FUNC) &_aptdag_aptdaggp_latent_predict, 7},
    {"_aptdag_vecchiagp_latent_predict", (DL_FUNC) &_aptdag_vecchiagp_latent_predict, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_aptdag(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
