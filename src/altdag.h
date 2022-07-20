#ifndef ALTDAG 
#define ALTDAG

// uncomment to disable openmp on compilation
//#undef _OPENMP

#include "RcppArmadillo.h"
#include <RcppEigen.h>
//#include <Eigen/CholmodSupport>

using namespace std;

class AltDAG {
public:
  arma::vec y;
};




#endif