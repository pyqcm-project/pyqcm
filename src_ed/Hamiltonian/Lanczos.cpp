#include "Lanczos.hpp"

//------------------------------------------------------------------------------
// external variables or declarations

using namespace std;
std::normal_distribution<double> normal_dis(0.0, 1.0);

//------------------------------------------------------------------------------
// variables local to this file



/**
 From the Lanczos coefficients alpha and beta, returns the lowest eigenvalue
 and the corresponding eigenvector in the Lanczos (reduced) basis.
 if \a evector_flag = false, calculates lowest eigenvalue only (no eigenvector);
 otherwise also calculates the associated eigenvector
 @param evector_flag flag for the calculation of the eigenvector
 @param alpha Diagonal
 @param beta Second diagonal
 @param energy eigenvalues (not pre-allocated)
 @param evector lowest eigenvector in the reduced basis (not pre-allocated)
 */
void EigensystemTridiagonal(bool evector_flag, vector<double> &alpha, vector<double> &beta, vector<double> &energy, vector<double> &evector)
{
  if(alpha.size()==1){
    energy.clear();
    erase(evector);
    energy.push_back(alpha[0]);
    if(evector_flag) evector.push_back(1.0);
    return;
  }
  size_t M = alpha.size();
  
  vector<double> d(alpha), e(beta), work(2*M), z(M*M);
  
  char jobz;
  
  if(evector_flag) jobz='V'; else jobz='N';
  integer info, ldz, nn;
  nn = M;
  ldz = M;
  
  // call to a LAPACK routine
  dstev_(&jobz, &nn, (doublereal*)d.data(), (doublereal*)e.data(), (doublereal*)z.data(), &ldz, (doublereal*)work.data(), &info);
  
  QCM_ASSERT((int)info==0);
  
  energy.assign(d.begin(),d.begin()+M);
  if(evector_flag) evector.assign(z.begin(), z.begin()+M);
}



