/*! \file
 \brief Methods of the struct \a matrix
 */

#include <cstdio>

#include "matrix.hpp"
#include "parser.hpp"

using namespace std;

/*
 NOTE : The general convention of this code is to use "Column Major Matrices",
 stored in linear arrays, such that if A is a r x c matrix, then
 
 A[i][j] --->  A[i+r*j]
 
 */

/**
 */
template<> void matrix<double>::inverse()
{
	assert(r==c);
	if(r==1){v[0] = 1/v[0]; return;}
	else if(r==0) return;
	
	integer INFO=0, M , N , LDA;
	
	vector<integer> IPIV(r);
	vector<doublereal> WORK(r);
	
	N = c;
	M = r;
	LDA = r;
	
	dgetrf_(&M, &N,(doublereal *)&v[0], &LDA,IPIV.data(),&INFO);
	assert((int)INFO==0);
	dgetri_(&N,(doublereal *)&v[0],&LDA,IPIV.data(),WORK.data(),&N,&INFO);
	assert((int)INFO==0);
}






/**
 checks whether the Complex matrix A is real
 */
template<> bool matrix<Complex >::is_real(double accuracy)
{
	for(size_t i=0; i<v.size(); ++i) if(abs(v[i].imag()) > accuracy) return false;
	return true;
}






/**
 Finds eigenvalues and eigenvectors of a  real symmetric matrix
 @param d [out] pre-allocated vector of eigenvalues
 @param U [out] pre-allocated orthogonal matrix whose columns are the eigenvectors of the matrix
 */
template<> void matrix<double>::eigensystem(vector<double> &d, matrix<double> &U)const
{
	assert(r == c);
	assert(U.r == r and U.c == c);
	assert(d.size() == r);
	
	char JOBZ, UPLO;
	integer INFO=0, N , LDA, LWORK;
	
	JOBZ = 'V'; // calculates eigenvectors too
	UPLO = 'L'; // lower triangle is stored
	N = (integer)r;
	erase(U.v);
	U.v = this->v;
	LDA = r;
	LWORK = 3*N; // N*(N+2);
	vector<doublereal> WORK(LWORK);
	
	dsyev_(&JOBZ, &UPLO, &N, (doublereal *)&U.v[0], &LDA, (doublereal *)&d[0], WORK.data(), &LWORK, &INFO);
	
	if (!((int)INFO==0))
	{
		cout << "dimension : " << r << " x " << c << endl;
		cout << *this << endl; // TEMPO
		throw std::runtime_error(std::string
			("bad INFO value from dsyev (eigensystem)")+std::to_string((int)INFO) );
	}
}






/**
 computes the eigenvalues of a real symmetric matrix
 @param d vector of eigenvalues
 */
template<> void matrix<double>::eigenvalues(vector<double> &d)
{
	assert(r == c);
	
	char JOBZ, UPLO;
	integer INFO, N , LDA, LWORK;
	
	vector<double> w(v);
	JOBZ = 'N'; // do not calculates eigenvectors
	UPLO = 'L'; // lower triangle is stored
	N = (integer)r;
	LDA = r;
	LWORK = N*(N+2);
	vector<doublereal> WORK(LWORK);
	
	dsyev_(&JOBZ, &UPLO, &N, (doublereal *)&w[0], &LDA, (doublereal *)&d[0], WORK.data(), &LWORK, &INFO);
	assert((int)INFO==0);
}






/**
 inverts the current matrix (replaces its content)
 */
template<> void matrix<Complex >::inverse()
{
	assert(r==c);
	if(r==0) return;
	integer INFO=0, M , N , LDA;
	
	vector<integer> IPIV(r);
	vector<doublecomplex> WORK(r);
	
	N = r;
	M = r;
	LDA = r;
	try{
		zgetrf_(&M, &N,(doublecomplex*)&v[0], &LDA, IPIV.data(),&INFO);
		if((int)INFO) qcm_throw("Error in matrix inversion:  zgetrf_() produces INFO = "+to_string<int>(INFO));
		zgetri_(&N,(doublecomplex*)&v[0],&LDA,IPIV.data(),WORK.data(),&N,&INFO);
		if((int)INFO) qcm_throw("Error in matrix inversion:  zgetri_() produces INFO = "+to_string<int>(INFO));
	} catch(const string& s) {qcm_catch(s);}
}






/**
 Absolute value (module) of the determinant of a complex matrix
 Destroys the matrix !
 */
template<> Complex  matrix<Complex >::determinant()
{
	Complex z;
	size_t i;
	if(r==0) return Complex(0);
	
	integer INFO=0, LDA;
	
	vector<integer> IPIV(r);
	LDA = r;
	zgetrf_((integer *)&r, (integer *)&c,(doublecomplex*)&v[0], &LDA, IPIV.data(), &INFO);
	assert((int)INFO==0);
	z = Complex(1.0);
	for(i=0; i<r; ++i) z *= v[i+i*r];
	return(z);
}






/**
 Finds eigenvalues and eigenvectors of a square hermitian matrix
 @param d [out] pre-allocated vector of eigenvalues
 @param U [out] pre-allocated unitary matrix whose columns are the eigenvectors of the matrix
 */
template<> void matrix<Complex >::eigensystem(vector<double> &d, matrix<Complex > &U)const
{
	assert(r == c);
	
	char JOBZ, UPLO;
	integer INFO=0, N , LDA, LWORK;
	
	JOBZ = 'V'; // calculates eigenvectors too
	UPLO = 'L'; // lower triangle is stored
	N = (integer)r;
	erase(U.v);
	U.v = this->v;
	LDA = r;
	LWORK = 2*N; // N*(N+2);
	vector<doublecomplex> WORK(LWORK);
	vector<doublereal> RWORK(3*N-2);
	zheev_(&JOBZ, &UPLO, &N, (doublecomplex *)&U.v[0], &LDA, (doublereal *)&d[0], WORK.data(), &LWORK, RWORK.data(), &INFO );
	if (!((int)INFO==0))
	{
		cout << "dimension : " << r << " x " << c << endl;
		cout << *this << endl; // TEMPO
		throw std::runtime_error(std::string
			("bad INFO value from zheev (eigensystem<complex>)")+std::to_string((int)INFO) );
	}
	assert((int)INFO==0);
}






/**
 computes the eigenvalues of a complex hermitian matrix
 @param d vector of eigenvalues
 */
template<> void matrix<Complex >::eigenvalues(vector<double> &d)
{
	assert(r == c);
	
	char JOBZ, UPLO;
	integer INFO=0, N , LDA, LWORK;
	
	vector<Complex> w(v);
	JOBZ = 'N'; // does not calculates eigenvectors
	UPLO = 'L'; // lower triangle is stored
	N = (integer)r;
	LDA = r;
	LWORK = 2*N-1;
	vector<doublecomplex> WORK(LWORK);
	vector<doublereal> RWORK(3*N-2);
	zheev_(&JOBZ, &UPLO, &N, (doublecomplex *)w.data(), &LDA, (doublereal *)d.data(), WORK.data(), &LWORK, RWORK.data(), &INFO );
	assert((int)INFO==0);
}





/**
 returns the norm of the matrix
 */
template<> double matrix<double>::norm()
{
	return cblas_dnrm2((int)v.size(),&v[0],1);
}






/**
 returns the norm of the matrix
 */
template<> double matrix<Complex >::norm()
{
	return cblas_dznrm2((int)v.size(),v.data(),1);;
}



/**
 returns the square of the norm of the matrix
 */
template<> double matrix<double>::norm2()
{
	double z = cblas_dnrm2((int)v.size(),v.data(),1);
	return z*z;
}



/**
 returns the square of the norm of the matrix
 */
template<> double matrix<Complex >::norm2()
{
	double z = cblas_dznrm2((int)v.size(),v.data(),1);
	return z*z;
}


/**
 returns true if the matrix is orthogonal
 */
template<> bool matrix<double>::is_orthogonal(double accuracy)
{
	if(r!=c) return false;
	for(size_t i=0; i<r; ++i){
		for(size_t j=0; j<=i; ++j){
			double z = 0.0;
			for(size_t k=0; k<r; ++k){
				z += v[k+i*r]*v[k+j*r];
			}
			if(i==j and abs(z-1.0) > accuracy) return false;
			if(i!=j and abs(z) > accuracy) return false;
		}
	}
	return true;
}


/**
 puts in the current matrix the matrix $L$ of the Cholesky decomposition of A = L.transpose(L)
 */
template<> bool matrix<double>::cholesky(matrix<double> &A){
	char UPLO;
	integer INFO=0, N , LDA;
	
	UPLO = 'L'; // lower triangle is stored
	N = (integer)r;
	LDA = A.r;
	set_size(A.r);
	for(size_t i=0; i<r; i++){
		for(size_t j=0; j<=i; j++) v[i+r*j] = A(i,j);
	}
	
	dpotrf_(&UPLO, &N, (doublereal *) &v[0], &LDA, &INFO);
	
	if((int)INFO==0) return true;
	else return false;
}



/**
 puts in the current matrix the matrix $L$ of the Cholesky decomposition of A = L.transpose(L)
 */
template<> bool matrix<double>::triangular_inverse(){
	char UPLO, DIAG;
	integer INFO=0, N , LDA;
	
	UPLO = 'L'; // lower triangle is stored
	DIAG = 'N'; // not a unit diagonal
	N = (integer)r;
	LDA = (integer)r;
	dtrtri_(&UPLO, &DIAG, &N, (doublereal *) &v[0], &LDA, &INFO);
	if((int)INFO==0) return true;
	else return false;
}




/*
transforms a real matrix into a complex one
**/
matrix<Complex> to_complex_matrix(const matrix<double> &x) {
	matrix<Complex> xc(x.r, x.c);
	for (size_t i = 0; i < x.v.size(); i++) xc.v[i] = x.v[i];
	return xc;
}
