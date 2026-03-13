/*! \file
 \brief Methods of the struct \a matrix
 */

#include <cstdio>
#include <Eigen/Dense>

#include "matrix.hpp"
#include "parser.hpp"

using namespace std;

/*
 NOTE : The general convention of this code is to use "Column Major Matrices",
 stored in linear arrays, such that if A is a r x c matrix, then

 A[i][j] --->  A[i+r*j]

 Eigen also uses column-major storage by default, so Eigen::Map wraps
 the v.data() buffer directly with zero copy.
 */

/**
 inverts the current matrix (replaces its content)
 */
template<> void matrix<double>::inverse()
{
	QCM_ASSERT(r==c);
	if(r==0) return;
	if(r==1){v[0] = 1/v[0]; return;}
	Eigen::Map<Eigen::MatrixXd> M(v.data(), r, c);
	M = M.inverse().eval();
}


/**
 checks whether the Complex matrix A is real
 */
template<> bool matrix<Complex>::is_real(double accuracy)
{
	for(size_t i=0; i<v.size(); ++i) if(abs(v[i].imag()) > accuracy) return false;
	return true;
}


/**
 Finds eigenvalues and eigenvectors of a real symmetric matrix
 @param d [out] pre-allocated vector of eigenvalues
 @param U [out] pre-allocated orthogonal matrix whose columns are the eigenvectors of the matrix
 */
template<> void matrix<double>::eigensystem(vector<double> &d, matrix<double> &U) const
{
	QCM_ASSERT(r == c);
	QCM_ASSERT(U.r == r and U.c == c);
	QCM_ASSERT(d.size() == r);

	Eigen::Map<const Eigen::MatrixXd> A(v.data(), r, c);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
	if(solver.info() != Eigen::Success)
	{
		cout << "dimension : " << r << " x " << c << endl;
		cout << *this << endl;
		throw std::runtime_error("SelfAdjointEigenSolver failed (eigensystem<double>)");
	}
	Eigen::Map<Eigen::VectorXd>(d.data(), r) = solver.eigenvalues();
	Eigen::Map<Eigen::MatrixXd>(U.v.data(), r, c) = solver.eigenvectors();
}


/**
 computes the eigenvalues of a real symmetric matrix
 @param d vector of eigenvalues
 */
template<> void matrix<double>::eigenvalues(vector<double> &d)
{
	QCM_ASSERT(r == c);
	Eigen::Map<const Eigen::MatrixXd> A(v.data(), r, c);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A, Eigen::EigenvaluesOnly);
	QCM_ASSERT(solver.info() == Eigen::Success);
	Eigen::Map<Eigen::VectorXd>(d.data(), r) = solver.eigenvalues();
}


/**
 inverts the current matrix (replaces its content)
 */
template<> void matrix<Complex>::inverse()
{
	QCM_ASSERT(r==c);
	if(r==0) return;
	Eigen::Map<Eigen::MatrixXcd> M(v.data(), r, c);
	M = M.inverse().eval();
}


/**
 Returns the determinant of a complex matrix.
 Note: unlike the previous LAPACK-based version (which returned only the
 product of the U diagonal in the LU factorisation), this returns the
 correct determinant including the permutation sign.
 */
template<> Complex matrix<Complex>::determinant()
{
	if(r==0) return Complex(0);
	Eigen::Map<Eigen::MatrixXcd> M(v.data(), r, c);
	return M.determinant();
}


/**
 Finds eigenvalues and eigenvectors of a square hermitian matrix
 @param d [out] pre-allocated vector of eigenvalues
 @param U [out] pre-allocated unitary matrix whose columns are the eigenvectors of the matrix
 */
template<> void matrix<Complex>::eigensystem(vector<double> &d, matrix<Complex> &U) const
{
	QCM_ASSERT(r == c);

	Eigen::Map<const Eigen::MatrixXcd> A(v.data(), r, c);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(A);
	if(solver.info() != Eigen::Success)
	{
		cout << "dimension : " << r << " x " << c << endl;
		cout << *this << endl;
		throw std::runtime_error("SelfAdjointEigenSolver failed (eigensystem<complex>)");
	}
	Eigen::Map<Eigen::VectorXd>(d.data(), r) = solver.eigenvalues();
	Eigen::Map<Eigen::MatrixXcd>(U.v.data(), r, c) = solver.eigenvectors();
}


/**
 computes the eigenvalues of a complex hermitian matrix
 @param d vector of eigenvalues
 */
template<> void matrix<Complex>::eigenvalues(vector<double> &d)
{
	QCM_ASSERT(r == c);
	Eigen::Map<const Eigen::MatrixXcd> A(v.data(), r, c);
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(A, Eigen::EigenvaluesOnly);
	QCM_ASSERT(solver.info() == Eigen::Success);
	Eigen::Map<Eigen::VectorXd>(d.data(), r) = solver.eigenvalues();
}


/**
 returns the norm of the matrix
 */
template<> double matrix<double>::norm()
{
	Eigen::Map<const Eigen::MatrixXd> M(v.data(), r, c);
	return M.norm();
}


/**
 returns the norm of the matrix
 */
template<> double matrix<Complex>::norm()
{
	Eigen::Map<const Eigen::MatrixXcd> M(v.data(), r, c);
	return M.norm();
}


/**
 returns the square of the norm of the matrix
 */
template<> double matrix<double>::norm2()
{
	Eigen::Map<const Eigen::MatrixXd> M(v.data(), r, c);
	return M.squaredNorm();
}


/**
 returns the square of the norm of the matrix
 */
template<> double matrix<Complex>::norm2()
{
	Eigen::Map<const Eigen::MatrixXcd> M(v.data(), r, c);
	return M.squaredNorm();
}


/**
 returns true if the matrix is orthogonal
 */
template<> bool matrix<double>::is_orthogonal(double accuracy)
{
	if(r!=c) return false;
	Eigen::Map<const Eigen::MatrixXd> M(v.data(), r, c);
	Eigen::MatrixXd diff = M.transpose() * M - Eigen::MatrixXd::Identity(r, r);
	return diff.cwiseAbs().maxCoeff() <= accuracy;
}


/**
 puts in the current matrix the matrix L of the Cholesky decomposition of A = L * transpose(L)
 */
template<> bool matrix<double>::cholesky(matrix<double> &A)
{
	Eigen::Map<const Eigen::MatrixXd> Ma(A.v.data(), A.r, A.c);
	Eigen::LLT<Eigen::MatrixXd> llt(Ma);
	if(llt.info() != Eigen::Success) return false;
	set_size(A.r);
	Eigen::Map<Eigen::MatrixXd>(v.data(), r, c) = llt.matrixL();
	return true;
}


/**
 replaces the current matrix by its lower-triangular inverse
 */
template<> bool matrix<double>::triangular_inverse()
{
	Eigen::Map<Eigen::MatrixXd> M(v.data(), r, c);
	Eigen::MatrixXd Linv = M.triangularView<Eigen::Lower>().solve(
		Eigen::MatrixXd::Identity(r, r));
	M = Linv;
	return true;
}



/*
transforms a real matrix into a complex one
**/
matrix<Complex> to_complex_matrix(const matrix<double> &x) {
	matrix<Complex> xc(x.r, x.c);
	for (size_t i = 0; i < x.v.size(); i++) xc.v[i] = x.v[i];
	return xc;
}

/*
Builds a hermitian complex matrix from a real vector
**/
matrix<Complex> hermitian_matrix_from_real_vector(size_t d, const vector<double> &x){
	matrix<Complex> M(d);
	size_t k = 0;
	for(int i = 0; i<d; i++){
      M(i,i) = x[k++];
      for(int j = 0; j<i; j++){
	  	M(i,j) = Complex(x[k++], 0);
		M(i,j) += Complex(0, x[k++]);
		M(j,i) = conjugate(M(i,j));
	  }
    }
	return M;
}

/*
Populates a real vector from a hermitian complex matrix
**/
void hermitian_matrix_to_real_vector(const matrix<Complex> &M, double *x){
	size_t k = 0;
	for(int i = 0; i<M.r; i++){
      x[k++] = M(i,i).real();
      for(int j = 0; j<i; j++){
        x[k++] = M(i,j).real();
        x[k++] = M(i,j).imag();
      }
    }
}

/*
Builds a complex matrix from a real vector
**/
matrix<Complex> matrix_from_real_vector(size_t d, const vector<double> &x){
	matrix<Complex> M(d);
	size_t k = 0;
	for(int i = 0; i<d; i++){
      for(int j = 0; j<d; j++){
	  	M(i,j) = Complex(x[k++], 0);
		M(i,j) += Complex(0, x[k++]);
	  }
    }
	return M;
}

/*
Populates a real vector from a complex matrix
**/
void matrix_to_real_vector(const matrix<Complex> &M, double *x){
	size_t k = 0;
	for(int i = 0; i<M.r; i++){
      for(int j = 0; j<M.r; j++){
        x[k++] = M(i,j).real();
        x[k++] = M(i,j).imag();
	  }
    }
}
