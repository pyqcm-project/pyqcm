#ifndef vector_num_h
#define vector_num_h

#include <cstring>
#include <cstdlib>
#include <vector>
#include <random>

/*
 operations defined on vector<>'s of numeric type
*/

#include "types.hpp"
#include "global_parameter.hpp"
#include "parser.hpp"
#include "lapack-blas.h"

#define SMALL_VALUE 1.0e-12
#define MIDDLE_VALUE 1.0e-8

/**
 prototypes
 */
pair<double,double> jackknife(const vector<double>& x);
bool random(vector<Complex> &x, std::normal_distribution<double> &ran);
bool random(vector<double> &x, std::normal_distribution<double> &ran);
Complex operator*(const vector<Complex>&x, const vector<Complex>&y);
double norm(const vector<Complex> &x);
double norm(const vector<double> &x);
double norm2(const vector<Complex> &x);
double norm2(const vector<double> &x);
double operator*(const vector<double >&x, const vector<double >&y);
double sum_negative(vector<double>&x);
double max_abs(vector<double>&x);
double min_abs(vector<double>&x);
double sum(vector<double>&x);
void fix_phase(vector<Complex> &x);
void mult_add(Complex a, const vector<Complex> &x, vector<Complex> &y);
void mult_add(double a, const vector<double > &x, vector<double > &y);
void operator*=(vector<Complex> &x, const Complex &c);
void operator*=(vector<double> &x, const double &c);
inline vector<complex<double>> to_complex(const vector<complex<double>> &v)
{
	return v;
}
vector<complex<double>> to_complex(const vector<double> &v);

// multiplication of a vector by a number
template<typename T, typename S>
vector<T> operator*(const vector<T>& x, S a){
  vector<T> y(x);
  for(size_t i=0; i<y.size(); i++) y[i] *= a;
  return y;
}



/**
 Adds a vector to an existing one: y += x
 */
template <typename T, typename S>
void operator+=(vector<T> &y, const vector<S> &x){
	for(size_t i=0; i<x.size(); i++) y[i] += x[i];
}



/**
 Subtracts a constant from a vector: y -= a
 */
template <typename T, typename S>
void operator-=(vector<T> &y, const S &a){
	for(size_t i=0; i<y.size(); i++) y[i] -= a;
}


/**
 Adds a constant to a vector: y += x
 */
template <typename T, typename S>
void operator+=(vector<T> &y, const S &a){
	for(size_t i=0; i<y.size(); i++) y[i] += a;
}



/**
 Subtracts a vector to an existing one: y -= x
 */
template <typename T>
void operator-=(vector<T> &y, const vector<T> &x){
	for(size_t i=0; i<x.size(); i++) y[i] -= x[i];
}



/**
 adds two vectors : returns x + y
 */
template <typename T>
inline vector<T> operator + (const vector<T>& x, const vector<T>& y){
	vector<T> tmp(x);
	tmp += y;
	return tmp;
}


/**
 adds two vectors : returns x - y
 */
template <typename T>
inline vector<T> operator - (const vector<T>& x, const vector<T>& y){
	vector<T> tmp(x);
	tmp -= y;
	return tmp;
}


/**
linear combination
 */
template <typename T>
void linear_combination(vector<T> &x, const vector<T> &v1, T a1, const vector<T> &v2, T a2){
	for(size_t i=0; i<x.size(); i++)	x[i] = a1*v1[i] + a2*v2[i];
}



/**
 normalizes the vector
 returns true if successful, false if vector is too small
 */
template <typename T>
bool normalize(vector<T> &x, double accur=SMALL_VALUE){
	double z = norm(x);
	if(z < accur) return false;
	z = 1.0/z;
	x *= z;
	return true;
}


/**
 sets to zero
 */
template <typename T>
inline void to_zero(vector<T>& x){
	x.assign(x.size(),T(0));
}


/**
 truly clears the vector: forces memory to be cleared
 */
template <typename T>
void erase(vector<T> &x){
	x.clear();
	vector<T>().swap(x);
}


/**
 prints to flux
 */
template <typename T>
ostream& operator << (ostream& flux, const vector<T>& x){
  if(x.size() > (size_t)global_int("max_dim_print")) return flux;
	flux << "(" << x[0];
	for(size_t i=1; i<x.size(); ++i){
		flux << ", " <<x[i];
	}
	flux << ")";
	return flux;
}


/**
 reads from flux
 */
template <typename T>
istream& operator >> (istream& in, vector<T>& x){
	for(size_t i=0; i<x.size(); i++){
		in >> x[i];
	}
	return in;
}

/**
 prints into a string
 */
template <typename T>
string to_string(const vector<T> &x)
{
	ostringstream sout;
	sout << '(' << x[0];
	for(size_t i=1; i<x.size(); i++) sout << ',' << x[i];
	sout << ')';
	return sout.str();
}


/**
 Performs the modified Gram-Schmidt method on the set of vectors b, assuming that the first k are already orthonormal
 When a vector has zero norm, it drops it from the list.
 @param b	basis to orthonormalize
 @param k	index of the vector where to start the procedure
 @param accur	accuracy that sets a vector to a null vector
 
 the list returned has the correct number of nonzero vectors
 */
template<typename T>
void gram_schmidt(vector<vector<T> > &b, size_t k, double accur=SMALL_VALUE)
{
	for(size_t i=0; i<b.size(); i++){
		if(i>=k){
			if(normalize(b[i],accur) ==  false){ // push it to the end
				b[i].swap(b.back());
				b.pop_back();
				i--;
			}
		}
		size_t j0 = max(k,i+1);
		for(size_t j=j0; j<b.size(); j++) {
			T z = b[i]*b[j];
			mult_add(-z,b[i],b[j]);
		}
	}
}



/**
 finds the nm lowest elements of the vector and puts the associated indices in "index"
 index is not pre-allocated
 */
template<typename T>
void find_lowest(vector<T> &x, const size_t Nm, vector<size_t> &index)
{
	size_t nm = Nm;
	QCM_ASSERT(nm <= x.size());
	vector<double> minvals(nm);
	index.resize(nm);
	
	for(size_t i=0; i<nm; i++) minvals[i] = 1.0e12 + 1.0*i;
	
	for(size_t i=0; i<x.size(); i++){
		int j;
		for(j=(int)nm-1; j>=0; j--) if(realpart(x[i]) > minvals[j]) break;
		j++;
		if(j<nm){
			for(int k=(int)nm-1; k>j; k--){
				minvals[k]=minvals[k-1];
				index[k] = index[k-1];
			}
			minvals[j]=realpart(x[i]);
			index[j]=i;
		}
	}
}




/**
 interchanges two vectors, with a factor
y <-- x/beta   and   x <-- -beta*y
 */
template<typename T, typename S>
void swap(vector<T> &x, vector<T> &y, S beta)
{
	T tmp, z=1.0/beta;
	for(size_t i=0; i<x.size(); ++i){
		tmp = y[i];
		y[i] = x[i]*z;
		x[i] = -beta*tmp;
	}
}



inline void conjugate(vector<double> &x) {}
inline void conjugate(vector<Complex> &x){
	for(size_t i=0; i<x.size(); i++) x[i] = conj(x[i]);
}


#endif
