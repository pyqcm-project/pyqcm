#ifndef vector3D_h
#define vector3D_h

#include <ostream>
#include <istream>
#include <iomanip>
#include <vector>
#include "matrix.hpp"
#include "parser.hpp"

using namespace std;
void qcm_throw(const string& s);

//! Used to represent the position of a lattice or cluster site
template<typename T>
struct vector3D
{
	T x; //!< x coordinate
	T y; //!< y coordinate
	T z; //!< z coordinate
	static int dimension;
	
	//! constructor
	vector3D() : x(0), y(0), z(0) {}
	
	
	
	//! constructor
	template<typename S>
	vector3D(S _x, S _y, S _z) : x(_x), y(_y), z(_z) {}
	
	
	
	//! constructor from a string
	vector3D(const string &str){
	char lpar='(';
	char vir=',';
	char rpar=')';
		istringstream sin(str);
		sin >> lpar >> x >> vir >> y >> vir >> z >> rpar;
		if(lpar != '(' or rpar != ')' or vir != ',')
		qcm_throw("3D vector has wrong format! Check parentheses, etc. Should have 3 components.");
	}
	
	
	
	
	//! copy constructor
	template<typename S>
	vector3D(vector3D<S> tmp): x(tmp.x), y(tmp.y), z(tmp.z) {}
	
	
	
	//! sign operator overloading
	vector3D operator-()
	{return(vector3D(-x, -y, -z));}
	
	
	
	//! assignation operator overloading
	template<typename S>
	vector3D& operator=(const vector3D<S>& q)
	{x = (T)q.x; y = (T)q.y; z = (T)q.z; return(*this);}
	
	
	
	//! addition operator overloading
	template<typename S>
	vector3D operator+(const vector3D<S>& q)
	{return vector3D(x+q.x,y+q.y,z+q.z);}
	
	
	
	//! subtraction operator overloading
	template<typename S>
	vector3D operator-(const vector3D<S> &q)
	{return vector3D(x-q.x,y-q.y,z-q.z);}
	
	
	
	//! += operator overloading
	template<typename S>
	vector3D& operator+=(const vector3D<S> &q)
	{x += (T)q.x; y += (T)q.y; z += (T)q.z; return(*this);}
	
	
	
	//! -= operator overloading
	vector3D& operator-=(const vector3D &q)
	{x -= q.x; y -= q.y; z -= q.z; return(*this);}
	
  
	
	//! post-multiplication by a constant : operator overloading
	vector3D& operator*=(const T c)
	{x *= c; y *= c; z *= c; return *this; }
	
	
	
	//! logical equality : operator overloading
	int operator==(const vector3D &p) const
	{
		if(x != p.x or y != p.y or z != p.z) return false;
		return true;
	}
	
	

	//! comparison : operator overloading
	int operator<(const vector3D &p) const
	{
		if(x<p.x) return true;
		else if(x>p.x) return false;
		else if(y<p.y) return true;
		else if(y>p.y) return false;
		else if(z<p.z) return true;
		else return false;
	}

	
	
	//! logical inequality : operator overloading
	int operator!=(const vector3D &p) const
	{
		if(x != p.x or y != p.y or z != p.z) return true;
		return false;
	}
	
	
	
	//! symmetrizes the vector w.r.t to some lattice operations
	void sym(char s){
		T tmp;
		switch(s){
			case 'D': tmp = x; x = y; y = tmp; break;
			case 'X': x = -x; break;
			case 'Y': y = -y; break;
			case 'I': x = -x ; y = -y; break;
			case 'R': tmp = x; x = -y; y = tmp; break;
		}
	}
	
	
	
	//! applies a matrix operation \a A to the wavevector
	void transform(matrix<T> &A){
		T xx,yy,zz;
		QCM_ASSERT(A.r == 3 and A.c == 3);
		xx = x;
		yy = y;
		zz = z;
		x = A(0,0)*xx + A(0,1)*yy + A(0,2)*zz;
		y = A(1,0)*xx + A(1,1)*yy + A(1,2)*zz;
		z = A(2,0)*xx + A(2,1)*yy + A(2,2)*zz;
	}
	
	
	
	//! applies a matrix operation \a A to the wavevector
	void transform_transpose(matrix<T> &A){
		T xx,yy,zz;
		QCM_ASSERT(A.r == 3 and A.c == 3);
		xx = x;
		yy = y;
		zz = z;
		x = A(0,0)*xx + A(1,0)*yy + A(2,0)*zz;
		y = A(0,1)*xx + A(1,1)*yy + A(2,1)*zz;
		z = A(0,2)*xx + A(1,2)*yy + A(2,2)*zz;
	}
	
  
  
	//! checks whether the vector is zero
	bool is_null(){
		if(x==0 and y==0 and z==0) return true;
		else return false;
	}
	
  
  
	// perform a cyclic permutation of the components: z becomes x, etc.
	void cyclic(){ T tmp = z; z = x; x = y; y = tmp;}

  
  
  
	// rotates in the xy plan by an angle theta (in degrees)
	void rotate(double theta){
		theta *= M_PI/180;
		double c = cos(theta);
		double s = sin(theta);
		T xx = x*c + y*s;
		T yy = y*c - x*s;
		x = xx;
		y = yy;
	}

	// length
	double norm(){
		return sqrt(x*x + y*y + z*z);
	}
};



template <typename T>
int vector3D<T>::dimension = 3;




/**
 writing to a stream
 */
template <typename T>
std::ostream & operator<<(std::ostream &flux, const vector3D<T> &x){
	ostringstream out;
	if(x.dimension==2) out << '(' << chop(x.x) << ',' << chop(x.y) << ')';
	else out << '(' << chop(x.x) << ',' << chop(x.y) << ',' << chop(x.z) << ')';
	flux << out.str();
	return flux;
}





/**
 reading from a stream
 */
template <typename T>
std::istream & operator>>(std::istream &flux, vector3D<T> &x){
  char lpar='(';
  char vir=',';
  char rpar=')';
	flux >> lpar >> x.x >> vir >> x.y >> vir >> x.z >> rpar;
	if(lpar != '(' or rpar != ')' or vir != ',')
	qcm_throw("3D vector has wrong format! Check parentheses, etc. Should have 3 components.");
	return flux;
}




/**
 overloading addition
 */
template <typename T>
vector3D<T> operator+(const vector3D<T> &k, const vector3D<T> &q)
{
  return(vector3D<T>(k.x+q.x,k.y+q.y,k.z+q.z));
}




/**
 overloading subtraction
 */
template <typename T>
vector3D<T> operator-(const vector3D<T> &k, const vector3D<T> &q)
{
	return(vector3D<T>(k.x-q.x,k.y-q.y,k.z-q.z));
}




/**
 overloading * for multiplying by a scalar
 */
template <typename T, typename S>
vector3D<T> operator*(const vector3D<T> &k, const S &a)
{
	return vector3D<T>(k.x*a,k.y*a,k.z*a);
}
template <typename T, typename S>
vector3D<T> operator*(const S &a, const vector3D<T> &k)
{
	return(vector3D<T>(k.x*a,k.y*a,k.z*a));
}




/**
 scalar product : operator overloading
 */
template<typename S, typename T>
T operator*(const vector3D<T> &A, const vector3D<S> &B)
{
	return(A.x*B.x + A.y*B.y + A.z*B.z);
}




/**
 cross, or vector product
 */
template <typename T>
vector3D<T> vector_product(const vector3D<T> &p1, const vector3D<T> &p2)
{
	return vector3D<T>(p1.y*p2.z - p1.z*p2.y, p1.z*p2.x - p1.x*p2.z, p1.x*p2.y - p1.y*p2.x);
}




/**
 triple product product
 */
template <typename T>
T triple_product(const vector3D<T> &E1, const vector3D<T> &E2, const vector3D<T> &E3)
{
	return(-E1.z*E2.y*E3.x + E1.y*E2.z*E3.x + E1.z*E2.x*E3.y - E1.x*E2.z*E3.y - E1.y*E2.x*E3.z + E1.x*E2.y*E3.z);
}



/**
 comparison operator, based on a dictionary-like order
 */
template <typename T>
bool operator<(const vector3D<T> &p1, const vector3D<T> &p2)
{
	if(p1.x < p2.x) return true;
	else if(p1.x > p2.x) return false;
	else if(p1.y < p2.y) return true;
	else if(p1.y > p2.y) return false;
	else if(p1.z < p2.z) return true;
	else return false;
}




/**
 overloading logical equality operator
 */
template <typename T>
bool operator==(const vector3D<T> &p1, const vector3D<T> &p2)
{
	if(p1.x != p2.x or p1.y != p2.y or p1.z != p2.z) return false;
	else return true;
}




/**
 overloading logical inequality operator
 */
template <typename T>
bool operator!=(const vector3D<T> &p1, const vector3D<T> &p2)
{
	if(p1.x != p2.x or p1.y != p2.y or p1.z != p2.z) return true;
	else return false;
}




/**
 reading the vectors
 */
template <typename T>
vector<vector3D<T>> read_vectors(istream &fin){
  vector<vector3D<T>> v;
  
	while(true){
	vector<string> input = read_strings(fin);
	if(input.size() == 0) break;
		vector3D<T> _x(input[0]);
		v.push_back(_x);
	}
  return v;
}



#endif
