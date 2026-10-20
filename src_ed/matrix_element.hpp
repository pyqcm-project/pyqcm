#ifndef matrix_element_h
#define matrix_element_h

/**
 * @file matrix_element.hpp
 * @brief Definitions of the matrix_element template and diagonal_matrix_element struct.
 */

#include <iostream>
#include <string>
#include <cstdint>

using namespace std;


//! used to store a matrix element, location and value
//Moise: this definition is compliant with Eigen Triplet class
template<typename T>
struct matrix_element
{
	size_t r;
	size_t c;
  T v;
	
	matrix_element() : r(0), c(0), v(0) {}
	
	matrix_element(size_t _r, size_t _c, T _v) : r(_r), c(_c), v(_v) {}
  
	friend std::ostream & operator<<(std::ostream &s, const matrix_element &e)
	{
    s << '(' << e.r+1 << ',' << e.c+1 << ") : " << e.v;
		return s;
	}
	
  string str() const{
		ostringstream sout;
		sout << *this;
		return sout.str();
	}

	friend inline bool operator<(const matrix_element &x, const matrix_element &y){
		if(x.r < y.r) return true;
		else if(x.r > y.r) return false;
		else if(x.c < y.c) return true;
		else return false;
	}
	
  const size_t& row() const { return r; }
  const size_t& col() const { return c; }
  const T& value() const { return v; }
	
};




struct diagonal_matrix_element{
  uint32_t r;
  double v;
  
  diagonal_matrix_element(uint32_t _r, double _v) : r(_r), v(_v) {}

  friend inline bool operator<(const diagonal_matrix_element &x, const diagonal_matrix_element &y){
    if(x.r < y.r) return true;
    else return false;
  }
} ;

#endif
