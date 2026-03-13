#ifndef sparse_matrix_h
#define sparse_matrix_h

#include "index_pair.hpp"
#include <map>
#include <string>

//! sparse matrix, in the simple format of element enumeration
template<typename T>
struct sparse_matrix
{
	size_t r;
	size_t c;
	map<index_pair, T> el;
	
	sparse_matrix() : r(0), c(0) {}
	
	sparse_matrix(size_t _r) : r(_r), c(_r) {}
	
	sparse_matrix(size_t _r, size_t _c) : r(_r), c(_c) {el.reserve(r);}
	
	sparse_matrix(size_t _r, size_t _c, const map<index_pair, T>& _el) : r(_r), c(_c), el(_el) {}
	
	inline size_t size()
	{
		return el.size();
	}
	
	inline void set_size(size_t _r, size_t _c)
	{
		r = _r;
		c = _c;
	}
	
	inline void set_size(size_t _r)
	{
		r = _r;
		c = _r;
	}
	
	inline void insert(size_t _r, size_t _c, T v)
	{
		if(_r > r or _c > c) QCM_ASSERT(_r < r and _c < c);
		el[index_pair(_r,_c)] += v;
	}
	
	inline void insert(size_t _r, T v)
	{
		QCM_ASSERT(_r < r);
		el[index_pair(_r,_r)] += v;
	}
	
	inline void insert(const index_pair &m, T v)
	{
		QCM_ASSERT(m.r < r and m.c < c);
		if(abs(v)>SMALL_VALUE) el[m] += v;
	}
	
	inline size_t n() const {return el.size();}
	
	friend std::ostream & operator<<(std::ostream &s, const sparse_matrix &x)
	{
		for(auto &y : x.el) s << std::dec << y.first.r+1 << '\t' << y.first.c+1 << '\t' << y.second << endl;
		return s;
	}
	
	friend std::istream & operator>>(std::istream &flux, sparse_matrix &x)
	{
		string line;
		size_t r,c;
		T elem;
		while(true){
			getline(flux,line);
			if(line.size()==0 or !isdigit(line[0]) or flux.eof()) break;
			istringstream iss(line);
			iss >> r >> c >> elem;
			x.insert(r-1,c-1,elem);
		}
		return flux;
	}
};

#endif
