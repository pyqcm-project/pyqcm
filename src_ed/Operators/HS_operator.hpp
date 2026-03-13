#ifndef HS_operator_H
#define HS_operator_H

#include <iostream>
#include <cstdio>
#include <cstring>

#include "matrix.hpp"

#define MIN_SIZE 16

//! Stores a non-Hermitian operator, such as a destruction operator, in the Hilbert space
template<typename T>
struct HS_operator
{
	size_t r; //! number of rows
	size_t c; //! number of columns
    vector<pair<T, vector<pair<uint32_t, uint32_t>>>> v; //! sparse form of the operator

	//! basic constructor
	HS_operator(uint32_t _r, uint32_t _c) : r(_r), c(_c) {v.reserve(8);}

	//! inserts an element 
	void insert(T value, uint32_t I, uint32_t J){
		size_t i;
		for(i=0; i<v.size(); i++){
			if(value == v[i].first){
			// if(abs(value - v[i].first)){
				v[i].second.push_back({I,J});
				break;
			}
		}
		if(i==v.size()){
			v.push_back({value, vector<pair<uint32_t, uint32_t>>()});
			v.back().second.push_back({I,J});
		}
	}

	void consolidate()
	{
		for(auto& x : v) std::sort(x.second.begin(), x.second.end());
	}

	//! multiplies by a vector
	void multiply_add(const vector<T> &x, vector<T> &y, T z)
	{
		for(auto& w : v){
			T z2(w.first*z);
			for(auto& e : w.second) y[e.first] += z2*x[e.second];
		}
	}

	//! multiplies the Hermitian conjugate by a vector
	void multiply_add_conjugate(const vector<T> &x, vector<T> &y, T z)
	{
		for(auto& w : v){
			T z2(conjugate(w.first)*z);
			for(auto& e : w.second) y[e.second] += z2*x[e.first];
		}
	}
	
	void dense_form(matrix<T> &h, double z=1.0){
		QCM_ASSERT(h.r == r && h.c == c);
		for(auto& w : v){
			T z2=w.first*z;
			for(auto& e : w.second) h(e.first, e.second) += z2;
		}
	}
};		

#endif
