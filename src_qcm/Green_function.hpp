#ifndef Green_function_hpp
#define Green_function_hpp

#include "vector3D.hpp"
#include "block_matrix.hpp"



//! structure defining the cluster Green function and related quantities (self-energy, hybridization function)
struct Green_function
{
	Complex w;
	int iw; //!< index of frequency (for models with frequency grids only, e.g. with external hybridization)
	bool spin_down;
	block_matrix<Complex> G;
	block_matrix<Complex> gamma; // hybridization functions
	block_matrix<Complex> sigma; // self-energies
};


//! structure defining the lattice, k-dependent Green function and related quantities (self-energy, periodized GF, t(k), etc.)
struct Green_function_k
{
	Green_function &G; //!< reference to a cluster Green function
	static size_t dim_GF; //!< dimension of the cluster Green function 
	static size_t dim_reduced_GF; //!< dimension of the periodized Green function
	vector3D<double> k; //!< wave vector in the superdual basis (reduced BZ (\tilde k), in units of 2pi. Its range is from 0 to 1 for each component)
	int ik; //!< index of wavevector (for models with external hybridization only)
	matrix<Complex> Gcpt; //!< CPT Green function (non periodized)
	matrix<Complex> t; //!< k-dependent one-body (hopping) matrix : \f$t_{\mathbf{k}} = \omega - G_0^{-1}\f$
	matrix<Complex> V; //!< \f$G_0^{-1}-G_0'{}^{-1}\f$
 	matrix<Complex> g; //!< periodized Green function
	matrix<Complex> sigma; //!< self energy associated with g
	vector<Complex> phase; //!< k-dependent phases related to neighboring clusters used in computations
	
	//! constructor from a cluster Green function and k value
	Green_function_k(Green_function &_G, const vector3D<double> &_k, const int &_ik = -1) : G(_G), k(_k), ik(_ik){
		QCM_ASSERT(G.G.r > 0);
		V.set_size(dim_GF);
		t.set_size(dim_GF);
		Gcpt.set_size(dim_GF);
		g.set_size(dim_reduced_GF);
		sigma.set_size(dim_reduced_GF);
	}

};


#endif
