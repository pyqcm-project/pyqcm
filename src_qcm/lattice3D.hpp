#ifndef lattice_h
#define lattice_h

/**
 * @file lattice3D.hpp
 * @brief Declaration of the lattice3D struct (integer-valued 3D Bravais lattice).
 */

#include <iostream>
#include <vector>
#include <utility>

#include "vector3D.hpp"
#include "basis3D.hpp"

//! used to represent the basis of a lattice
/**
 The use of a "working lattice" (e.g. Cartesian) is assumed throughout, and all vectors of the lattice object
 are expressed as integer combinations of this working basis.
 It is very similar to the basis3D class, except that its basis vectors have integer components
 */
struct lattice3D
{
	size_t D; //!< dimension of the lattice (0, 1, 2 or 3)
	vector<vector3D<int64_t>> e; //!< Bravais vectors of the lattice (integer)
	int64_t vol; //!< Volume of the unit cell (or equivalent measure in dimension D)
	matrix<double> M; //!< matrix allowing to express any vector in the integer basis
	
	lattice3D() {}
	lattice3D(vector<vector3D<int64_t>> _e) : e(_e) {init();}
	lattice3D(vector<int64_t> _e);
	void read(istream &fin);
	void trivial();
	void init();
	
	std::pair<vector3D<int64_t>, vector3D<int64_t>> fold(const vector3D<int64_t> &r);
	void dual(basis3D &x);
	vector3D<double> to(vector3D<int64_t> v);
	vector3D<double> from(vector3D<double> v);
	
	friend std::ostream & operator<<(std::ostream &flux, lattice3D &latt);
};


#endif


