#ifndef lattice_hybrid_h
#define lattice_hybrid_h
#include <iostream>

#include <complex>
#include <cstring>
#include <ctype.h>
#include "types.hpp"
#include "vector3D.hpp"

using namespace std;

struct lattice_hybrid {
  vector<double> w; //!< array of frequencies (along the imaginary axis)
  vector<double> weight; //!< weight of each frequency in an integral over frequencies
	vector<vector3D<double>> k; //!< list of wavevectors (k-points) used with external hybridization. In the superdual basis.
  size_t nw=0; //!< number of frequencies
  size_t nk=0; //!< number of k points
  size_t d=0; //!< dimension of hybridization function
  vector<double> R; //!< array of values (real part, ow-major)
  vector<double> I; //!< array of values (imaginary part, ow-major)
  int mixing; //!< mixing state of the hybridization function
  lattice_hybrid(const string &filename); //!< constructor from HDF5 file
};

#endif


