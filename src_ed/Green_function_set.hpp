#ifndef Green_function_set_h
#define Green_function_set_h

#include <memory>

#include "block_matrix.hpp"
#include "symmetry_group.hpp"
#include "Q_matrix.hpp"
#include "hdf5_io.hpp"

enum GF_FORMAT {GF_format_BL, GF_format_CF, GF_format_MCF, GF_format_Q_to_MCF};

//! Abstract class for a Green function solution that can provide the Green function at any complex frequency
struct Green_function_set
{
  shared_ptr<symmetry_group> group; //!< point group
  size_t L; //!< number of rows in the Green function matrix
  int n_mixed; //!< size of the GF matrix, in terms of the number of sites (= 1, 2 or 4)
  
  Green_function_set(shared_ptr<symmetry_group> _group, int mixing) : group(_group)
  {
    L = group->n_sites;
    if(mixing & HS_mixing::anomalous) L *= 2;
    if(mixing & HS_mixing::spin_flip) L *= 2;
    n_mixed = L/group->n_sites;
  }
 
  
  // fills the block_matrix G with the GF at frequency z
  virtual void  Green_function(const Complex &z, block_matrix<Complex> &G) = 0;
  virtual void  integrated_Green_function(block_matrix<Complex> &M) = 0;
  
  // HDF5 serialization (write into / read from an already-opened group)
  virtual void write_hdf5(H5::Group& grp) = 0;
  virtual void read_hdf5(H5::Group& grp)  = 0;
};

#endif /* Green_function_set_h */
