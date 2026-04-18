#include <H5Cpp.h>
#include "lattice_hybrid.hpp"

using namespace std;

lattice_hybrid::lattice_hybrid(const string &filename){
    
  const H5std_string FILE_NAME(filename);

  try {
      H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

      // Get frequencies
      H5::DataSet w_set = file.openDataSet("w");
      H5::DataSpace dataspace = w_set.getSpace();
      int rank = dataspace.getSimpleExtentNdims();
      if (rank != 1) qcm_throw("Dataset containing frequencies is not of rank 1");
      hsize_t dims[4];
      dataspace.getSimpleExtentDims(dims);
      nw = dims[0];
      w.resize(nw);
      w_set.read(w.data(), H5::PredType::NATIVE_DOUBLE);

      // Get the weights
      H5::DataSet weight_set = file.openDataSet("weight");
      dataspace = weight_set.getSpace();
      rank = dataspace.getSimpleExtentNdims();
      if (rank != 1) qcm_throw("Dataset containing weights is not of rank 1");
      dataspace.getSimpleExtentDims(dims);
      nw = dims[0];
      weight.resize(nw);
      weight_set.read(weight.data(), H5::PredType::NATIVE_DOUBLE);


      // Get k-points
      H5::DataSet k_set = file.openDataSet("k");
      dataspace = k_set.getSpace();
      rank = dataspace.getSimpleExtentNdims();
      if (rank != 2) qcm_throw("Dataset containing k-points is not of rank 2");
      dataspace.getSimpleExtentDims(dims);
      nk = dims[0];
      vector<double> k_tmp(3*nk);
      k.resize(nk);
      k_set.read(k_tmp.data(), H5::PredType::NATIVE_DOUBLE);
      for(int ik; ik<nk; ik++){
        k[ik] = {k_tmp[3*ik], k_tmp[3*ik+1], k_tmp[3*ik+2]};
      }

      // Get hybridization function (real part)
      H5::DataSet hybrid_real_set = file.openDataSet("hybrid_real");
      dataspace = hybrid_real_set.getSpace();
      rank = dataspace.getSimpleExtentNdims();
      if (rank != 4) qcm_throw("Dataset containing hybridization is not of rank 4");
      dataspace.getSimpleExtentDims(dims);
      d = dims[2];
      R.resize(nw*nk*d*d);
      hybrid_real_set.read(R.data(), H5::PredType::NATIVE_DOUBLE);

      // Get hybridization function (imaginary part)
      H5::DataSet hybrid_imag_set = file.openDataSet("hybrid_imag");
      dataspace = hybrid_imag_set.getSpace();
      rank = dataspace.getSimpleExtentNdims();
      if (rank != 4) qcm_throw("Dataset containing hybridization is not of rank 4");
      dataspace.getSimpleExtentDims(dims);
      d = dims[2];
      I.resize(nw*nk*d*d);
      hybrid_imag_set.read(I.data(), H5::PredType::NATIVE_DOUBLE);

      std::cout << "Read HDF5 file for external hybridization:\n"
                << dims[0] << " frequencies\n"
                << dims[1] << " k-points\n"
                << dims[2] << " x "
                << dims[3] << " matrices" << std::endl;

      // Get the mixing
      H5::Attribute attr = file.openAttribute("mixing");
      attr.read(H5::PredType::NATIVE_INT, &mixing);

  } catch (H5::Exception& error) {
      qcm_throw("Failed to read HDF5 file");
  }
} 
