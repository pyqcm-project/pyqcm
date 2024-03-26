#ifndef qcm_wrap_h
#define qcm_wrap_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <memory>
#include <array>
// #include<unordered_map>
#include<map>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include<string>
#include "ndarrayobject.h"
#include "float.h"
#include "parser.hpp"
#include "QCM.hpp"
#include "qcm_ED.hpp"
#include "parameter_set.hpp"
#include "lattice_model.hpp"
#include "common_Py.hpp"

vector3D<int64_t> intvector_from_Py(PyArrayObject *k_pyobj);
vector3D<double> vector_from_Py(PyArrayObject *k_pyobj);
vector<vector3D<double>> many_vectors_from_Py(PyArrayObject *k_pyobj);
vector<string> strings_from_PyList(PyObject* lst);

extern shared_ptr<lattice_model> qcm_model;
extern run_statistics STATS;

// extern unordered_map<string, global_parameter<bool>> GP_bool;
extern map<string, global_parameter<bool>> GP_bool;
void qcm_catch(const string& s);

static PyObject *qcm_Error;

//******************************************************************************
// Wrappers
//==============================================================================
const char* add_cluster_help =
R"{(
Adds a cluster to the repeated unit
arguments:
1. name of cluster model (string)
2. position of cluster (array of ints)
3. list of positions of sites (2D array of ints)
4. label (starts at 1) of cluster to which this one is entirely equivalent [default = 0, no such equivalence]
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* add_cluster_python(PyObject *self, PyObject *args)
{
  char* s1 = nullptr;
  PyArrayObject *pos_pyobj = nullptr;
  PyArrayObject *cpos_pyobj = nullptr;
  int ref = 0;
  
  try{
    if(!PyArg_ParseTuple(args, "sOO|i", &s1, &cpos_pyobj, &pos_pyobj, &ref))
      qcm_throw("failed to read parameters in call to lattice_model (python)");
  
    QCM::add_cluster(string(s1), intvector_from_Py(cpos_pyobj), many_intvectors_from_Py(pos_pyobj), ref);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* averages_help =
R"{(
returns the average values of all operators in a model instance.
arguments :
label (optional) :  label of the model instance (default = 0)
returns : a dict of string:value){";
//------------------------------------------------------------------------------
static PyObject* averages_python(PyObject *self, PyObject *args)
{
  vector<pair<string,double>> ave;
  int label=0;
  PyObject *lst = nullptr;

  try{
    if(!PyArg_ParseTuple(args, "|Oi", &lst, &label))
      qcm_throw("failed to read parameters in call to averages (python)");
    vector<string> ops = {};
    if(lst != nullptr and PyList_Size(lst)>0) ops = strings_from_PyList(lst);
    ave =  QCM::averages(ops,label);
  } catch(const string& s) {qcm_catch(s);}
  
  PyObject *dic = PyDict_New();
  for(auto& x : ave){
    PyDict_SetItem(dic, Py_BuildValue("s#", x.first.c_str(), x.first.length()), Py_BuildValue("d", x.second));
  }
  return dic;
}

//==============================================================================
const char* Berry_flux_help =
R"{(
computes the Berry curvature in a region of the Brillouin zone. Works in 2D only.
arguments :
1. k1 : wavevector, lower left bound of the region
2. k2 : wavevector, upper right bound of the region
3. nk : number of wavevectors in x and y
4. orb : the lattice orbital number to use, from 1 to the number of lattice orbitals (default 0 = all)
5. rec : true is recursion is used (subdivision of intervals)
6. label (optional) :  label of the model instance (default 0)
returns : a 2D array of values.

if the boolean global parameter 'dual_basis' is set, then it is possible to limit
the region to the Brillouin zone (length 1 in x and y) and then the Chern number
is the average value of the array.
){";

//------------------------------------------------------------------------------
static PyObject* Berry_flux_python(PyObject *self, PyObject *args)
{
  int label=0;
  int orb=0;
  PyArrayObject *k_pyobj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "O|ii", &k_pyobj, &orb, &label))
      qcm_throw("failed to read parameters in call to Berry_flux (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  vector<vector3D<double>> k = many_vectors_from_Py(k_pyobj);
  double g;
  try{
    g = QCM::Berry_flux(k, orb, label);
  } catch(const string& s) {qcm_catch(s);}

  return Py_BuildValue("d", g);
}

//==============================================================================
const char* Berry_curvature_help =
R"{(
computes the Berry curvature in a region of the Brillouin zone. Works in 2D only.
arguments :
1. k1 : wavevector, lower left bound of the region
2. k2 : wavevector, upper right bound of the region
3. nk : number of wavevectors in x and y
4. orb : the lattice orbital to use, from 1 to the number of lattice orbitals (default 0 = all)
5. rec : true is recursion is used (subdivision of intervals)
6. label (optional) :  label of the model instance (default 0)
returns : a 2D array of values.

if the boolean global parameter 'dual_basis' is set, then it is possible to limit
the region to the Brillouin zone (length 1 in x and y) and then the Chern number
is the average value of the array.
){";

//------------------------------------------------------------------------------
static PyObject* Berry_curvature_python(PyObject *self, PyObject *args)
{
  int label=0;
  int orb=0;
  int rec=0;
  PyArrayObject *k1_pyobj = nullptr;
  PyArrayObject *k2_pyobj = nullptr;
  int nk;
  int dir;
  
  try{
    if(!PyArg_ParseTuple(args, "OOi|iiii",&k1_pyobj, &k2_pyobj, &nk, &orb, &rec, &dir, &label))
      qcm_throw("failed to read parameters in call to Berry_curvature (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  vector3D<double> k1 = vector_from_Py(k1_pyobj);
  vector3D<double> k2 = vector_from_Py(k2_pyobj);
  
  vector<double> g;
  try{
    g = QCM::Berry_curvature(k1, k2, nk, orb, (bool)rec, dir, label);
  } catch(const string& s) {qcm_catch(s);}

  npy_intp dims[2];
  dims[0] = dims[1] = nk;
  
  PyObject *out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(double));
  PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  return out;
}


//==============================================================================
const char* cluster_Green_function_help =
R"{(
computes the cluster Green function at a given frequency
arguments:
1. cluster : label of the cluster (0 to the number of clusters -1)
2. z : complex frequency
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a complex-valued matrix
){";
//------------------------------------------------------------------------------
static PyObject* cluster_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int clus;
  int spin_down=0;
  int blocks=0;
  complex<double> z;
  
  try{
    if(!PyArg_ParseTuple(args, "iD|iii", &clus, &z, &spin_down, &label, &blocks))
      qcm_throw("failed to read parameters in call to cluster_Green_function (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  size_t d;
  matrix<complex<double>> g;
  try{
    g = QCM::cluster_Green_function((size_t)clus, z, (bool)spin_down, label, blocks);
    d = qcm_model->GF_dims[clus];
  } catch(const string& s) {qcm_catch(s);}
  
  
  npy_intp dims[2];
  dims[0] = dims[1] = d;
  
  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  return out;
}


//==============================================================================
const char* cluster_self_energy_help =
R"{(
computes the cluster self energy at a given frequency
arguments:
1. cluster : label of the cluster (0 to the number of clusters -1)
2. z : complex frequency
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a complex-valued matrix
){";
//------------------------------------------------------------------------------
static PyObject* cluster_self_energy_python(PyObject *self, PyObject *args)
{
  int label=0;
  int clus;
  int spin_down=0;
  complex<double> z;
  
  try{
    if(!PyArg_ParseTuple(args, "iD|ii", &clus, &z, &spin_down, &label))
      qcm_throw("failed to read parameters in call to cluster_self_energy (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  size_t d;
  matrix<complex<double>> g;
  try{
    g = QCM::cluster_self_energy((size_t)clus, z, (bool)spin_down, label);
    d = qcm_model->GF_dims[clus];
  } catch(const string& s) {qcm_catch(s);}
  
  
  npy_intp dims[2];
  dims[0] = dims[1] = d;
  
  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  return out;
}



//==============================================================================
const char* cluster_hopping_matrix_help =
R"{(
returns the one-body matrix of cluster no i for instance 'label'
arguments:
1. cluster : label of the cluster (0 to the number of clusters -1)
2. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
3. label (optional) :  label of the model instance (default 0)
returns: a complex-valued matrix
){";
//------------------------------------------------------------------------------
static PyObject* cluster_hopping_matrix_python(PyObject *self, PyObject *args)
{
  int label=0;
  int clus;
  int spin_down=0;
  PyObject *out;
  
  try{
    if(!PyArg_ParseTuple(args, "i|ii", &clus, &spin_down, &label))
      qcm_throw("failed to read parameters in call to cluster_hopping_matrix (python)");
  
    auto g = QCM::cluster_hopping_matrix((size_t)clus, (bool)spin_down, label);
    
    size_t d = qcm_model->GF_dims[clus];
    
    npy_intp dims[2];
    dims[0] = dims[1] = d;
    
    out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}

//==============================================================================
const char* cluster_info_help =
R"(
arguments:
None
returns:
  A list of 5-tuples : (str, int, int) : name of the cluster model, number of sites, number of bath sites, dimension of the Green function, number of symmetry  operations
)";
//------------------------------------------------------------------------------
static PyObject* cluster_info_python(PyObject *self, PyObject *args)
{
  vector<tuple<string, int, int, int, int>> info = QCM::cluster_info();
  PyObject *lst = PyList_New(info.size());
  for(size_t i=0; i< info.size(); i++){
    PyObject* elem = PyTuple_New(5);
    PyTuple_SetItem(elem, 0, Py_BuildValue("s", get<0>(info[i]).c_str()));
    PyTuple_SetItem(elem, 1, Py_BuildValue("i", get<1>(info[i])));
    PyTuple_SetItem(elem, 2, Py_BuildValue("i", get<2>(info[i])));
    PyTuple_SetItem(elem, 3, Py_BuildValue("i", get<3>(info[i])));
    PyTuple_SetItem(elem, 4, Py_BuildValue("i", get<4>(info[i])));
    PyList_SET_ITEM(lst, i, elem);
  }
  return lst;
}
  
  
//==============================================================================
const char*  CDMFT_variational_set_help =
R"{(
defines the set of CDMFT variational parameters
arguments:
1. a vector of strings
){";
//------------------------------------------------------------------------------
static PyObject*  CDMFT_variational_set_python(PyObject *self, PyObject *args)
{
  PyObject *v = nullptr;

  try{
    if(!PyArg_ParseTuple(args, "O", &v))
      qcm_throw("failed to read parameters in call to CDMFT_variational_set (python)");
    qcm_model->param_set->CDMFT_variational_set(strings_from_PyList(v));
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char*  CDMFT_host_help =
R"{(
computes the CDMFT host function
arguments:
1. a vector of double : frequencies
2. a vector of double : weights
3. int : label of instance
){";
//------------------------------------------------------------------------------
static PyObject*  CDMFT_host_python(PyObject *self, PyObject *args)
{
  PyObject *freqs = nullptr;
  PyObject *weights = nullptr;
  int label = 0;

  try{
    if(!PyArg_ParseTuple(args, "OO|i", &freqs, &weights, &label))
      qcm_throw("failed to read parameters in call to CDMFT_host (python)");
    vector<double> _freqs = doubles_from_Py(freqs);
    vector<double> _weights = doubles_from_Py(weights);
    QCM::CDMFT_host(_freqs, _weights, label);
  }catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

  
//==============================================================================
const char*  set_CDMFT_host_help =
R"{(
sets the CDMFT host function from data
arguments:
1. a vector of double : frequencies
2. int : a cluster label
3. a vector of complex matrices : host function
4. bool : spin-down tag
5. int : label of instance
){";
//------------------------------------------------------------------------------
static PyObject*  set_CDMFT_host_python(PyObject *self, PyObject *args)
{
  PyObject *freqs = nullptr;
  PyObject *H = nullptr;
  int label = 0;
  int clus = 0;
  int spin_down = 0;

  try{
    if(!PyArg_ParseTuple(args, "iOO|ii", &label, &freqs, &H, &clus, &spin_down))
      qcm_throw("failed to read parameters in call to set_CDMFT_host (python)");
    vector<double> _freqs = doubles_from_Py(freqs);
    vector<matrix<complex<double>>> Hv = complex_array3_from_Py((PyArrayObject*)H);
    QCM::set_CDMFT_host(label, _freqs, clus, Hv, spin_down);
  }catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}



//==============================================================================
const char*  get_CDMFT_host_help =
R"{(
retrieves the CDMFT host function
arguments:
1. int : the cluster label (from 0 to the number of clusters - 1)
2. boolean : spin_down sector flag
3. int : label of instance
){";
//------------------------------------------------------------------------------
static PyObject*  get_CDMFT_host_python(PyObject *self, PyObject *args)
{
  int label = 0;
  int spin_down = 0;
  int clus = 0;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "|iii", &clus, &spin_down, &label))
      qcm_throw("failed to read parameters in call to get_CDMFT_host (python)");
    auto g = QCM::get_CDMFT_host(spin_down, label);
    size_t d = qcm_model->GF_dims[clus];
    npy_intp dims[3];
    dims[0] = g.size();
    dims[1] = dims[2] = d;

    vector<Complex> H(d*d*dims[0]);
    int l = 0;
    for(int i=0; i<dims[0]; i++){
      for(int j=0; j<d; j++){
        for(int k=0; k<d; k++){
          H[l++] = g[i][clus](j,k);
        }
      } 
    }
    
    out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject*) out), H.data(), dims[0]*d*d*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  
  }catch(const string& s) {qcm_catch(s);}
  return out;
}


//==============================================================================
const char*  CDMFT_distance_help =
R"{(
defines the set of CDMFT variational parameters
arguments:
1. a vector of double : values of variational parameters
2. int : label of the instance
returns : a float
){";
//------------------------------------------------------------------------------
static PyObject*  CDMFT_distance_python(PyObject *self, PyObject *args)
{
  PyObject *val = nullptr;
  int label = 0;
  double d;
  try{
    if(!PyArg_ParseTuple(args, "O|i", &val, &label))
      qcm_throw("failed to read parameters in call to CPT_Green_function (python)");
    vector<double> _val = doubles_from_Py(val);;
    d = QCM::CDMFT_distance(_val, label);
  }catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("d", d);
}





//==============================================================================
const char* CPT_Green_function_help =
R"{(
computes the CPT Green function at a given frequency
arguments:
1. z : complex frequency
2. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single or an array of complex-valued matrices
){";
//------------------------------------------------------------------------------
static PyObject* CPT_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to CPT_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'CPT_Green_function' should be of dimension 1 or 2");
    
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      matrix<complex<double>> g = QCM::CPT_Green_function(z, k, (bool)spin_down, label);
      size_t d = QCM::Green_function_dimension();
      npy_intp dims[2];
      dims[0] = dims[1] = d;
      out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
      memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
      PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
    }
    else{
      vector<vector3D<double>> k = many_vectors_from_Py(k_pyobj);
      vector<matrix<complex<double>>> g = QCM::CPT_Green_function(z, k, (bool)spin_down, label);
      size_t d = QCM::Green_function_dimension();
      npy_intp dims[3];
      dims[0] = g.size();
      dims[1] = dims[2] = d;
      out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
      for(size_t j = 0; j<g.size(); j++){
        memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
      }
      PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
    } 
  }catch(const string& s) {qcm_catch(s);}
  return out;
}

  
  
//==============================================================================
const char* CPT_Green_function_inverse_help =
R"{(
computes the inverse CPT Green function at a given frequency
arguments:
1. z : complex frequency
2. k : array of wavevectors (ndarray(N,3))
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single or an array of complex-valued matrices
){";
//------------------------------------------------------------------------------
static PyObject* CPT_Green_function_inverse_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyObject *out;
  PyArrayObject *k_pyobj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to CPT_Green_function_inverse (python)");
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim != 2) qcm_throw("Argument 3 of 'CPT_Green_function_inverse' should be of dimension 2");
    
    vector<vector3D<double>> k = many_vectors_from_Py(k_pyobj);
    vector<matrix<complex<double>>> g = QCM::CPT_Green_function_inverse(z, k, (bool)spin_down, label);
    
    size_t d = QCM::Green_function_dimension();
    npy_intp dims[3];
    dims[0] = g.size();
    dims[1] = dims[2] = d;
    out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    for(size_t j = 0; j<g.size(); j++){
      memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}


//==============================================================================
const char* dispersion_help =
R"{(
computes the dispersion relation for a single or an array of wavevectors
arguments:
1. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
2. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
3. label (optional): label of the model instance (default 0)
returns: a single (ndarray(d)) or an array (ndarray(N,d)) of real values (energies). d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* dispersion_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;
  
  try{
    if(!PyArg_ParseTuple(args, "O|ii", &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to dispersion (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'dispersion' should be of dimension 1 or 2");
    
    vector<vector3D<double>> kk;
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      kk.assign(1,k);
    }
    else{
      kk = many_vectors_from_Py(k_pyobj);
    }
    vector<vector<double>> g;
    try{
      g = QCM::dispersion(kk, (bool)spin_down, label);
    } catch(const string& s) {qcm_catch(s);}
    
    size_t d = QCM::reduced_Green_function_dimension();
    
    npy_intp dims[2];
    dims[0] = g.size();
    dims[1] = d;
    
    out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    for(size_t j = 0; j<g.size(); j++){
      memcpy((double*)PyArray_DATA((PyArrayObject*) out)+j*d, g[j].data(), d*sizeof(double));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}


//==============================================================================
const char* tk_help =
R"{(
computes the k-dependent one-body matrix
arguments:
1. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
2. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
3. label (optional) :  label of the model instance (default 0)
returns: a single or an array of complex-valued matrices
){";
//------------------------------------------------------------------------------
static PyObject* tk_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "O|ii", &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to CPT_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'CPT_Green_function' should be of dimension 1 or 2");
    
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      auto g = QCM::tk(k, (bool)spin_down, label);
      size_t d = QCM::Green_function_dimension();
      npy_intp dims[2];
      dims[0] = dims[1] = d;
      out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
      memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
      PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
    }
    else{
      vector<vector3D<double>> k = many_vectors_from_Py(k_pyobj);
      auto g = QCM::tk(k, (bool)spin_down, label);
      size_t d = QCM::Green_function_dimension();
      npy_intp dims[3];
      dims[0] = g.size();
      dims[1] = dims[2] = d;
      out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
      for(size_t j = 0; j<g.size(); j++){
        memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
      }
      PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
    } 
  }catch(const string& s) {qcm_catch(s);}
  return out;
}

//==============================================================================
const char* dos_help =
R"{(
computes the density of states at a given frequency.
arguments:
1. z : complex frequency
2. label (optional) :  label of the model instance (default 0)
returns: ndarray(d) of real values, d being the reduced GF dimension
){";
//------------------------------------------------------------------------------
static PyObject* dos_python(PyObject *self, PyObject *args)
{
  int label=0;
  complex<double> z;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "D|i", &z, &label))
      qcm_throw("failed to read parameters in call to dos (python)");
  
    vector<double> g;
    g = QCM::dos(z, label);
    npy_intp dims[1];
    dims[0] = g.size();
    
    out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(double));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}


    
//==============================================================================
const char* Green_function_dimension_help =
R"{(
returns the dimension of the CPT Green function matrix
arguments: None
returns: integer
){";
//------------------------------------------------------------------------------
static PyObject* Green_function_dimension_python(PyObject *self, PyObject *args)
{
  return Py_BuildValue("i", QCM::Green_function_dimension());
}



//==============================================================================
const char* Green_function_solve_help =
R"{(
forces the computation of Green functions in all clusters, i.e. in a non-lazy fashion.
arguments: None
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* Green_function_solve_python(PyObject *self, PyObject *args)
{
  int label=0;
  try{
    if(!PyArg_ParseTuple(args, "i", &label))
      qcm_throw("failed to read parameters in call to dos (python)");
      QCM::Green_function_solve(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}




//==============================================================================
const char* ground_state_help =
R"{(
computes the ground state of the cluster(s)
arguments:
1. label (optional) :  label of the model instance (default 0)
returns: a list of pairs (real/string) of the ground state energy and sector string, for each cluster of the system
){";
//------------------------------------------------------------------------------
static PyObject* ground_state_python(PyObject *self, PyObject *args)
{
  
  int label=0;
  PyObject *lst;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to ground_state (python)");
  
    vector<pair<double,string>> gs;
    gs = QCM::ground_state(label);
    lst = PyList_New(gs.size());
    for(size_t i=0; i< gs.size(); i++){
      PyObject* elem = PyTuple_New(2);
      PyTuple_SetItem(elem, 0, Py_BuildValue("d", gs[i].first));
      PyTuple_SetItem(elem, 1, Py_BuildValue("s#", gs[i].second.c_str(), gs[i].second.length()));
      PyList_SET_ITEM(lst, i, elem);
    }
  } catch(const string& s) {qcm_catch(s);}
  return lst;
}


//==============================================================================
const char* hybridization_function_help =
R"{(
returns the hybridization function for cluster 'cluster' and instance 'label'
arguments:
1. cluster : label of the cluster (0 to the number of clusters -1)
2. z : complex frequency
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a complex-valued matrix
){";
//------------------------------------------------------------------------------
static PyObject* hybridization_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int clus=0;
  int spin_down=0;
  complex<double> z;
  PyObject *out;
  try{
    if(!PyArg_ParseTuple(args, "D|iii", &z, &spin_down, &clus, &label))
      qcm_throw("failed to read parameters in call to hybridization_function (python)");
  
    auto g = QCM::hybridization_function(z, (bool)spin_down, (size_t)clus, label);
    size_t d = qcm_model->GF_dims[clus];
    
    npy_intp dims[2];
    dims[0] = dims[1] = d;
    
    out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}



//==============================================================================
const char* lattice_model_help =
R"{(
initiates the lattice model.
arguments:
1. name : the name of the model
2. superlattice : array of integers of shape (d,3), d being the dimension
3. lattice : array of integers of shape (d,3), d being the dimension
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* lattice_model_python(PyObject *self, PyObject *args)
{
  char* s1 = nullptr;
  PyArrayObject *super_pyobj = nullptr;
  PyArrayObject *unit_pyobj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "sO|O", &s1, &super_pyobj, &unit_pyobj))
      qcm_throw("failed to read parameters in call to lattice_model (python)");
    if(qcm_model==nullptr) qcm_throw("no cluster has been added to the model!");
  
    vector<int64_t> superlattice = intvectors_from_Py(super_pyobj);
    vector<int64_t> unit_cell;
    if(unit_pyobj==nullptr or (PyObject*)unit_pyobj==Py_None){
      unit_cell.assign(superlattice.size(),0);
      size_t dim = superlattice.size()/3;
      for(int i=0; i<dim; i++) unit_cell[i*4] = 1;
    }
    else unit_cell = intvectors_from_Py(unit_pyobj);
  
    QCM::new_lattice_model(string(s1), superlattice, unit_cell);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}



//==============================================================================
const char* mixing_help =
R"{(
returns the mixing state of the system:
0 : normal.  GF matrix is n x n, n being the number of sites
1 : anomalous. GF matrix is 2n x 2n
2 : spin-flip.  GF matrix is 2n x 2n
3 : anomalous and spin-flip (full Nambu doubling).  GF matrix is 4n x 4n
4 : up and down spins different.  GF matrix is n x n, but computed twice, with spin_down = false and true
arguments: None
returns: integer
){";
//------------------------------------------------------------------------------
static PyObject* mixing_python(PyObject *self, PyObject *args)
{
  PyArg_ParseTuple(args, "");
  int result = QCM::mixing();
  return Py_BuildValue("i", result);
}


//==============================================================================
const char* model_size_help =
R"(
arguments:
None
returns:
  A 4-tuple:
    1. the size of the supercell
    2. the number of lattice orbitals
    3. a tuple containing the sizes of each cluster
    4. a tuple containing the sizes of each cluster's bath
  
)";
//------------------------------------------------------------------------------
static PyObject* model_size_python(PyObject *self, PyObject *args)
{
  PyObject* elem = PyTuple_New(qcm_model->clusters.size());
  PyObject* bath = PyTuple_New(qcm_model->clusters.size());
  PyObject* ref = PyTuple_New(qcm_model->clusters.size());
  for(int i=0; i<qcm_model->clusters.size(); i++){
    auto s = ED::model_size(qcm_model->clusters[i].name);
    PyTuple_SetItem(elem, i, Py_BuildValue("i", get<0>(s)));
    PyTuple_SetItem(bath, i, Py_BuildValue("i", get<1>(s)));
    PyTuple_SetItem(ref, i, Py_BuildValue("i", qcm_model->clusters[i].ref));
  }
  return Py_BuildValue("iiOOO", qcm_model->sites.size(), qcm_model->n_band, elem, bath, ref);
}
  

//==============================================================================
const char* model_is_closed_help =
R"(
arguments:
None
returns:
  True if the model is no longer modifiable, False otherwise  
)";
//------------------------------------------------------------------------------
static PyObject* model_is_closed_python(PyObject *self, PyObject *args)
{
  return Py_BuildValue("i", (int)qcm_model->is_closed);
}


//==============================================================================
const char* momentum_profile_help =
R"{(
computes the momentum-resolved average of an operator
arguments:
1. string : name of the lattice operator
2. k : array of wavevectors (ndarray(N,3))
3. label (optional) :  label of the model instance (default 0)
returns: an array of values
){";
//------------------------------------------------------------------------------
static PyObject* momentum_profile_python(PyObject *self, PyObject *args)
{
  int label=0;
  char* s1 = nullptr;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;
  try{
    if(!PyArg_ParseTuple(args, "sO|i", &s1, &k_pyobj, &label))
      qcm_throw("failed to read parameters in call to momentum_profile (python)");

    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim != 2) qcm_throw("Argument 3 of 'momentum_profile' should be of dimension 2");

    vector<vector3D<double>> kk;
    kk = many_vectors_from_Py(k_pyobj);
    vector<double> g;
    g = QCM::momentum_profile(string(s1), kk, label);
    
    
    npy_intp dims[1];
    dims[0] = g.size();
    out = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(double));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}




//==============================================================================
const char* new_model_instance_help =
R"{(
Creates a new instance of the lattice model, with values associated to terms of the Hamiltonian.
arguments:
1. label (optional) :  label we choose for the instance and used later to refer to it (default 0)
returns: void
){";
//------------------------------------------------------------------------------
static PyObject* new_model_instance_python(PyObject *self, PyObject *args)
{
  int label=0;  
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to new_model_instance (python)");
    QCM::new_model_instance(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* set_parameters_help =
R"{(
arguments:
  1. the values/dependence of the parameters (array of 2- or 3-tuples)
returns: void
){";
//------------------------------------------------------------------------------
static PyObject* set_parameters_python(PyObject *self, PyObject *args)
{
  PyObject *py_values = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "O", &py_values))
      qcm_throw("failed to read parameters in call to set_parameters (python)");
    if(!PySequence_Check(py_values)) qcm_throw("argument 2 of 'set_parameters' should be a list");
    
    PyObject *pkey = nullptr;    
    size_t n = PySequence_Size(py_values);
    vector<pair<string,double>> values;
    pair<string,double> val;
    values.reserve(n);
    vector<tuple<string,double,string>> equiv;
    tuple<string,double,string> eq;
    equiv.reserve(n);
    for(size_t i=0; i<n; i++){
      pkey = PySequence_GetItem(py_values,i);
      if(PyTuple_Size(pkey) == 2){
        val.first = Py2string(PyTuple_GetItem(pkey, 0));
        val.second = PyFloat_AsDouble(PyTuple_GetItem(pkey, 1));
        values.push_back(val);
      }
      else if(PyTuple_Size(pkey) == 3){
        get<0>(eq) = Py2string(PyTuple_GetItem(pkey, 0));
        get<1>(eq) = PyFloat_AsDouble(PyTuple_GetItem(pkey, 1));
        get<2>(eq) = Py2string(PyTuple_GetItem(pkey, 2));
        equiv.push_back(eq);
      }
      else{
        qcm_throw("element "+to_string(i)+" of argument 2 of 'set_parameters' should be a 2-tuple or a 3-tuple");
      }
    }
    QCM::set_parameters(values, equiv);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* set_target_sectors_help =
R"{(
defines the target Hilbert space sectors to be used
arguments:
  1. the target sectors (array of strings)
returns: void
){";
//------------------------------------------------------------------------------
static PyObject* set_target_sectors_python(PyObject *self, PyObject *args)
{
  PyObject *py_sectors = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "O", &py_sectors))
      qcm_throw("failed to read parameters in call to set_target_sectors (python)");
    if(!PySequence_Check(py_sectors)) qcm_throw("argument 1 of 'set_target_sectors' should be a sequence");
    
    PyObject *pkey = nullptr;
    // processing py_sectors
    size_t n = PySequence_Size(py_sectors);
    if(n != qcm_model->clusters.size())
      qcm_throw("The number of strings in argument of 'set_target_sectors' ("+to_string(n)+") should be the number of clusters in the repeated unit ("+to_string(qcm_model->clusters.size())+")");
    qcm_model->sector_strings.resize(n);
    for(size_t i=0; i<n; i++){
      qcm_model->sector_strings[i] = Py2string(PySequence_GetItem(py_sectors,i));
    }
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* parameters_help =
R"{(
returns the values of the parameters in the parameter set
returns: a dictionary of string/real pairs
){";
//------------------------------------------------------------------------------
static PyObject* parameters_python(PyObject *self, PyObject *args)
{
  map<string,double> gs = QCM::parameters();
  PyObject *lst = PyDict_New();
  for(auto& x : gs){
    PyDict_SetItem(lst, Py_BuildValue("s#", x.first.c_str(), x.first.length()), Py_BuildValue("d", x.second));
  }
  return lst;
}



//==============================================================================
const char* instance_parameters_help =
R"{(
returns the values of the parameters in a given instance
arguments:
1. label (optional) :  label of the model instance (default 0)
returns: a dictionary of string/real pairs
){";
//------------------------------------------------------------------------------
static PyObject* instance_parameters_python(PyObject *self, PyObject *args)
{
  int label=0;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to parameters (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  map<string,double> gs = QCM::instance_parameters(label);
  PyObject *lst = PyDict_New();
  for(auto& x : gs){
    PyDict_SetItem(lst, Py_BuildValue("s#", x.first.c_str(), x.first.length()), Py_BuildValue("d", x.second));
  }
  return lst;
}




//==============================================================================
const char* print_parameter_set_help =
R"{(
Print the parameter set to the screen
){";
//------------------------------------------------------------------------------
static PyObject* print_parameter_set_python(PyObject *self, PyObject *args)
{
  qcm_model->param_set->print(cout);
  return Py_BuildValue("");
}



//==============================================================================
const char* parameter_set_help =
R"{(
returns the content of the parameter set
  :return: a dictionary of str:(float, str, float). The three components are 
    (1) the value of the parameter, 
    (2) the name of its overlord (or None), 
    (3) the multiplier by which its value is obtained from that of the overlord.
){";
//------------------------------------------------------------------------------
static PyObject* parameter_set_python(PyObject *self, PyObject *args)
{
  PyObject *lst;
  int opt=0;

  try{
    if(qcm_model->param_set == nullptr)
      qcm_throw("The parameter set has not been defined yet");
    
    lst = PyDict_New();
    for(auto& x : qcm_model->param_set->param){
      string name;
      if(x.second->ref){
        name = x.second->ref->name;
        // if(x.second->ref->label) name += x.second->separator + to_string(x.second->ref->label);
      }
      PyObject* elem = PyTuple_New(3);
      PyTuple_SetItem(elem, 0, Py_BuildValue("d", x.second->value));
      if(x.second->ref){
        PyTuple_SetItem(elem, 1, Py_BuildValue("s#", name.c_str(), name.length()));
        PyTuple_SetItem(elem, 2, Py_BuildValue("d", x.second->multiplier));
      }
      else{
        PyTuple_SetItem(elem, 1, Py_None);
        PyTuple_SetItem(elem, 2, Py_None);
      }
      PyDict_SetItem(lst, Py_BuildValue("s#", x.first.c_str(), x.first.length()), Py_BuildValue("O", elem));
    }
  } catch(const string& s) {qcm_catch(s);}
  return lst;
}



//==============================================================================
const char* periodized_Green_function_help =
R"{(
computes the periodized Green function at a given frequency and wavevectors
arguments:
1. z : complex frequency
2. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* periodized_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to periodized_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'periodized_Green_function' should be of dimension 1 or 2");
    
    vector<vector3D<double>> kk;
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      kk.assign(1,k);
    }
    else{
      kk = many_vectors_from_Py(k_pyobj);
    }
    auto g = QCM::periodized_Green_function(z, kk, (bool)spin_down, label);
    
    size_t d = QCM::reduced_Green_function_dimension();
    
    npy_intp dims[3];
    dims[0] = g.size();
    dims[1] = dims[2] = d;
    
    out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    for(size_t j = 0; j<g.size(); j++){
      memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}


//==============================================================================
const char* band_Green_function_help =
R"{(
computes the periodized Green function in the band  basis at a given frequency and wavevectors
arguments:
1. z : complex frequency
2. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* band_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to band_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'band_Green_function' should be of dimension 1 or 2");
    
    vector<vector3D<double>> kk;
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      kk.assign(1,k);
    }
    else{
      kk = many_vectors_from_Py(k_pyobj);
    }
    auto g = QCM::band_Green_function(z, kk, (bool)spin_down, label);
    
    size_t d = QCM::reduced_Green_function_dimension();
    
    npy_intp dims[3];
    dims[0] = g.size();
    dims[1] = dims[2] = d;
    
    out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    for(size_t j = 0; j<g.size(); j++){
      memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}

//==============================================================================
const char* periodized_Green_function_element_help =
R"{(
computes a matrix element of the periodized Green function at a given frequency and wavevectors
arguments:
1. r : row index
2. c : column index
3. z : complex frequency
4. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
5. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
6. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* periodized_Green_function_element_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  int r, c;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "iiDO|ii", &r, &c, &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to periodized_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'periodized_Green_function' should be of dimension 1 or 2");
    
    vector<vector3D<double>> kk;
    kk = many_vectors_from_Py(k_pyobj);
    auto g = QCM::periodized_Green_function_element(r, c, z, kk, (bool)spin_down, label);
        
    npy_intp dims[1];
    dims[0] = g.size();
    
    out = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
    for(size_t j = 0; j<g.size(); j++){
      memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j, &g[j], sizeof(complex<double>));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}




//==============================================================================
const char* Potthoff_functional_help =
R"{(
computes the Potthoff functional for a given instance
arguments:
1. label (optional) :  label of the model instance (default 0)
returns: the value of the self-energy functional
){";
//------------------------------------------------------------------------------
static PyObject* Potthoff_functional_python(PyObject *self, PyObject *args)
{
  int label=0;
  double sef;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to Potthoff_functional (python)");
    sef = QCM::Potthoff_functional(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("d", sef);
}



//==============================================================================
const char* potential_energy_help =
R"{(
computes the potential_energy for a given instance
arguments:
1. label (optional) :  label of the model instance (default 0)
returns: the value of the potential energy
){";
//------------------------------------------------------------------------------
static PyObject* potential_energy_python(PyObject *self, PyObject *args)
{
  int label=0;
  double PE;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to potential_energy (python)");
    PE = QCM::potential_energy(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("d", PE);
}



//==============================================================================
const char* kinetic_energy_help =
R"{(
returns the kinetic_energy for a given instance
arguments:
1. label (optional) :  label of the model instance (default 0)
returns: the value of the kinetic energy
){";
//------------------------------------------------------------------------------
static PyObject* kinetic_energy_python(PyObject *self, PyObject *args)
{
  int label=0;
  double KE;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to kinetic_energy (python)");
    KE = QCM::kinetic_energy(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("d", KE);
}



//==============================================================================
const char* print_model_help =
R"{(
Prints a description of the model into a file
arguments:
1. file : name of the file
returns: void
){";
//------------------------------------------------------------------------------
static PyObject* print_model_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  char* s1 = nullptr;
  const char *kwlist[] = {"", "asy_operators", "asy_labels", "asy_orb", "asy_neighbors", "asy_working_basis",  NULL};

  try{
    if(!PyArg_ParseTuple(args, "s", &s1))
      qcm_throw("failed to read parameters in call to print_model (python)");
    QCM::print_model(string(s1));
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}



//==============================================================================
const char* print_options_help =
R"{(
Prints the list of global options and parameters on the screen
argument:
  1. int to_file : 0 -> prints to screen. 1 -> prints to latex. 2-> prints to RST
){";
//------------------------------------------------------------------------------
static PyObject* print_options_python(PyObject *self, PyObject *args)
{
  int to_file = 0;
  QCM::global_parameter_init();

  try{
    if(!PyArg_ParseTuple(args, "i", &to_file))
      qcm_throw("failed to read parameters in call to print_model (python)");
    print_options(to_file);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}



//==============================================================================
const char* projected_Green_function_help =
R"{(
computes the projected Green function at a given frequency, as used in CDMFT.
arguments:
1. z : a complex frequency
2. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
3. label (optional) :  label of the model instance (default 0)
returns: the projected Green function matrix (d x d), d being the dimension of the CPT Green function.
){";
//------------------------------------------------------------------------------
static PyObject* projected_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyObject *out;
  try{
    if(!PyArg_ParseTuple(args, "D|ii", &z, &spin_down, &label))
      qcm_throw("failed to read parameters in call to projected_Green_function (python)");
  
    auto g = QCM::projected_Green_function(z, (bool)spin_down, label);
    
    size_t d = QCM::Green_function_dimension();
    
    npy_intp dims[2];
    dims[0] = dims[1] = d;
    
    out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}



//==============================================================================
const char* reduced_Green_function_dimension_help =
R"{(
returns the dimension of the reduced Green function, i.e. a simple multiple of the
number n of lattice orbitals, depending on the mixing state: n, 2n or 4n.
){";
//------------------------------------------------------------------------------
static PyObject* reduced_Green_function_dimension_python(PyObject *self, PyObject *args)
{
  PyArg_ParseTuple(args, "");
  size_t d = QCM::reduced_Green_function_dimension();
  return Py_BuildValue("i", d);
}



//==============================================================================
const char* self_energy_help =
R"{(
computes the self-energy associated with the periodized Green function at a given frequency and wavevectors
arguments:
1. z : complex frequency
2. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* self_energy_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;

  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to self_energy (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 3 of 'periodized_Green_function' should be of dimension 1 or 2");
    
    vector<vector3D<double>> kk;
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      kk.assign(1,k);
    }
    else{
      kk = many_vectors_from_Py(k_pyobj);
    }
    auto g = QCM::self_energy(z, kk, (bool)spin_down, label);
    
    size_t d = QCM::reduced_Green_function_dimension();
    
    npy_intp dims[3];
    dims[0] = g.size();
    dims[1] = dims[2] = d;
    
    out = PyArray_SimpleNew(3, dims, NPY_COMPLEX128);
    for(size_t j = 0; j<g.size(); j++){
      memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out)+j*d*d, g[j].data(), g[j].size()*sizeof(complex<double>));
    }
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}



//==============================================================================
const char* set_global_parameter_help =
R"{(
sets the value of a global parameter. Several possibilities:
set_global_parameter(string opt)
sets the value of a boolean global parameter to 'True'
arguments:
opt : name of the parameter
returns : void
set_global_parameter(string opt, value)
sets the value of a boolean global parameter to 'value'
arguments:
opt : name of the parameter
value : value of the parameter, either int, real or string.
returns : void
){";
//------------------------------------------------------------------------------
static PyObject* set_global_parameter_python(PyObject *self, PyObject *args)
{
  char* S1 = nullptr;
  PyObject *obj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "s|O", &S1, &obj))
      qcm_throw("failed to read parameters in call to set_global_parameter (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  string name(S1);
  try{
    if(obj == nullptr){
      set_global_bool(name, true);
      cout << "global parameter " << name << " set to true" << endl;
    }
    else{
      if(PyLong_Check(obj)){
        size_t I = (int)PyLong_AsLong(obj);
        if(GP_bool.find(name) != GP_bool.end()){
          if(I==0){
            set_global_bool(name, false);
            cout << "global parameter " << name << " set to false" << endl;
          }
          else{
            set_global_bool(name, true);
            cout << "global parameter " << name << " set to true" << endl;
          }
        }
        else{
          set_global_int(name, I);
          cout << "global parameter " << name << " set to " << I << endl;
        }
      }
      else if(PyFloat_Check(obj)){
        double I = (double)PyFloat_AsDouble(obj);
        set_global_double(name, I);
        cout << "global parameter " << name << " set to " << I << endl;
      }
      else if(PyUnicode_Check(obj)){
        Py_ssize_t s = PyUnicode_GetLength(obj);
        const char* op_char = PyUnicode_AsUTF8(obj);
        set_global_char(name, op_char[0]);
        cout << "global parameter " << name << " set to " << op_char[0] << endl;
      }
      else qcm_throw("unknown type of global_parameter");
    }
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}




//==============================================================================
const char* get_global_parameter_help =
R"{(
gets the value of a global parameter.
returns : the value of the global parameter, either boolean, integer, float or string
){";
//------------------------------------------------------------------------------
static PyObject* get_global_parameter_python(PyObject *self, PyObject *args)
{
  char* S1 = nullptr;
  PyObject *obj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "s", &S1))
      qcm_throw("failed to read parameters in call to get_global_parameter (python)");
  } catch(const string& s) {qcm_catch(s);}
  
  string name(S1);
  char C[2];
  if(is_global_bool(name)) return Py_BuildValue("i", global_bool(name));
  else if(is_global_int(name)) return Py_BuildValue("i", global_int(name));
  else if(is_global_double(name)) return Py_BuildValue("d", global_double(name));
  else if(is_global_char(name)) {
    C[0] = global_char(name);
    return Py_BuildValue("s", C);
  }
  else qcm_throw(name+" is not the name of a global_parameter");
  return Py_BuildValue("");
}



//==============================================================================
const char* set_parameter_help =
R"{(
sets the value of a parameter within a parameter_set
arguments:
1. param : name of the parameter
2. value : its value
returns : void
){";
//------------------------------------------------------------------------------
static PyObject* set_parameter_python(PyObject *self, PyObject *args)
{
  char* s1 = nullptr;
  double v = 0.0;
  try{
    if(!PyArg_ParseTuple(args, "sd", &s1, &v))
      qcm_throw("failed to read parameters in call to set_parameter (python)");
    QCM::set_parameter(string(s1), v);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}



//==============================================================================
const char* spatial_dimension_help =
R"{(
returns the spatial dimension (0,1,2 or 3) of the model
){";
//------------------------------------------------------------------------------
static PyObject* spatial_dimension_python(PyObject *self, PyObject *args)
{
  PyArg_ParseTuple(args, "");
  int d = QCM::spatial_dimension();
  return Py_BuildValue("i", d);
}

//==============================================================================
const char* spectral_average_help =
R"{(
returns the contribution of a frequency to the average of an operator
arguments:
1. name : name of the operator
2. z : complex frequency
3. label (optional) :  label of the model instance (default 0)
returns: double
){";
//------------------------------------------------------------------------------
static PyObject* spectral_average_python(PyObject *self, PyObject *args)
{
  int label=0;
  char* s1 = nullptr;
  double ave;
  Complex z(0.0);
  try{
    if(!PyArg_ParseTuple(args, "sD|i", &s1, &z, &label)) 
      qcm_throw("failed to read parameters in call to spectral_average (python)");
    ave = QCM::spectral_average(string(s1), z, label);
  } catch(const string& s) {qcm_catch(s);}
  
  return Py_BuildValue("d", ave);
}



//==============================================================================
const char* set_basis_help =
R"{(
arguments:
1. the basis (a (Dx3) real matrix)
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* set_basis_python(PyObject *self, PyObject *args)
{
  PyArrayObject *basis_pyobj = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "O", &basis_pyobj))
      qcm_throw("failed to read parameters in call to lattice_model (python)");
  
    vector<double> basis = vectors_from_Py(basis_pyobj);
    QCM::set_basis(basis);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* interaction_operator_help =
R"{(
Defines an interaction operator of type Hubbard, Hund, Heisenberg or X, Y, Z
arguments:
1. name of the operator
kwargs
       2. 'link" : 3 component integer vector, (0,0,0)by default
       3. 'amplitude" : double
       4. 'orb1' : int. lattice orbital label of first index (1 by default)
       5. 'orb2' : int. lattice orbital label of second index (1 by default)
       6. 'type' : one of 'Hubbard', 'Heisenberg', 'Hund', 'X', 'Y', 'Z'
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* interaction_operator_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  double amplitude=1.0;
  int orb1=1;
  int orb2=1;
  char *name = nullptr;
  char *type = nullptr;
  PyArrayObject *link_pyobj = nullptr;
  
  const char *kwlist[] = {"", "link", "amplitude", "orb1", "orb2", "type",  NULL};
  try{
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "s|Odiis", const_cast<char **>(kwlist),
                                     &name,
                                     &link_pyobj,
                                     &amplitude,
                                     &orb1,
                                     &orb2,
                                     &type))
      qcm_throw("failed to read parameters in call to interaction_operator (python)");

    string the_type("Hubbard");
    if(type != nullptr) the_type = string(type);
  
    vector3D<int64_t> link(0,0,0);
    if(link_pyobj != nullptr and (PyObject*)link_pyobj != Py_None) link = intvector_from_Py(link_pyobj);
  
    QCM::interaction_operator(string(name), link, amplitude, orb1, orb2, the_type);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* hopping_operator_help =
R"{(
Defines a hopping term or, more generally, a one-body operator
arguments:
1. name of operator
2. link (3 component integer array)
3. amplitude (real number)
kwargs:
  4. 'orb1' : int. lattice orbital label of first index (1 by default)
  5. 'orb2' : int. lattice orbital label of first index (1 by default)
  6. 'tau' : int. specifies the tau Pauli matrix (0,1,2,3)
  7. 'sigma' : int. specifies the sigma Pauli matrix (0,1,2,3)
  
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* hopping_operator_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  char *name = nullptr;
  double amplitude = 1.0;
  int orb1=1;
  int orb2=1;
  int tau=1;
  int sigma=0;
  PyArrayObject *link_pyobj = nullptr;
  
  const char *kwlist[] = {"", "", "", "orb1", "orb2", "tau", "sigma",  NULL};
  try{
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "sOd|iiii", const_cast<char **>(kwlist),
                                     &name,
                                     &link_pyobj,
                                     &amplitude,
                                     &orb1,
                                     &orb2,
                                     &tau,
                                     &sigma))
      qcm_throw("failed to read parameters in call to hopping_operator (python)");

    vector3D<int64_t> link = intvector_from_Py(link_pyobj);

    QCM::hopping_operator(string(name), link, amplitude, orb1, orb2, tau, sigma);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* anomalous_operator_help =
R"{(
Defines an anomalous operator
arguments:
  1. name of operator
  2. link (3 component integer array)
  3. amplitude (complex number)
kwargs:
  4. 'orb1' : int. lattice orbital label of first index (1 by default)
  5. 'orb2' : int. lattice orbital label of second index (1 by default)
  6. 'type' : one of 'singlet' (default), 'dz', 'dy', 'dx'
  
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* anomalous_operator_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  char* name = nullptr;
  char* type = nullptr;
  complex<double> amplitude = 1.0;
  int orb1=1;
  int orb2=1;
  PyArrayObject *link_pyobj = nullptr;
  
  const char *kwlist[] = {"", "", "", "orb1", "orb2", "type",  NULL};
  try{
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "sOD|iis", const_cast<char **>(kwlist),
                                     &name,
                                     &link_pyobj,
                                     &amplitude,
                                     &orb1,
                                     &orb2,
                                     &type
                                     ))
      qcm_throw("failed to read parameters in call to anomalous_operator "+string(name)+" (python)");
  
    string the_type("singlet");
    if(type != nullptr) the_type = string(type);
    vector3D<int64_t> link = intvector_from_Py(link_pyobj);
    QCM::anomalous_operator(string(name), link, amplitude, orb1, orb2, the_type);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* explicit_operator_help =
R"{(
Defines an anomalous operator
arguments:
1. name of operator
2. array of 3-tuples (int vector, int vector, complex)
  
kwargs:
3. 'tau' : int. specifies the tau Pauli matrix (0,1,2,3)
4. 'sigma' : int. specifies the sigma Pauli matrix (0,1,2,3)
5. 'type' : one of 'one-body' [default], 'singlet', 'dz', 'dy', 'dx', 'Hubbard', 'Hund', 'Heisenberg', 'X', 'Y', 'Z'

returns: None
){";
//------------------------------------------------------------------------------
static PyObject* explicit_operator_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  char* name = nullptr;
  char* type = nullptr;
  int tau=1;
  int sigma=0;
  PyObject *elem_obj = nullptr;
  
  const char *kwlist[] = {"", "", "tau", "sigma", "type",  NULL};
  try{
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "sO|iis", const_cast<char **>(kwlist),
                                    &name,
                                    &elem_obj,
                                    &tau,
                                    &sigma,
                                    &type
                                    ))
      qcm_throw("failed to read parameters in call to anomalous_operator "+string(name)+" (python)");
    if(!PySequence_Check(elem_obj)) qcm_throw("Argument 2 passed to 'explicit_operator' is not a python list");

    string the_type("one-body");
    if(type != nullptr) the_type = string(type);
    size_t n = PySequence_Size(elem_obj);
    vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> elem(n);
    for(size_t i=0; i<elem.size(); i++){
      PyObject* PyObj2 = PySequence_GetItem(elem_obj,i);
      if(!PyTuple_Check(PyObj2)) qcm_throw("Element "+to_string(i)+" passed to explicit_operator is not a tuple");
      if(PyTuple_Size(PyObj2)!=3) qcm_throw("Element "+to_string(i)+" passed to explicit_operator is not a 3-tuple");
      vector3D<int64_t> pos1 =  intvector_from_Py((PyArrayObject*)PyTuple_GetItem(PyObj2,0));
      vector3D<int64_t> pos2 =  intvector_from_Py((PyArrayObject*)PyTuple_GetItem(PyObj2,1));
      Py_complex z = PyComplex_AsCComplex(PyTuple_GetItem(PyObj2,2));
      get<0>(elem[i]) = pos1;
      get<1>(elem[i]) = pos2;
      get<2>(elem[i]) = complex<double>(z.real, z.imag);
    }
    QCM::explicit_operator(string(name), the_type, elem, tau, sigma);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* density_wave_help =
R"{(
Defines a density wave
arguments:
1. name of operator
2. one of 'Z', 'X', 'cdw', 'singlet', 'dz', 'dy', 'dx'
3. Q vector (3-component real vector)
kwargs:
  4. link (3 component integer vector) for bond density waves
  5. amplitude (complex number)
  6. 'orb' : int. lattice orbital label (0 by default = all orbitals)
  7. 'phase' : real phase (times pi)
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* density_wave_python(PyObject *self, PyObject *args, PyObject *keywds)
{
  char* name = nullptr;
  char* type = nullptr;
  complex<double> amplitude = 1.0;
  int orb=0;
  double phase = 0.0;
  PyArrayObject *link_pyobj = nullptr;
  PyArrayObject *Q_pyobj = nullptr;
  
  const char* kwlist[] = {"", "", "", "link", "amplitude", "orb", "phase",  NULL};
  try{
    if(!PyArg_ParseTupleAndKeywords(args, keywds, "ssO|ODid", const_cast<char **>(kwlist),
                                     &name,
                                     &type,
                                     &Q_pyobj,
                                     &link_pyobj,
                                     &amplitude,
                                     &orb,
                                     &phase))
      qcm_throw("failed to read parameters in call to density_wave (python)");
  
    vector3D<int64_t> link(0,0,0);
    if(link_pyobj != nullptr and (PyObject*)link_pyobj != Py_None) link = intvector_from_Py(link_pyobj);
  
    QCM::density_wave(string(name), link, amplitude, orb, vector_from_Py(Q_pyobj), phase, string(type));
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}

//==============================================================================
const char* site_and_bond_profile_help =
R"{(
Computes the site and bond profiles in all clusters of the repeated unit
arguments:
None
returns: A pair of ndarrays
  site profile: the components are
  x y z n Sx Sy Sz psi.real psi.imag
  bond profile: the components are
  x1 y1 z1 x2 y2 z2 b0 bx by bz d0.real dx.real dy.real dz.real d0.imag dx.imag dy.imag dz.imag
){";

//------------------------------------------------------------------------------
static PyObject* site_and_bond_profile_python(PyObject *self, PyObject *args)
{
  int label=0;
  PyObject *out, *out2;
  pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> prof;
  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
    qcm_throw("failed to read parameters in call to site_and_bond_profile (python)");
    prof = QCM::site_and_bond_profile(label);

    npy_intp dims[2];
    dims[0] = prof.first.size();
    dims[1] = 9;

    out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    for(int i=0; i<dims[0]; i++)
      memcpy(PyArray_BYTES((PyArrayObject*) out)+i*9*sizeof(double), (prof.first[i]).data(), 9*sizeof(double));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
    
    dims[0] = prof.second.size();
    dims[1] = 11;
    out2 = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    for(int i=0; i<dims[0]; i++)
      memcpy(PyArray_BYTES((PyArrayObject*) out2)+i*11*sizeof(complex<double>), (prof.second[i]).data(), 11*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out2, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("OO", out, out2);
}

  
  

//==============================================================================
const char* V_matrix_help =
R"{(
computes the V_matrix at a given frequency and wavevectors
arguments:
1. z : complex frequency
2. k : single wavevector (ndarray(3)) 
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* V_matrix_python(PyObject *self, PyObject *args)
{
  int label=0;
  int spin_down=0;
  complex<double> z;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *out;
  try{
    if(!PyArg_ParseTuple(args, "DO|ii", &z, &k_pyobj, &spin_down, &label))
      qcm_throw("failed to read parameters in call to periodized_Green_function (python)");
  
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>1) qcm_throw("Argument 2 of 'V_matrix' should be of dimension 1");
    vector3D<double> k = vector_from_Py(k_pyobj);
    auto g = QCM::V_matrix(z, k, (bool)spin_down, label);
    size_t d = QCM::Green_function_dimension();
    npy_intp dims[2];
    dims[0] = d;
    dims[1] = d;
    out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy((complex<double>*)PyArray_DATA((PyArrayObject*) out), g.data(), g.size()*sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject*) out, NPY_ARRAY_OWNDATA);
  } catch(const string& s) {qcm_catch(s);}
  return out;
}



//==============================================================================
const char* Lehmann_Green_function_help =
R"{(
computes a Lehmann representation of the periodized Green function at a given frequency and wavevectors
arguments:
1. k : single wavevector (ndarray(3)) or array of wavevectors (ndarray(N,3))
2. lattice orbital index (starts at 1)
3. spin_down (optional): true is the spin down sector is to be computed (applies if mixing = 4)
4. label (optional) :  label of the model instance (default 0)
returns: a single (d,d) or an array (N,d,d) of complex-valued matrices. d is the reduced GF dimension.
){";
//------------------------------------------------------------------------------
static PyObject* Lehmann_Green_function_python(PyObject *self, PyObject *args)
{
  int label=0;
  int orb=1;
  int spin_down=0;
  PyArrayObject *k_pyobj = nullptr;
  PyObject *lst;

  try{
    if(!PyArg_ParseTuple(args, "O|iii", &k_pyobj, &orb, &spin_down, &label))
      qcm_throw("failed to read parameters in call to periodized_Green_function (python)");
  
    orb -= 1;
    int ndim = PyArray_NDIM(k_pyobj);
    if(ndim>2) qcm_throw("Argument 2 of 'Lehmann_Green_function' should be of dimension 1 or 2");

    vector<vector3D<double>> kk;
    if(ndim == 1){
      vector3D<double> k = vector_from_Py(k_pyobj);
      kk.assign(1,k);
    }
    else kk = many_vectors_from_Py(k_pyobj);
  
    lst = PyList_New(kk.size());
    auto g = QCM::Lehmann_Green_function(kk, orb, (bool)spin_down, label);

    for(size_t i=0; i<g.size(); i++){
      auto P = PyTuple_New(2);
      npy_intp dims[1];
      dims[0] = g[i].first.size();
      PyObject *poles = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
      PyObject *res = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
      memcpy((double*)PyArray_DATA((PyArrayObject*) poles), g[i].first.data(), g[i].first.size()*sizeof(double));
      memcpy((double*)PyArray_DATA((PyArrayObject*) res), g[i].second.data(), g[i].second.size()*sizeof(double));
      PyArray_ENABLEFLAGS((PyArrayObject*) poles, NPY_ARRAY_OWNDATA);
      PyArray_ENABLEFLAGS((PyArrayObject*) res, NPY_ARRAY_OWNDATA);
      PyTuple_SetItem(P, 0, poles);
      PyTuple_SetItem(P, 1, res);
      PyList_SET_ITEM(lst, i, P);
    }
  } catch(const string& s) {qcm_catch(s);}
  return lst;
}


//==============================================================================
const char* monopole_help =
R"{(
returns the charge of a node in a Weyl semi-metal
arguments: 
1. k : wavevector, position of the node
2. a : float, half-side of the cube surrounding the node 
3. nk : number of divisions along the side of the cube
4. orb : lattice orbital to compute the charge of
5. rec : boolean, true if subdivision is allowed
6. label : int, label of the model instance
returns: float
){";
//------------------------------------------------------------------------------
static PyObject* monopole_python(PyObject *self, PyObject *args)
{
  int nk = 0;
  int label = 0;
  int orb = 0;
  int rec = 0;
  double a = 0.0;
  vector3D<double> k;
  PyArrayObject *k_pyobj = nullptr;
  try{
    if(!PyArg_ParseTuple(args, "Odii|ii", &k_pyobj, &a, &nk, &orb, &rec, &label))
      qcm_throw("failed to read parameters in call to monopole (python)");
    k = vector_from_Py(k_pyobj);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("d", QCM::monopole(k, a, nk, orb, rec, label));
}



//==============================================================================
const char* switch_cluster_model_help =
R"{(
switches the cluster model
arguments:
1. name of new cluster model (string)
returns: None
){";
//------------------------------------------------------------------------------
static PyObject* switch_cluster_model_python(PyObject *self, PyObject *args)
{
  char* s1 = nullptr;
  
  try{
    if(!PyArg_ParseTuple(args, "s", &s1))
      qcm_throw("failed to read parameters in call to switch_cluster_model (python)");
  
    QCM::switch_cluster_model(string(s1));
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* complex_HS_help =
R"{(
arguments:
1. label : int, label of the model instance
returns: True if the model instance requires a complex Hilbert space, False otherwise
){";
//------------------------------------------------------------------------------
static PyObject* complex_HS_python(PyObject *self, PyObject *args)
{
  int label=0;
  int result=0;

  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to complex_HS (python)");
      result = (int)QCM::complex_HS(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("i", result);
}


//==============================================================================
const char* great_reset_help =
R"{(
Resets the whole system and allows to redefine the model
returns None
){";
//------------------------------------------------------------------------------
static PyObject* great_reset_python(PyObject *self, PyObject *args)
{
  QCM::great_reset();
  return Py_BuildValue("");
}

//==============================================================================
const char* erase_model_instance_help =
R"{(
Erases a model instance from the list
arguments:
1. label : int, label of the model instance
returns None
){";
//------------------------------------------------------------------------------
static PyObject* erase_model_instance_python(PyObject *self, PyObject *args)
{
  int label=0;

  try{
    if(!PyArg_ParseTuple(args, "|i", &label))
      qcm_throw("failed to read parameters in call to complex_HS (python)");
      QCM::erase_lattice_model_instance(label);
  } catch(const string& s) {qcm_catch(s);}
  return Py_BuildValue("");
}


//==============================================================================
const char* print_statistics_help =
R"{(
Prints run statistics on the screen
){";
//------------------------------------------------------------------------------
static PyObject* print_statistics_python(PyObject *self, PyObject *args)
{
  try{
    if(!PyArg_ParseTuple(args, ""))
      qcm_throw("failed to read parameters in call to print_statistics (python)");
  } catch(const string& s) {qcm_catch(s);}
  cout << "Number of recycled evaluations of the cluster Green function : " << STATS.n_GF_recycled << endl;
  cout << "Number of computed evaluations of the cluster Green function : " << STATS.n_GF_computed << endl;
  return Py_BuildValue("");
}

#endif