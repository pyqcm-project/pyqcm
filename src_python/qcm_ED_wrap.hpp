#ifndef qcm_ED_wrap_h
#define qcm_ED_wrap_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "numpy/arrayobject.h"
#include <Python.h>
// #include "ndarrayobject.h"
#include "common_Py.hpp"
#include "float.h"
#include "model_instance.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"

extern map<string, shared_ptr<model>> models;
extern map<size_t, shared_ptr<model_instance_base>> model_instances;

static PyObject *qcm_ED_Error;

//==============================================================================
// Wrappers
//==============================================================================
const char *ED_complex_HS_help =
    R"(
return 1 if the Hilbert space is complex, 0 if it is real
arguments:
1. label of model_instance (optional, default=0)
)";
//------------------------------------------------------------------------------
static PyObject *ED_complex_HS_python(PyObject *self, PyObject *args) {
  int label = 0;
  int result = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to hopping_matrix (python)");
    result = (int)ED::complex_HS((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("i", result);
}
//==============================================================================
const char *density_matrix_help =
    R"(
computes the density_matrix from the ground state
arguments:
1. list of sites indices defining subsystem A (starting from 0)
2. label of model instance
returns:
the density matrix and the basis used
)";
//------------------------------------------------------------------------------
static PyObject *density_matrix_python(PyObject *self, PyObject *args) {
  int label = 0;
  PyArrayObject *sites;
  npy_intp dims[2];
  pair<matrix<complex<double>>, vector<uint64_t>> g;

  try {
    if (!PyArg_ParseTuple(args, "O|i", &sites, &label))
      qcm_ED_throw(
          "failed to read parameters in call to density_matrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  try {
    vector<int> sitesI = intarray_from_Py(sites);
    g = ED::density_matrix(sitesI, label);

    dims[0] = dims[1] = g.first.r;

  } catch (const string &s) {
    cerr << s << "(in density_matrix)" << endl;
    exit(1);
  }

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.first.data(),
         g.first.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);

  PyObject *out2 = PyArray_SimpleNew(1, dims, NPY_UINT64);
  memcpy(PyArray_DATA((PyArrayObject *)out2), g.second.data(),
         g.second.size() * sizeof(uint64_t));
  PyArray_ENABLEFLAGS((PyArrayObject *)out2, NPY_ARRAY_OWNDATA);

  PyObject *out3 = PyTuple_New(2);
  PyTuple_SetItem(out3, 0, out);
  PyTuple_SetItem(out3, 1, out2);

  return out3;
}
//==============================================================================
const char *Green_function_average_help =
    R"(
arguments:
1. label of model_instance (optional, default=0)
returns:
the average of c^\dagger_i c_j (matrix)
)";
//------------------------------------------------------------------------------
static PyObject *Green_function_average_python(PyObject *self, PyObject *args) {
  int label = 0;
  int spin_down = 0;
  try {
    if (!PyArg_ParseTuple(args, "|ii", &label, &spin_down))
      qcm_ED_throw("failed to read parameters in call to "
                   "Green_function_average (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  matrix<complex<double>> g;
  try {
    g = ED::Green_function_average((bool)spin_down, (size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  dims[0] = dims[1] = g.r;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}
//==============================================================================
const char *Green_function_density_help =
    R"(
arguments:
1. label of model_instance (optional, default=0)
returns:
the density of the cluster, computed from the trace of the Green function average
)";
//------------------------------------------------------------------------------
static PyObject *Green_function_density_python(PyObject *self, PyObject *args) {
  int label = 0;
  double dens = 0.0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw("failed to read parameters in call to "
                   "Green_function_average (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  try {
    dens = ED::Green_function_density((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  return Py_BuildValue("d", dens);
}
//==============================================================================
const char *Green_function_dimensionC_help =
    R"(
arguments:
1. label of model_instance (optional, default=0)
returns:
the dimension of the Green function matrix (int)
)";
//------------------------------------------------------------------------------
static PyObject *Green_function_dimensionC_python(PyObject *self,
                                                  PyObject *args) {
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw("failed to read parameters in call to "
                   "Green_function_dimension (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  size_t d;
  try {
    d = ED::Green_function_dimension((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("i", d);
}
//==============================================================================
const char *Green_function_solveC_help =
    R"(

arguments:
1. label of model_instance (optional, default=0)
returns:
)";
//------------------------------------------------------------------------------
static PyObject *Green_function_solveC_python(PyObject *self, PyObject *args) {
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to Green_function_solve (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  try {
    ED::Green_function_solve((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *Green_function_help =
    R"(
computes the Green function matrix at a given complex frequency
arguments:
1. frequency (complex)
2. True for the spin down sector (optional)
3. label of model_instance (optional, default=0)
returns:
Green function matrix
)";
//------------------------------------------------------------------------------
static PyObject *Green_function_python(PyObject *self, PyObject *args) {
  int label = 0;
  int spin_down = 0;
  int blocks = 0;
  complex<double> z;

  try {
    if (!PyArg_ParseTuple(args, "D|iii", &z, &spin_down, &label, &blocks))
      qcm_ED_throw(
          "failed to read parameters in call to Green_function (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  vector<complex<double>> g;
  size_t d;
  try {
    g = ED::Green_function(z, (bool)spin_down, (size_t)label, (bool)blocks).v;
    d = ED::Green_function_dimension((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  dims[0] = dims[1] = d;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}

//==============================================================================
const char *cluster_averages_help =
    R"(
computes the ground state averages of the operators defined in the model
arguments:
1. label of model_instance (optional, default=0)
returns:
A dictionnary string : tuple(average, variance)
)";
//------------------------------------------------------------------------------
static PyObject *cluster_averages_python(PyObject *self, PyObject *args) {
  vector<tuple<string, double, double>> ave;
  int label = 0;

  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to cluster_averages (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  try {
    ave = ED::cluster_averages((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  PyObject *lst = PyDict_New();
  for (auto &x : ave) {
    PyDict_SetItem(lst, Py_BuildValue("s", get<0>(x).c_str()),
                   Py_BuildValue("dd", get<1>(x), get<2>(x)));
  }
  return lst;
}
//==============================================================================
const char *ground_state_solve_help =
    R"(
computes the ground state of the model
arguments:
1. label of model_instance (optional, default=0)
returns:
the ground state energy and the ground state Hilbert space sector
)";
//------------------------------------------------------------------------------
static PyObject *ground_state_solve_python(PyObject *self, PyObject *args) {
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to ground_state_solve (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  pair<double, string> result;
  try {
    result = ED::ground_state_solve((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  return Py_BuildValue("ds#", result.first, result.second.c_str(),
                       result.second.size());
}
//==============================================================================
const char *hopping_matrix_help =
    R"(
Computes the hopping matrix of the model
arguments:
1. True for the spin down sector (optional)
2. label of model_instance (optional, default=0)
returns:
The hopping matrix
)";
//------------------------------------------------------------------------------
static PyObject *hopping_matrix_python(PyObject *self, PyObject *args) {
  int label = 0;
  int spin_down = 0;
  int full = 0;
  int diag = 0;

  try {
    if (!PyArg_ParseTuple(args, "|iiii", &spin_down, &diag, &label, &full))
      qcm_ED_throw(
          "failed to read parameters in call to hopping_matrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  size_t d;
  vector<complex<double>> g;
  if (full) {
    g = ED::hopping_matrix_full((bool)spin_down, (bool)diag, (size_t)label).v;
    d = (size_t)sqrt(g.size());
  } else {
    g = ED::hopping_matrix((bool)spin_down, (size_t)label).v;
    try {
      d = ED::Green_function_dimension((size_t)label);
    } catch (const string &s) {
      qcm_ED_catch(s);
    }
  }

  npy_intp dims[2];
  dims[0] = dims[1] = d;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}
//==============================================================================
const char *interactions_help =
    R"(
Computes the hopping matrix of the model
arguments:
1. label of model_instance (optional, default=0)
returns:
The list of density-density interactions
)";
//------------------------------------------------------------------------------
static PyObject *interactions_python(PyObject *self, PyObject *args) {
  int label = 0;

  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to interactions (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  auto E = ED::interactions((size_t)label);
  PyObject *lst = PyList_New(E.size());
  for (size_t i = 0; i < E.size(); i++) {
    PyObject *elem = PyTuple_New(3);
    PyTuple_SetItem(elem, 0, Py_BuildValue("i", get<0>(E[i])));
    PyTuple_SetItem(elem, 1, Py_BuildValue("i", get<1>(E[i])));
    PyTuple_SetItem(elem, 2, Py_BuildValue("d", get<2>(E[i])));
    PyList_SET_ITEM(lst, i, elem);
  }

  return Py_BuildValue("O", lst);
}
//==============================================================================
const char *hybridization_functionC_help =
    R"(
Computes the hybridization function (for models with baths)
arguments:
1. A complex frequency
2. True for the spin down sector (optional)
3. label of model_instance (optional, default=0)
returns:
the hybridization matrix
)";
//------------------------------------------------------------------------------
static PyObject *hybridization_functionC_python(PyObject *self,
                                                PyObject *args) {
  int label = 0;
  int spin_down = 0;
  complex<double> z;

  try {
    if (!PyArg_ParseTuple(args, "D|ii", &z, &spin_down, &label))
      qcm_ED_throw(
          "failed to read parameters in call to hopping_matrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  vector<complex<double>> g =
      ED::hybridization_function(z, (bool)spin_down, (size_t)label).v;
  size_t d;
  try {
    d = ED::Green_function_dimension((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  dims[0] = dims[1] = d;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}
//==============================================================================
const char *matrix_elements_help =
    R"(
returns the list of matrix elements that defines an operator
arguments:
1. name of the model
2. name of the operator
returns:
a list of tuples (int, int, complex)

)";
//------------------------------------------------------------------------------
static PyObject *matrix_elements_python(PyObject *self, PyObject *args) {
  char *S1 = nullptr;
  char *S2 = nullptr;
  try {
    if (!PyArg_ParseTuple(args, "ss", &S1, &S2))
      qcm_ED_throw(
          "failed to read parameters in call to hopping_matrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  auto ET = ED::matrix_elements(string(S1), string(S2));
  auto E = ET.second;
  PyObject *lst = PyTuple_New(E.size());
  for (size_t i = 0; i < E.size(); i++) {
    PyObject *elem = PyTuple_New(3);
    PyTuple_SetItem(elem, 0, Py_BuildValue("i", E[i].r));
    PyTuple_SetItem(elem, 1, Py_BuildValue("i", E[i].c));
    PyTuple_SetItem(elem, 2, Py_BuildValue("D", &E[i].v));
    PyTuple_SET_ITEM(lst, i, elem);
  }
  return Py_BuildValue("sO", ET.first.c_str(), lst);
}
//==============================================================================
const char *mixingC_help =
    R"(
return the mixing state of the model
arguments:
1. label of model_instance (optional, default=0)
returns:
an integer code for the mixing. 0 : no mixing, 1 : anomalous, 2 : spin-flip, 3 : anomalous and spin-flip
)";
//------------------------------------------------------------------------------
static PyObject *mixingC_python(PyObject *self, PyObject *args) {
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to hopping_matrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  int result = ED::mixing((size_t)label);

  return Py_BuildValue("i", result);
}
//==============================================================================
const char *model_sizeC_help =
    R"(
arguments:
The name of the model
returns:
a pair of integers: the number of cluster sites and the number of bath sites
)";
//------------------------------------------------------------------------------
static PyObject *model_sizeC_python(PyObject *self, PyObject *args) {
  char *S1 = nullptr;
  try {
    if (!PyArg_ParseTuple(args, "s", &S1))
      qcm_ED_throw("failed to read parameters in call to model_size (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  tuple<int, int, int> d;
  try {
    d = ED::model_size(string(S1));
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("ii", get<0>(d), get<1>(d));
}
//==============================================================================
const char *new_model_help =
    R"(
Initiates a new model (no operators yet)
arguments:
1. name to be given to the model
2. number of cluster sites
3. number of bath sites
4. symmetry generators (2D array of ints)
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *new_model_python(PyObject *self, PyObject *args) {

  char *S1 = nullptr;
  int n_sites = 0;
  int n_bath = 0;
  vector<vector<int>> gen;
  int bath_irrep = 0;
  PyArrayObject *gen_pyobj = nullptr;
  try {
    if (!PyArg_ParseTuple(args, "sii|Oi", &S1, &n_sites, &n_bath, &gen_pyobj,
                          &bath_irrep))
      qcm_ED_throw("failed to read parameters in call to new_model (python)");
    if (gen_pyobj != nullptr)
      gen = intmatrix_from_Py(gen_pyobj);
    int n_orb = n_sites + n_bath;
    for (int j = 0; j < gen.size(); j++) {
      if (gen[j].size() != n_orb)
        qcm_ED_throw("generator " + to_string(j + 1) + " should have " +
                     to_string(n_orb) + " elements");
      for (int i = 0; i < n_sites; i++)
        gen[j][i] -= 1;
      if (bath_irrep == false)
        for (int i = n_sites; i < n_orb; i++)
          gen[j][i] -= 1;
    }

    ED::new_model(string(S1), n_sites, n_bath, gen, bath_irrep);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}
//==============================================================================
const char *new_model_instanceC_help =
    R"(
Initiates a new instance of the model
arguments:
1. name of the model (cluster)
2. values of the operators (dict of names:values)
3. target Hilbert space sectors (string)
4. label of model_instance (optional, default=0)
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *new_model_instanceC_python(PyObject *self, PyObject *args) {
  char *name = nullptr;
  char *sec = nullptr;
  int label = 0;
  PyObject *val;

  try {
    if (!PyArg_ParseTuple(args, "sOs|i", &name, &val, &sec, &label))
      qcm_ED_throw(
          "failed to read parameters in call to new_model_instance (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  map<string, double> param;
  try {
    param = py_dict_to_map(val);
  } catch (const string &s) {
    cerr << s << "(in new_model_instance)" << endl;
    exit(1);
  }

  try {
    ED::new_model_instance(string(name), param, string(sec), (size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  return Py_BuildValue("");
}
//==============================================================================
const char *new_operator_help =
    R"(
creates a new operator from its matrix elements
arguments:
1. name of the cluster to which the operator belong
2. name of the operator
3. type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z', 'general_interaction')
4. array of matrix elements (real)
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *new_operator_python(PyObject *self, PyObject *args) {
  char *name = nullptr;
  char *op = nullptr;
  char *type = nullptr;

  vector<matrix_element<double>> elem;
  PyObject *elem_pyobj;

  try {
    if (!PyArg_ParseTuple(args, "sssO", &name, &op, &type, &elem_pyobj))
      qcm_ED_throw(
          "failed to read parameters in call to new_operator (python)");

    double fac = 1.0;
    bool check_upper = true;
    if (strcmp(type, "anomalous") == 0)
      fac = 0.5; // correction for anomalous operators
    if (strcmp(type, "general_interaction") == 0)
      check_upper = false; // correction for anomalous operators

    if (PyArray_Check(elem_pyobj)) {
      size_t nelem = PyArray_DIMS((PyArrayObject *)elem_pyobj)[0];
      elem.resize(nelem);
      memcpy(elem.data(), PyArray_DATA((PyArrayObject *)elem_pyobj),
             nelem * PyArray_STRIDES((PyArrayObject *)elem_pyobj)[0]);
    } else if (PySequence_Check(elem_pyobj)) {
      size_t n = PySequence_Size(elem_pyobj);
      elem.assign(n, matrix_element<double>());
      for (int i = 0; i < n; i++) {
        PyObject *pkey = PySequence_GetItem(elem_pyobj, i);
        if (PyTuple_Size(pkey) == 3) {
          elem[i].r = PyLong_AsLong(PyTuple_GetItem(pkey, 0));
          elem[i].c = PyLong_AsLong(PyTuple_GetItem(pkey, 1));
          if ((elem[i].r > elem[i].c) and check_upper)
            qcm_ED_throw("the first index of element " + to_string<size_t>(i) +
                         " of argument 4 of 'new_operator' cannot be bigger "
                         "than the second index");
          if (elem[i].r == 0 or elem[i].c == 0)
            qcm_ED_throw("indices in matrix elements of operators are labelled "
                         "starting at 1, not at 0.");
          elem[i].r--;
          elem[i].c--;
          elem[i].v = fac * PyFloat_AsDouble(PyTuple_GetItem(pkey, 2));
        } else {
          qcm_ED_throw("element " + to_string<size_t>(i) +
                       " of argument 4 of 'new_operator' should be a 3-tuple");
        }
      }
    } else
      qcm_ED_throw("argument 4 of new_operator() must be a list or an array");

    ED::new_operator(string(name), string(op), string(type), elem);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}
//==============================================================================
const char *new_operator_complex_help =
    R"(
creates a new operator from its matrix elements (complex matrix elements)
arguments:
1. name of the cluster to which the operator belong
2. name of the operator
3. type of operator ('one-body', 'anomalous', 'interaction', 'Hund', 'Heisenberg', 'X', 'Y', 'Z', 'general_interaction')
4. array of matrix elements (complex)
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *new_operator_complex_python(PyObject *self, PyObject *args) {
  char *name = nullptr;
  char *op = nullptr;
  char *type = nullptr;

  vector<matrix_element<complex<double>>> elem;
  PyObject *elem_pyobj;

  try {
    if (!PyArg_ParseTuple(args, "sssO", &name, &op, &type, &elem_pyobj))
      qcm_ED_throw(
          "failed to read parameters in call to new_operator (python)");

    double fac = 1.0;
    if (strcmp(type, "anomalous") == 0)
      fac = 0.5; // correction for anomalous operators

    if (PyArray_Check(elem_pyobj)) {
      size_t nelem = PyArray_DIMS((PyArrayObject *)elem_pyobj)[0];
      elem.resize(nelem);
      memcpy(elem.data(), PyArray_DATA((PyArrayObject *)elem_pyobj),
             nelem * PyArray_STRIDES((PyArrayObject *)elem_pyobj)[0]);
    } else if (PySequence_Check(elem_pyobj)) {
      size_t n = PySequence_Size(elem_pyobj);
      elem.assign(n, matrix_element<complex<double>>());
      for (int i = 0; i < n; i++) {
        PyObject *pkey = PySequence_GetItem(elem_pyobj, i);
        if (PyTuple_Size(pkey) == 3) {
          elem[i].r = PyLong_AsLong(PyTuple_GetItem(pkey, 0));
          elem[i].c = PyLong_AsLong(PyTuple_GetItem(pkey, 1));
          if (elem[i].r > elem[i].c)
            qcm_ED_throw("the first index of element " + to_string<size_t>(i) +
                         " of argument 4 of 'new_operator' cannot be bigger "
                         "than the second index");
          if (elem[i].r == 0 or elem[i].c == 0)
            qcm_ED_throw("indices in matrix elements of operators are labelled "
                         "starting at 1, not at 0.");
          elem[i].r--;
          elem[i].c--;
          Py_complex z = PyComplex_AsCComplex(PyTuple_GetItem(pkey, 2));
          elem[i].v = {fac * z.real, fac * z.imag};
        } else {
          qcm_ED_throw("element " + to_string<size_t>(i) +
                       " of argument 4 of 'new_operator' should be a 3-tuple");
        }
      }
    } else
      qcm_ED_throw("argument 4 of new_operator() must be a list or an array");

    ED::new_operator(string(name), string(op), string(type), elem);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *parametersC_help =
    R"(
returns a dict of the parameters
arguments:
1. label of model_instance (optional, default=0)
returns: a dict of the parameters, the model name
)";
//------------------------------------------------------------------------------
static PyObject *parametersC_python(PyObject *self, PyObject *args) {
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw("failed to read parameters in call to parameters (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  string model_name;
  map<string, double> M;
  try {
    if (model_instances.find(label) == model_instances.end())
      qcm_ED_throw("The label " + to_string(label) + " is out of range.");
    M = model_instances.at(label)->value;
    model_name = model_instances.at(label)->the_model->name;
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  PyObject *lst = PyDict_New();
  for (auto &x : M) {
    PyDict_SetItem(lst, Py_BuildValue("s#", x.first.c_str(), x.first.length()),
                   Py_BuildValue("d", x.second));
  }
  return Py_BuildValue("Os", lst, model_name.c_str());
}

//==============================================================================
const char *print_models_help =
    R"(
prints the model description to the screen
arguments: None
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *print_models_python(PyObject *self, PyObject *args) {

  try {
    ED::print_models(cout);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *print_graph_help =
    R"(
prints a graphiz file for the model specified
arguments: 
1. name of the model
2. vector of positions
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *print_graph_python(PyObject *self, PyObject *args) {
  char *name = nullptr;
  PyArrayObject *pos_pyobj = nullptr;

  try {
    if (!PyArg_ParseTuple(args, "sO", &name, &pos_pyobj))
      qcm_ED_throw("failed to read parameters in call to print_graph (python)");
    vector<vector<double>> pos = pos_from_Py(pos_pyobj);
    auto model_name = string(name);
    if (models.find(model_name) == models.end())
      qcm_ED_throw("model " + model_name + " does not exist!");
    models[model_name]->print_graph(pos);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *self_energyC_help =
    R"(
computes the self-energy matrix at a given complex frequency
arguments:
1. frequency (complex)
2. True for the spin down sector (optional)
3. label of model_instance (optional, default=0)
returns:
self-energy matrix
)";
//------------------------------------------------------------------------------
static PyObject *self_energyC_python(PyObject *self, PyObject *args) {
  int label = 0;
  int spin_down = 0;
  complex<double> z;

  try {
    if (!PyArg_ParseTuple(args, "D|ii", &z, &spin_down, &label))
      qcm_ED_throw("failed to read parameters in call to self_energy (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  vector<complex<double>> g =
      ED::self_energy(z, (bool)spin_down, (size_t)label).v;
  size_t d;
  try {
    d = ED::Green_function_dimension((size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  dims[0] = dims[1] = d;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}

//==============================================================================
const char *set_global_parameterC_help =
    R"(
sets the value of a global parameter
arguments:
1. name of the parameter
2. value (leave out if it is a boolean parameter)
returns: None
)";
//------------------------------------------------------------------------------
static PyObject *set_global_parameterC_python(PyObject *self, PyObject *args) {
  char *S1 = nullptr;
  PyObject *obj = nullptr;

  try {
    if (!PyArg_ParseTuple(args, "s|O", &S1, &obj))
      qcm_ED_throw(
          "failed to read parameters in call to set_global_parameter (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  string name(S1);
  try {
    if (obj == nullptr) {
      set_global_bool(name, true);
      cout << "global parameter " << name << " set to true" << endl;
    } else {
      if (PyLong_Check(obj)) {
        size_t I = (int)PyLong_AsLong(obj);
        set_global_int(name, I);
        cout << "global parameter " << name << " set to " << I << endl;
      } else if (PyFloat_Check(obj)) {
        double I = (double)PyFloat_AsDouble(obj);
        set_global_double(name, I);
        cout << "global parameter " << name << " set to " << I << endl;
      } else
        qcm_ED_throw("unknown type of global_parameter");
    }
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *susceptibility_poles_help =
    R"(
computes the dynamic susceptibility of an operator
arguments:
1. name of the operator
2. label of model_instance (optional, default=0)
returns:
array of pairs (residue, pole)
)";
//------------------------------------------------------------------------------
static PyObject *susceptibility_poles_python(PyObject *self, PyObject *args) {
  int label = 0;
  char *op = nullptr;

  try {
    if (!PyArg_ParseTuple(args, "s|i", &op, &label))
      qcm_ED_throw(
          "failed to read parameters in call to susceptibility_poles (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  vector<pair<double, double>> g;
  try {
    g = ED::susceptibility_poles(string(op), (size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  dims[0] = g.size();
  dims[1] = 2;

  PyObject *out = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(pair<double, double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}

//==============================================================================
const char *susceptibility_help =
    R"(
computes the dynamic susceptibility of an operator
arguments:
1. name of the operator
2. array of complex frequencies
3. label of model_instance (optional, default=0)
returns:
array of complex susceptibilities
)";
//------------------------------------------------------------------------------
static PyObject *susceptibility_python(PyObject *self, PyObject *args) {
  int label = 0;
  char *op = nullptr;
  PyArrayObject *w_pyobj = nullptr;

  try {
    if (!PyArg_ParseTuple(args, "sO|i", &op, &w_pyobj, &label))
      qcm_ED_throw(
          "failed to read parameters in call to susceptibility (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  size_t nw = PyArray_DIMS(w_pyobj)[0];
  vector<complex<double>> w(nw);
  memcpy(w.data(), PyArray_DATA((PyArrayObject *)w_pyobj),
         w.size() * sizeof(complex<double>));

  vector<complex<double>> g;
  try {
    g = ED::susceptibility(string(op), w, (size_t)label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[1];
  dims[0] = nw;

  PyObject *out = PyArray_SimpleNew(1, dims, NPY_COMPLEX128);
  memcpy(PyArray_DATA((PyArrayObject *)out), g.data(),
         g.size() * sizeof(complex<double>));
  PyArray_ENABLEFLAGS((PyArrayObject *)out, NPY_ARRAY_OWNDATA);
  return out;
}

//==============================================================================
const char *print_wavefunction_help =
    R"{(
Prints the ground state wavefunction(s) on the screen 
argument:
1. label of model_instance (optional, default=0)
){";
//------------------------------------------------------------------------------
static PyObject *print_wavefunction_python(PyObject *self, PyObject *args) {
  int lab = 0;
  string out;
  try {
    if (!PyArg_ParseTuple(args, "|i", &lab))
      qcm_ED_throw("failed to read parameters in call to print_model (python)");
    out = ED::print_wavefunction(lab);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("z#", out.c_str(), out.length());
}

//==============================================================================
const char *qmatrix_help =
    R"{(
Returns the Lehmann representation of the Green function
returns a tuple:
  1. the array of M real eigenvalues, M being the number of poles in the representation
  2. a rectangular (L x M) matrix (real of complex), L being the dimension of the Green function
){";
//------------------------------------------------------------------------------
static PyObject *qmatrix_python(PyObject *self, PyObject *args) {
  int label = 0;

  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw("failed to read parameters in call to qmatrix (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  PyObject *out1, *out2;
  try {
    auto Q = ED::qmatrix(false, label);
    dims[0] = Q.first.size();
    out1 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject *)out1), Q.first.data(),
           Q.first.size() * sizeof(double));
    PyArray_ENABLEFLAGS((PyArrayObject *)out1, NPY_ARRAY_OWNDATA);
    dims[1] = Q.second.size() / Q.first.size();
    dims[0] = Q.first.size();
    out2 = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject *)out2), Q.second.data(),
           Q.second.size() * sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject *)out2, NPY_ARRAY_OWNDATA);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("OO", out1, out2);
}

//==============================================================================
const char *hybridization_help =
    R"{(
Returns the Lehmann representation of the hybridization function
returns a tuple:
  1. the array of M real eigenvalues, M being the number of poles in the representation
  2. a rectangular (L x M) matrix (real of complex), L being the dimension of the Green function
){";
//------------------------------------------------------------------------------
static PyObject *hybridization_python(PyObject *self, PyObject *args) {
  int label = 0;

  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to hybridization (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  npy_intp dims[2];
  PyObject *out1, *out2;
  try {
    auto Q = ED::hybridization(false, label);
    dims[0] = Q.first.size();
    out1 = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    memcpy(PyArray_DATA((PyArrayObject *)out1), Q.first.data(),
           Q.first.size() * sizeof(double));
    PyArray_ENABLEFLAGS((PyArrayObject *)out1, NPY_ARRAY_OWNDATA);
    dims[0] = Q.second.size() / Q.first.size();
    dims[1] = Q.first.size();
    out2 = PyArray_SimpleNew(2, dims, NPY_COMPLEX128);
    memcpy(PyArray_DATA((PyArrayObject *)out2), Q.second.data(),
           Q.second.size() * sizeof(complex<double>));
    PyArray_ENABLEFLAGS((PyArrayObject *)out2, NPY_ARRAY_OWNDATA);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("OO", out1, out2);
}

//==============================================================================
const char *write_instance_to_file_help =
    R"{(
Writes the solved model instance to a text file
argument:
    1. name of the file
    2. The instance label (default = 0)
returns None
){";
//------------------------------------------------------------------------------
static PyObject *write_instance_to_file_python(PyObject *self, PyObject *args) {
  char *op = nullptr;
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "s|i", &op, &label))
      qcm_ED_throw("failed to read parameters in call to "
                   "qcm_ED.write_instance_to_file()");
    ofstream fout(string(op).c_str());
    if (!fout.good())
      qcm_ED_throw("failed to open file " + string(op));
    ED::write_instance(fout, label);
    fout.close();
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}
//==============================================================================
const char *write_instance_help =
    R"{(
Writes the solved model instance to a string
argument:
    1. The instance label (default = 0)
returns None
){";
//------------------------------------------------------------------------------
static PyObject *write_instance_python(PyObject *self, PyObject *args) {
  int label = 0;
  ostringstream fout;
  try {
    if (!PyArg_ParseTuple(args, "|i", &label))
      qcm_ED_throw(
          "failed to read parameters in call to qcm_ED.write_instance()");
    ED::write_instance(fout, label);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("s#", fout.str().c_str(), fout.str().length());
}
//==============================================================================
const char *read_instance_help =
    R"{(
Reads the solved model instance from a text file
argument:
    1. name of the file
    2. The instance label (default=0)
returns None
){";
//------------------------------------------------------------------------------
static PyObject *read_instance_python(PyObject *self, PyObject *args) {
  char *op = nullptr;
  int label = 0;
  try {
    if (!PyArg_ParseTuple(args, "s|i", &op, &label))
      qcm_ED_throw(
          "failed to read parameters in call to read_instance (python)");
    string S(op);
    if (S.length() < 65) {
      ifstream fin(string(op).c_str());
      if (!fin.good())
        qcm_ED_throw("failed to open file " + string(op));
      ED::read_instance(fin, label);
    } else {
      istringstream fin(S);
      ED::read_instance(fin, label);
    }
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("");
}

//==============================================================================
const char *fidelity_help =
    R"{(
Reads the solved model instance from a text file
argument:
    1. name of the file
    2. The instance label (default=0)
returns None
){";
//------------------------------------------------------------------------------
static PyObject *fidelity_python(PyObject *self, PyObject *args) {
  int label1 = 0;
  int label2 = 0;
  double fid;
  try {
    if (!PyArg_ParseTuple(args, "ii", &label1, &label2))
      qcm_ED_throw("failed to read parameters in call to fidelity (*python*)");
    fid = ED::fidelity(label1, label2);
  } catch (const string &s) {
    qcm_ED_catch(s);
  }
  return Py_BuildValue("d", fid);
}

#endif
