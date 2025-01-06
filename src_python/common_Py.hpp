#ifndef common_Py_h
#define common_Py_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <array>
#include <memory>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "QCM.hpp"
#include "float.h"
#include "lattice_model.hpp"
#include "numpy/ndarrayobject.h"
#include "parameter_set.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"
#include <string>

extern shared_ptr<lattice_model> qcm_model;

void check_signals();
string Py2string(PyObject *PyObj);
vector3D<int64_t> intvector_from_Py(PyArrayObject *k_pyobj);
vector3D<double> vector_from_Py(PyArrayObject *k_pyobj);
vector<vector3D<double>> many_vectors_from_Py(PyArrayObject *k_pyobj);
vector<vector3D<int64_t>> many_intvectors_from_Py(PyArrayObject *k_pyobj);
vector<int64_t> intvectors_from_Py(PyArrayObject *k_pyobj);
vector<int> intarray_from_Py(PyArrayObject *k_pyobj);
vector<double> vectors_from_Py(PyArrayObject *k_pyobj);
vector<complex<double>> complex_array1_from_Py(PyArrayObject *k_pyobj);
matrix<complex<double>> complex_array2_from_Py(PyArrayObject *k_pyobj);
vector<matrix<complex<double>>> complex_array3_from_Py(PyArrayObject *k_pyobj);
vector<string> strings_from_PyList(PyObject *lst);
vector<double> doubles_from_Py(PyObject *lst);

map<string, double> py_dict_to_map(PyObject *D);
vector<vector<int>> intmatrix_from_Py(PyArrayObject *k_pyobj);
vector<vector<double>> pos_from_Py(PyArrayObject *k_pyobj);

#endif
