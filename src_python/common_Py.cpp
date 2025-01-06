/**
 This files defines the Python wrappers and the python module
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL QCM_ARRAY_API
#define NO_IMPORT_ARRAY

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

//==============================================================================
// python utilities

void check_signals() {
  if (PyErr_CheckSignals() == -1)
    throw;
}
//------------------------------------------------------------------------------
string Py2string(PyObject *PyObj) {
  Py_ssize_t s = PyUnicode_GetLength(PyObj);
  const char *c = PyUnicode_AsUTF8(PyObj);
  return string(c, s);
}
//------------------------------------------------------------------------------
vector3D<int64_t> intvector_from_Py(PyArrayObject *k_pyobj) {
  vector3D<int64_t> k;
  if (PyArray_Check(k_pyobj)) {
    auto dims = PyArray_DIMS(k_pyobj);
    size_t nc = dims[0];
    if (nc != 3)
      qcm_throw("The supplied wavevector must have 3 components!");
    k.x = *(int *)PyArray_GETPTR1(k_pyobj, 0);
    k.y = *(int *)PyArray_GETPTR1(k_pyobj, 1);
    k.z = *(int *)PyArray_GETPTR1(k_pyobj, 2);
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    if (PySequence_Size(PyObj) != 3)
      qcm_throw("a position must have 3 components");
    PyObject *pkey;
    pkey = PySequence_GetItem(PyObj, 0);
    if (!PyLong_Check(pkey))
      qcm_throw("element no 1 of list is not an integer");
    k.x = PyLong_AsLong(pkey);
    pkey = PySequence_GetItem(PyObj, 1);
    if (!PyLong_Check(pkey))
      qcm_throw("element no 2 of list is not an integer");
    k.y = PyLong_AsLong(pkey);
    pkey = PySequence_GetItem(PyObj, 2);
    if (!PyLong_Check(pkey))
      qcm_throw("element no 3 of list is not an integer");
    k.z = PyLong_AsLong(pkey);
  } else
    qcm_throw("object is not a sequence");
  return k;
}
//------------------------------------------------------------------------------
vector3D<double> vector_from_Py(PyArrayObject *k_pyobj) {
  vector3D<double> k;
  if (PyArray_Check(k_pyobj)) {
    auto dims = PyArray_DIMS(k_pyobj);
    size_t nc = dims[0];
    if (nc != 3)
      qcm_throw("The supplied wavevector must have 3 components!");
    k.x = *(double *)PyArray_GETPTR1(k_pyobj, 0);
    k.y = *(double *)PyArray_GETPTR1(k_pyobj, 1);
    k.z = *(double *)PyArray_GETPTR1(k_pyobj, 2);
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    if (PySequence_Size(PyObj) != 3)
      qcm_throw("a wavevector must have 3 components");
    k.x = PyFloat_AsDouble(PySequence_GetItem(PyObj, 0));
    k.y = PyFloat_AsDouble(PySequence_GetItem(PyObj, 1));
    k.z = PyFloat_AsDouble(PySequence_GetItem(PyObj, 2));
  } else
    qcm_throw("object is not a sequence");
  return k;
}
//------------------------------------------------------------------------------
vector<vector3D<double>> many_vectors_from_Py(PyArrayObject *k_pyobj) {
  vector<vector3D<double>> k;
  if (PyArray_Check(k_pyobj)) {
    if (PyArray_NDIM(k_pyobj) != 2)
      qcm_throw("The supplied wavevectors must form a 2D Numpy array!");

    vector<npy_intp> dims(2);
    dims[0] = *PyArray_DIMS(k_pyobj);
    dims[1] = *(PyArray_DIMS(k_pyobj) + 1);
    if (dims[1] != 3)
      qcm_throw("The supplied wavevectors must have 3 components!");

    k.resize(dims[0]);
    for (size_t i = 0; i < k.size(); i++) {
      k[i].x = *(double *)PyArray_GETPTR2(k_pyobj, i, 0);
      k[i].y = *(double *)PyArray_GETPTR2(k_pyobj, i, 1);
      k[i].z = *(double *)PyArray_GETPTR2(k_pyobj, i, 2);
    }
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    k.resize(PySequence_Size(PyObj));
    for (int i = 0; i < k.size(); i++)
      k[i] = vector_from_Py((PyArrayObject *)PySequence_GetItem(PyObj, i));
  } else
    qcm_throw("object is neither a numpy array nor a sequence");
  return k;
}
//------------------------------------------------------------------------------
vector<vector3D<int64_t>> many_intvectors_from_Py(PyArrayObject *k_pyobj) {
  vector<vector3D<int64_t>> k;
  if (PyArray_Check(k_pyobj)) {
    if (PyArray_NDIM(k_pyobj) != 2)
      qcm_throw("The supplied wavevectors must form a 2D Numpy array!");

    vector<npy_intp> dims(2);
    dims[0] = *PyArray_DIMS(k_pyobj);
    dims[1] = *(PyArray_DIMS(k_pyobj) + 1);

    if (dims[1] != 3)
      qcm_throw("The supplied integer vectors must have 3 components!");
    k.assign(dims[0], vector3D<int64_t>());

    for (size_t i = 0; i < k.size(); i++) {
      k[i].x = *(int64_t *)PyArray_GETPTR2(k_pyobj, i, 0);
      k[i].y = *(int64_t *)PyArray_GETPTR2(k_pyobj, i, 1);
      k[i].z = *(int64_t *)PyArray_GETPTR2(k_pyobj, i, 2);
    }
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    k.resize(PySequence_Size(PyObj));
    for (size_t i = 0; i < k.size(); i++)
      k[i] = intvector_from_Py((PyArrayObject *)PySequence_GetItem(PyObj, i));
  }
  return k;
}
//------------------------------------------------------------------------------
vector<int64_t> intvectors_from_Py(PyArrayObject *k_pyobj) {
  vector<vector3D<int64_t>> kk = many_intvectors_from_Py(k_pyobj);
  vector<int64_t> k(3 * kk.size());
  for (int i = 0; i < kk.size(); i++) {
    k[3 * i] = kk[i].x;
    k[3 * i + 1] = kk[i].y;
    k[3 * i + 2] = kk[i].z;
  }
  return k;
}
//------------------------------------------------------------------------------
vector<int> intarray_from_Py(PyArrayObject *k_pyobj) {
  vector<int> k;
  if (PyArray_Check(k_pyobj)) {
    vector<npy_intp> dims(1);
    dims[0] = *PyArray_DIMS(k_pyobj);

    k.assign(dims[0], 0);

    int l = 0;
    for (size_t i = 0; i < dims[0]; i++) {
      k[l++] = *(int *)PyArray_GETPTR1(k_pyobj, i);
    }
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    size_t n = PySequence_Size(PyObj);
    k.assign(n, 0);
    for (size_t i = 0; i < n; i++)
      k[i] = PyLong_AsLong(PySequence_GetItem(PyObj, i));
  }
  return k;
}
//------------------------------------------------------------------------------
vector<double> vectors_from_Py(PyArrayObject *k_pyobj) {
  vector<vector3D<double>> kk = many_vectors_from_Py(k_pyobj);
  vector<double> k(3 * kk.size());
  for (int i = 0; i < kk.size(); i++) {
    k[3 * i] = kk[i].x;
    k[3 * i + 1] = kk[i].y;
    k[3 * i + 2] = kk[i].z;
  }
  return k;
}

//------------------------------------------------------------------------------
vector<complex<double>> complex_array1_from_Py(PyArrayObject *k_pyobj) {
  vector<complex<double>> k;
  if (PyArray_Check(k_pyobj)) {
    assert(PyArray_NDIM((PyArrayObject *)k_pyobj) == 1);
    npy_intp *dims;
    dims = PyArray_DIMS((PyArrayObject *)k_pyobj);

    int nelem = PyArray_SIZE(k_pyobj);
    k.assign(nelem, 0.0);
    for (size_t i = 0; i < nelem; i++) {
      k[i] = *(complex<double> *)PyArray_GETPTR1(k_pyobj, i);
    }
  }
  return k;
}

//------------------------------------------------------------------------------
matrix<complex<double>> complex_array2_from_Py(PyArrayObject *k_pyobj) {
  matrix<complex<double>> k;
  if (PyArray_Check(k_pyobj)) {
    assert(PyArray_NDIM((PyArrayObject *)k_pyobj) == 2);
    npy_intp *dims;
    dims = PyArray_DIMS((PyArrayObject *)k_pyobj);

    int nelem = PyArray_SIZE(k_pyobj);
    k.set_size(dims[0], dims[1]);
    for (size_t i = 0; i < k.r; i++) {
      for (size_t j = 0; j < k.c; j++) {
        k(i, j) = *(complex<double> *)PyArray_GETPTR2(k_pyobj, i, j);
      }
    }
  }
  return k;
}

//------------------------------------------------------------------------------
vector<matrix<complex<double>>> complex_array3_from_Py(PyArrayObject *k_pyobj) {
  vector<matrix<complex<double>>> k;
  if (PyArray_Check(k_pyobj)) {
    assert(PyArray_NDIM((PyArrayObject *)k_pyobj) == 3);
    npy_intp *dims;
    dims = PyArray_DIMS((PyArrayObject *)k_pyobj);

    k.assign(dims[0], matrix<complex<double>>(dims[1], dims[2]));
    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        for (size_t l = 0; l < dims[2]; l++) {
          k[i](j, l) = *(complex<double> *)PyArray_GETPTR3(k_pyobj, i, j, l);
        }
      }
    }
  }
  return k;
}

//------------------------------------------------------------------------------
vector<string> strings_from_PyList(PyObject *lst) {
  if (!PySequence_Check(lst))
    qcm_throw("expected a sequence of strings");
  size_t n = PySequence_Size(lst);
  vector<string> out(n);
  for (size_t i = 0; i < n; i++) {
    PyObject *pkey = PySequence_GetItem(lst, i);
    Py_ssize_t s = PyUnicode_GetLength(pkey);
    const char *key_char = PyUnicode_AsUTF8(pkey);
    out[i] = string(key_char, s);
  }
  return out;
}
//------------------------------------------------------------------------------
vector<double> doubles_from_Py(PyObject *lst) {
  vector<double> out;
  if (PyArray_Check(lst)) {
    PyArrayObject *alst = (PyArrayObject *)(lst);
    if (PyArray_NDIM(alst) != 1)
      qcm_throw("The supplied NumPy array must be one-dimensional!");
    vector<npy_intp> dims(1);
    dims[0] = *PyArray_DIMS(alst);
    out.resize(dims[0]);
    int l = 0;
    for (size_t i = 0; i < dims[0]; i++) {
      out[i] = *(double *)PyArray_GETPTR1(alst, i);
    }
  } else if (PySequence_Check(lst)) {
    size_t n = PySequence_Size(lst);
    out.resize(n);
    for (size_t i = 0; i < n; i++) {
      out[i] = PyFloat_AsDouble(PySequence_GetItem(lst, i));
    }
  } else {
    qcm_throw("expected a sequence or NumPy array and got something else!");
  }
  return out;
}

//------------------------------------------------------------------------------
map<string, double> py_dict_to_map(PyObject *D) {

  if (!PyDict_Check(D))
    qcm_ED_throw("argument of dict_to_map() is not a python dictionary");

  map<string, double> the_map;
  Py_ssize_t ppos = 0;
  PyObject *key = nullptr;
  PyObject *v = nullptr;
  while (PyDict_Next(D, &ppos, &key, &v)) {
    Py_ssize_t s = PyUnicode_GetLength(key);
    const char *key_char = PyUnicode_AsUTF8(key);
    the_map[string(key_char, s)] = PyFloat_AsDouble(v);
  }
  return the_map;
}
//------------------------------------------------------------------------------
vector<vector<int>> intmatrix_from_Py(PyArrayObject *k_pyobj) {
  vector<vector<int>> k;
  if (PyArray_Check(k_pyobj)) {
    if (PyArray_NDIM(k_pyobj) != 2)
      qcm_ED_throw(
          "generators or positions should form a two-dimensional array!");
    vector<npy_intp> dims(2);
    dims[0] = *PyArray_DIMS(k_pyobj);
    dims[1] = *(PyArray_DIMS(k_pyobj) + 1);
    k.assign(dims[0], vector<int>(dims[1]));

    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        k[i][j] = *(int *)PyArray_GETPTR2(k_pyobj, i, j);
      }
    }
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    size_t n = PySequence_Size(PyObj);
    k.assign(n, vector<int>());
    for (size_t i = 0; i < n; i++) {
      PyObject *PyObj2 = PySequence_GetItem(PyObj, i);
      if (!PySequence_Check(PyObj2))
        qcm_ED_throw("generator or position " + to_string(i + 1) +
                     " should be a sequence!");
      size_t m = PySequence_Size(PyObj2);
      k[i].assign(m, 0);
      for (size_t j = 0; j < m; j++) {
        k[i][j] = (int)PyLong_AsLong(PySequence_GetItem(PyObj2, j));
      }
    }
  }
  return k;
}

//------------------------------------------------------------------------------
vector<vector<double>> pos_from_Py(PyArrayObject *k_pyobj) {
  vector<vector<double>> k;
  if (PyArray_Check(k_pyobj)) {
    if (PyArray_NDIM(k_pyobj) != 2)
      qcm_ED_throw("positions should form a two-dimensional array!");
    vector<npy_intp> dims(2);
    dims[0] = *PyArray_DIMS(k_pyobj);
    dims[1] = *(PyArray_DIMS(k_pyobj) + 1);

    k.assign(dims[0], vector<double>(dims[1]));

    for (size_t i = 0; i < dims[0]; i++) {
      for (size_t j = 0; j < dims[1]; j++) {
        k[i][j] = *(double *)PyArray_GETPTR2(k_pyobj, i, j);
      }
    }
  } else if (PySequence_Check((PyObject *)k_pyobj)) {
    PyObject *PyObj = (PyObject *)k_pyobj;
    size_t n = PySequence_Size(PyObj);
    k.assign(n, vector<double>());
    for (size_t i = 0; i < n; i++) {
      PyObject *PyObj2 = PySequence_GetItem(PyObj, i);
      if (!PySequence_Check(PyObj2))
        qcm_ED_throw("position " + to_string(i + 1) + " should be a sequence!");
      size_t m = PySequence_Size(PyObj2);
      k[i].assign(m, 0);
      for (size_t j = 0; j < m; j++) {
        k[i][j] = (double)PyFloat_AsDouble(PySequence_GetItem(PyObj2, j));
      }
    }
  }
  return k;
}
