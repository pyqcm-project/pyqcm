#ifndef qcm_ED_wrap_h
#define qcm_ED_wrap_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include "numpy/arrayobject.h"
#include <Python.h>
#include "common_Py.hpp"
#include "float.h"
#include "model_instance.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"

#include "qcm_nb.hpp"

extern map<string, shared_ptr<model>> models;
extern map<size_t, shared_ptr<model_instance_base>> model_instances;

static PyObject *qcm_ED_Error;

//==============================================================================
// Registration of the QCM_ED part of the `qcm` module.
//==============================================================================
inline void register_qcm_ED(nb::module_ &m) {

  m.def("ED_complex_HS",
        [](size_t label) { return (int)ED::complex_HS(label); },
        "label"_a = 0,
        "return 1 if the Hilbert space is complex, 0 if it is real");

  m.def("density_matrix",
        [](std::vector<int> sites, int label) {
          auto g = ED::density_matrix(sites, label);
          size_t r = g.first.r;
          auto dm = nb_array_<complex<double>>(g.first.data(), {r, r});
          auto basis = nb_array_<uint64_t>(g.second.data(), {g.second.size()});
          return nb::make_tuple(dm, basis);
        },
        "sites"_a, "label"_a = 0,
        "computes the density matrix from the ground state; returns the matrix "
        "and the basis used");

  m.def("Green_function_average",
        [](int label, int spin_down) {
          auto g = ED::Green_function_average((bool)spin_down, (size_t)label);
          return nb_array_<complex<double>>(g.data(), {g.r, g.c});
        },
        "label"_a = 0, "spin_down"_a = 0,
        "returns the average of c^dagger_i c_j (matrix)");

  m.def("Green_function_density",
        [](int label) { return ED::Green_function_density((size_t)label); },
        "label"_a = 0,
        "the density of the cluster, from the trace of the Green function average");

  m.def("Green_function_dimensionC",
        [](int label) { return (int)ED::Green_function_dimension((size_t)label); },
        "label"_a = 0, "the dimension of the Green function matrix (int)");

  m.def("Green_function_solveC",
        [](int label) { ED::Green_function_solve((size_t)label); },
        "label"_a = 0, "solves the cluster Green function");

  m.def("Green_function",
        [](complex<double> z, int spin_down, int label, int blocks) {
          auto g = ED::Green_function(z, (bool)spin_down, (size_t)label,
                                      (bool)blocks);
          return nb_array_<complex<double>>(g.data(), {g.r, g.c});
        },
        "z"_a, "spin_down"_a = 0, "label"_a = 0, "blocks"_a = 0,
        "computes the Green function matrix at a given complex frequency");

  m.def("cluster_averages",
        [](int label) {
          auto ave = ED::cluster_averages((size_t)label);
          nb::dict d;
          for (auto &x : ave)
            d[nb::str(get<0>(x).c_str())] =
                nb::make_tuple(get<1>(x), get<2>(x));
          return d;
        },
        "label"_a = 0,
        "ground state averages; returns a dict string : (average, variance)");

  m.def("ground_state_solve",
        [](int label) {
          auto r = ED::ground_state_solve((size_t)label);
          return nb::make_tuple(r.first, r.second);
        },
        "label"_a = 0,
        "computes the ground state; returns (energy, Hilbert space sector)");

  m.def("hopping_matrix",
        [](int spin_down, int diag, int label, int full) {
          if (full) {
            auto g = ED::hopping_matrix_full((bool)spin_down, (bool)diag,
                                             (size_t)label);
            return nb_array_<complex<double>>(g.data(), {g.r, g.c});
          } else {
            auto g = ED::hopping_matrix((bool)spin_down, (size_t)label);
            return nb_array_<complex<double>>(g.data(), {g.r, g.c});
          }
        },
        "spin_down"_a = 0, "diag"_a = 0, "label"_a = 0, "full"_a = 0,
        "computes the hopping matrix of the model");

  m.def("interactions",
        [](int label) { return ED::interactions((size_t)label); },
        "label"_a = 0,
        "the list of density-density interactions (list of (int,int,double))");

  m.def("hybridization_functionC",
        [](complex<double> z, int spin_down, int label) {
          auto g = ED::hybridization_function(z, (bool)spin_down, (size_t)label);
          return nb_array_<complex<double>>(g.data(), {g.r, g.c});
        },
        "z"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the hybridization function (for models with baths)");

  m.def("matrix_elements",
        [](const std::string &S1, const std::string &S2) {
          auto ET = ED::matrix_elements(S1, S2);
          nb::list lst;
          for (auto &e : ET.second)
            lst.append(nb::make_tuple(e.r, e.c, e.v));
          return nb::make_tuple(ET.first, lst);
        },
        "model"_a, "op"_a,
        "the list of matrix elements that defines an operator");

  m.def("mixingC", [](int label) { return ED::mixing((size_t)label); },
        "label"_a = 0,
        "the mixing state of the model (0:none, 1:anomalous, 2:spin-flip, "
        "3:both)");

  m.def("model_sizeC",
        [](const std::string &S1) {
          auto d = ED::model_size(S1);
          return nb::make_tuple((int)get<0>(d), (int)get<1>(d));
        },
        "name"_a,
        "returns (number of cluster sites, number of bath sites)");

  m.def("new_model",
        [](const std::string &S1, int n_sites, int n_bath, nb::object gen_obj,
           int bath_irrep) {
          vector<vector<int>> gen;
          if (!gen_obj.is_none())
            gen = intmatrix_from_Py((PyArrayObject *)gen_obj.ptr());
          int n_orb = n_sites + n_bath;
          for (size_t j = 0; j < gen.size(); j++) {
            if ((int)gen[j].size() != n_orb)
              qcm_ED_throw("generator " + to_string(j + 1) + " should have " +
                           to_string(n_orb) + " elements");
            for (int i = 0; i < n_sites; i++) gen[j][i] -= 1;
            if (bath_irrep == false)
              for (int i = n_sites; i < n_orb; i++) gen[j][i] -= 1;
          }
          ED::new_model(S1, n_sites, n_bath, gen, bath_irrep);
        },
        "name"_a, "n_sites"_a, "n_bath"_a, "generators"_a = nb::none(),
        "bath_irrep"_a = 0, "initiates a new model (no operators yet)");

  m.def("new_model_instanceC",
        [](const std::string &name, nb::object val, const std::string &sec,
           int label) {
          map<string, double> param = py_dict_to_map(val.ptr());
          ED::new_model_instance(name, param, sec, (size_t)label);
        },
        "name"_a, "values"_a, "sectors"_a, "label"_a = 0,
        "initiates a new instance of the model");

  m.def("new_operator",
        [](const std::string &name, const std::string &op,
           const std::string &type, nb::object elem_obj) {
          vector<matrix_element<double>> elem;
          PyObject *elem_pyobj = elem_obj.ptr();
          double fac = 1.0;
          bool check_upper = true;
          if (type == "anomalous") fac = 0.5;
          if (type == "general_interaction") check_upper = false;
          if (PyArray_Check(elem_pyobj)) {
            size_t nelem = PyArray_DIMS((PyArrayObject *)elem_pyobj)[0];
            elem.resize(nelem);
            memcpy(elem.data(), PyArray_DATA((PyArrayObject *)elem_pyobj),
                   nelem * PyArray_STRIDES((PyArrayObject *)elem_pyobj)[0]);
          } else if (PySequence_Check(elem_pyobj)) {
            size_t n = PySequence_Size(elem_pyobj);
            elem.assign(n, matrix_element<double>());
            for (size_t i = 0; i < n; i++) {
              PyObject *pkey = PySequence_GetItem(elem_pyobj, i);
              if (PyTuple_Size(pkey) == 3) {
                elem[i].r = PyLong_AsLong(PyTuple_GetItem(pkey, 0));
                elem[i].c = PyLong_AsLong(PyTuple_GetItem(pkey, 1));
                if ((elem[i].r > elem[i].c) and check_upper)
                  qcm_ED_throw("the first index of element " + to_string(i) +
                               " of argument 4 of 'new_operator' cannot be "
                               "bigger than the second index");
                if (elem[i].r == 0 or elem[i].c == 0)
                  qcm_ED_throw("indices in matrix elements of operators are "
                               "labelled starting at 1, not at 0.");
                elem[i].r--;
                elem[i].c--;
                elem[i].v = fac * PyFloat_AsDouble(PyTuple_GetItem(pkey, 2));
              } else {
                qcm_ED_throw("element " + to_string(i) +
                             " of argument 4 of 'new_operator' should be a "
                             "3-tuple");
              }
            }
          } else
            qcm_ED_throw("argument 4 of new_operator() must be a list or array");
          ED::new_operator(name, op, type, elem);
        },
        "model"_a, "op"_a, "type"_a, "elements"_a,
        "creates a new operator from its (real) matrix elements");

  m.def("new_operator_complex",
        [](const std::string &name, const std::string &op,
           const std::string &type, nb::object elem_obj) {
          vector<matrix_element<complex<double>>> elem;
          PyObject *elem_pyobj = elem_obj.ptr();
          double fac = 1.0;
          bool check_upper = true;
          if (type == "anomalous") fac = 0.5;
          if (type == "general_interaction") check_upper = false;
          if (PyArray_Check(elem_pyobj)) {
            size_t nelem = PyArray_DIMS((PyArrayObject *)elem_pyobj)[0];
            elem.resize(nelem);
            memcpy(elem.data(), PyArray_DATA((PyArrayObject *)elem_pyobj),
                   nelem * PyArray_STRIDES((PyArrayObject *)elem_pyobj)[0]);
          } else if (PySequence_Check(elem_pyobj)) {
            size_t n = PySequence_Size(elem_pyobj);
            elem.assign(n, matrix_element<complex<double>>());
            for (size_t i = 0; i < n; i++) {
              PyObject *pkey = PySequence_GetItem(elem_pyobj, i);
              if (PyTuple_Size(pkey) == 3) {
                elem[i].r = PyLong_AsLong(PyTuple_GetItem(pkey, 0));
                elem[i].c = PyLong_AsLong(PyTuple_GetItem(pkey, 1));
                if ((elem[i].r > elem[i].c) and check_upper)
                  qcm_ED_throw("the first index of element " + to_string(i) +
                               " of argument 4 of 'new_operator' cannot be "
                               "bigger than the second index");
                if (elem[i].r == 0 or elem[i].c == 0)
                  qcm_ED_throw("indices in matrix elements of operators are "
                               "labelled starting at 1, not at 0.");
                elem[i].r--;
                elem[i].c--;
                Py_complex z = PyComplex_AsCComplex(PyTuple_GetItem(pkey, 2));
                elem[i].v = {fac * z.real, fac * z.imag};
              } else {
                qcm_ED_throw("element " + to_string(i) +
                             " of argument 4 of 'new_operator' should be a "
                             "3-tuple");
              }
            }
          } else
            qcm_ED_throw("argument 4 of new_operator() must be a list or array");
          ED::new_operator(name, op, type, elem);
        },
        "model"_a, "op"_a, "type"_a, "elements"_a,
        "creates a new operator from its (complex) matrix elements");

  m.def("parametersC",
        [](int label) {
          if (model_instances.find(label) == model_instances.end())
            qcm_ED_throw("The label " + to_string(label) + " is out of range.");
          auto M = model_instances.at(label)->value;
          string model_name = model_instances.at(label)->the_model->name;
          nb::dict d;
          for (auto &x : M) d[nb::str(x.first.c_str())] = x.second;
          return nb::make_tuple(d, model_name);
        },
        "label"_a = 0, "returns (dict of parameters, model name)");

  m.def("print_models", []() { ED::print_models(cout); },
        "prints the model description to the screen");

  m.def("print_graph",
        [](const std::string &name, nb::object pos_obj) {
          vector<vector<double>> pos =
              pos_from_Py((PyArrayObject *)pos_obj.ptr());
          if (models.find(name) == models.end())
            qcm_ED_throw("model " + name + " does not exist!");
          models[name]->print_graph(pos);
        },
        "name"_a, "pos"_a, "prints a graphviz file for the model specified");

  m.def("self_energyC",
        [](complex<double> z, int spin_down, int label) {
          auto g = ED::self_energy(z, (bool)spin_down, (size_t)label);
          return nb_array_<complex<double>>(g.data(), {g.r, g.c});
        },
        "z"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the self-energy matrix at a given complex frequency");

  m.def("set_global_parameterC",
        [](const std::string &name, nb::object obj) {
          if (obj.is_none()) {
            set_global_bool(name, true);
            cout << "global parameter " << name << " set to true" << endl;
          } else {
            PyObject *o = obj.ptr();
            if (PyLong_Check(o)) {
              size_t I = (int)PyLong_AsLong(o);
              set_global_int(name, I);
              cout << "global parameter " << name << " set to " << I << endl;
            } else if (PyFloat_Check(o)) {
              double I = PyFloat_AsDouble(o);
              set_global_double(name, I);
              cout << "global parameter " << name << " set to " << I << endl;
            } else
              qcm_ED_throw("unknown type of global_parameter");
          }
        },
        "name"_a, "value"_a = nb::none(),
        "sets the value of a global parameter");

  m.def("susceptibility_poles",
        [](const std::string &op, int label) {
          auto g = ED::susceptibility_poles(op, (size_t)label);
          return nb_array_<double>(reinterpret_cast<const double *>(g.data()),
                                   {g.size(), 2});
        },
        "op"_a, "label"_a = 0,
        "dynamic susceptibility of an operator; array of (residue, pole)");

  m.def("susceptibility",
        [](const std::string &op, std::vector<complex<double>> w, int label) {
          auto g = ED::susceptibility(op, w, (size_t)label);
          return nb_array_<complex<double>>(g.data(), {g.size()});
        },
        "op"_a, "w"_a, "label"_a = 0,
        "dynamic susceptibility of an operator at the given frequencies");

  m.def("print_wavefunction",
        [](int label) { return ED::print_wavefunction((size_t)label); },
        "label"_a = 0, "prints the ground state wavefunction(s) to a string");

  m.def("qmatrix",
        [](int label) {
          auto Q = ED::qmatrix(false, label);
          size_t M = Q.first.size();
          size_t L = M ? Q.second.size() / M : 0;
          auto evals = nb_array_<double>(Q.first.data(), {M});
          auto mat = nb_array_<complex<double>>(Q.second.data(), {M, L});
          return nb::make_tuple(evals, mat);
        },
        "label"_a = 0,
        "Lehmann representation of the Green function: (eigenvalues, MxL matrix)");

  m.def("hybridization",
        [](bool spin_down, int label) {
          auto Q = ED::hybridization(spin_down, label);
          size_t M = Q.first.size();
          size_t L = M ? Q.second.size() / M : 0;
          auto evals = nb_array_<double>(Q.first.data(), {M});
          auto mat = nb_array_<complex<double>>(Q.second.data(), {M, L});
          return nb::make_tuple(evals, mat);
        },
        "spin_down"_a = false, "label"_a = 0,
        "Lehmann representation of the hybridization function: (eigenvalues, "
        "MxL matrix)");

  m.def("write_instance_to_hdf5",
        [](const std::string &filename, const std::string &group, int label) {
          ED::write_instance_hdf5(filename, label, group);
        },
        "filename"_a, "group"_a, "label"_a = 0,
        "writes the solved model instance to a group inside an HDF5 file");

  m.def("read_instance_from_hdf5",
        [](const std::string &filename, const std::string &group, int label) {
          ED::read_instance_hdf5(filename, label, group);
        },
        "filename"_a, "group"_a, "label"_a = 0,
        "reads the solved model instance from a group inside an HDF5 file");

  m.def("fidelity",
        [](int label1, int label2) { return ED::fidelity(label1, label2); },
        "label1"_a, "label2"_a,
        "the fidelity between two model instances");
}

#endif
