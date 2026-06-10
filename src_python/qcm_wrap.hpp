#ifndef qcm_wrap_h
#define qcm_wrap_h

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <array>
#include <map>
#include <memory>
#include <string>

#define PY_SSIZE_T_CLEAN
#include "Python.h"
#include "float.h"
#include "numpy/ndarrayobject.h"

#include "QCM.hpp"
#include "common_Py.hpp"
#include "lattice_model.hpp"
#include "parameter_set.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"

#include "qcm_nb.hpp"

vector3D<int64_t> intvector_from_Py(PyArrayObject *k_pyobj);
vector3D<double> vector_from_Py(PyArrayObject *k_pyobj);
vector<vector3D<double>> many_vectors_from_Py(PyArrayObject *k_pyobj);
vector<string> strings_from_PyList(PyObject *lst);

extern shared_ptr<lattice_model> qcm_model;
extern run_statistics STATS;

extern map<string, global_parameter<bool>> GP_bool;
void qcm_catch(const std::exception &e);

extern vector<double> grid_freqs; // optional imaginary frequency grid for discrete integrals
extern vector<double> grid_weights; // weights associated with grid_freqs

extern PyObject *qcm_Error;

//==============================================================================
// helper: require that a Python object is a NumPy array and return its ndim.
//==============================================================================
inline int require_array_ndim(PyObject *o, const char *who) {
  if (!PyArray_Check(o))
    qcm_throw(string("the wavevector argument of '") + who +
              "' must be a numpy array");
  return PyArray_NDIM((PyArrayObject *)o);
}

//==============================================================================
// Registration of the QCM part of the `qcm` module.
//==============================================================================
inline void register_qcm(nb::module_ &m) {

  //------------------------------------------------------------------ structure
  m.def("add_cluster",
        [](nb::object cpos, nb::object pos, int ref, int conj) {
          QCM::add_cluster(intvector_from_Py((PyArrayObject *)cpos.ptr()),
                           many_intvectors_from_Py((PyArrayObject *)pos.ptr()),
                           ref, (bool)conj);
        },
        "cpos"_a, "pos"_a, "ref"_a = 0, "conj"_a = 0,
        "adds a cluster to the repeated unit");

  m.def("add_system",
        [](const std::string &name, int clus) { QCM::add_system(name, clus); },
        "name"_a, "clus"_a, "adds a system to the repeated unit");

  m.def("lattice_model",
        [](const std::string &name, nb::object super, nb::object unit,
           nb::object lattice_hybrid) {
          if (qcm_model == nullptr)
            qcm_throw("no cluster has been added to the model!");
          vector<int64_t> superlattice =
              intvectors_from_Py((PyArrayObject *)super.ptr());
          vector<int64_t> unit_cell;
          if (unit.is_none()) {
            unit_cell.assign(superlattice.size(), 0);
            size_t dim = superlattice.size() / 3;
            for (size_t i = 0; i < dim; i++) unit_cell[i * 4] = 1;
          } else
            unit_cell = intvectors_from_Py((PyArrayObject *)unit.ptr());
          if (lattice_hybrid.is_none())
            QCM::new_lattice_model(name, superlattice, unit_cell);
          else
            QCM::new_lattice_model(name, superlattice, unit_cell,
                                   nb::cast<std::string>(lattice_hybrid));
        },
        "name"_a, "superlattice"_a, "lattice"_a = nb::none(),
        "lattice_hybridization"_a = nb::none(), "initiates the lattice model");

  m.def("new_model_instance",
        [](int label) { QCM::new_model_instance(label); }, "label"_a = 0,
        "creates a new instance of the lattice model");

  m.def("erase_model_instance",
        [](int label) { QCM::erase_lattice_model_instance(label); },
        "label"_a = 0, "erases a model instance from the list");

  m.def("great_reset", []() { QCM::great_reset(); },
        "resets the whole system and allows to redefine the model");

  m.def("set_basis",
        [](nb::object basis) {
          vector<double> b = vectors_from_Py((PyArrayObject *)basis.ptr());
          QCM::set_basis(b);
        },
        "basis"_a, "sets the working basis (a Dx3 real matrix)");

  //----------------------------------------------------------------- operators
  m.def("hopping_operator",
        [](const std::string &name, nb::object link, double amplitude, int orb1,
           int orb2, int tau, int sigma) {
          vector3D<int64_t> lk = intvector_from_Py((PyArrayObject *)link.ptr());
          QCM::hopping_operator(name, lk, amplitude, orb1, orb2, tau, sigma);
        },
        "name"_a, "link"_a, "amplitude"_a, "orb1"_a = 1, "orb2"_a = 1,
        "tau"_a = 1, "sigma"_a = 0, "defines a one-body (hopping) operator");

  m.def("interaction_operator",
        [](const std::string &name, nb::object link, double amplitude, int orb1,
           int orb2, const std::string &type) {
          vector3D<int64_t> lk(0, 0, 0);
          if (!link.is_none())
            lk = intvector_from_Py((PyArrayObject *)link.ptr());
          QCM::interaction_operator(name, lk, amplitude, orb1, orb2, type);
        },
        "name"_a, "link"_a = nb::none(), "amplitude"_a = 1.0, "orb1"_a = 1,
        "orb2"_a = 1, "type"_a = "Hubbard",
        "defines an interaction operator (Hubbard, Hund, Heisenberg, X, Y, Z)");

  m.def("current_operator",
        [](const std::string &name, nb::object link, double amplitude, int orb1,
           int orb2, int dir, int real) {
          vector3D<int64_t> lk = intvector_from_Py((PyArrayObject *)link.ptr());
          QCM::current_operator(name, lk, amplitude, orb1, orb2, dir, real);
        },
        "name"_a, "link"_a, "amplitude"_a, "orb1"_a = 1, "orb2"_a = 1,
        "dir"_a = 0, "real"_a = 1, "defines a current operator");

  m.def("anomalous_operator",
        [](const std::string &name, nb::object link, complex<double> amplitude,
           int orb1, int orb2, const std::string &type) {
          vector3D<int64_t> lk = intvector_from_Py((PyArrayObject *)link.ptr());
          QCM::anomalous_operator(name, lk, amplitude, orb1, orb2, type);
        },
        "name"_a, "link"_a, "amplitude"_a = complex<double>(1.0), "orb1"_a = 1,
        "orb2"_a = 1, "type"_a = "singlet",
        "defines an anomalous (pairing) operator");

  m.def("density_wave",
        [](const std::string &name, const std::string &type, nb::object Q,
           nb::object link, complex<double> amplitude, int orb, double phase) {
          vector3D<int64_t> lk(0, 0, 0);
          if (!link.is_none())
            lk = intvector_from_Py((PyArrayObject *)link.ptr());
          vector3D<double> Qv = vector_from_Py((PyArrayObject *)Q.ptr());
          QCM::density_wave(name, lk, amplitude, orb, Qv, phase, type);
        },
        "name"_a, "type"_a, "Q"_a, "link"_a = nb::none(),
        "amplitude"_a = complex<double>(1.0), "orb"_a = 0, "phase"_a = 0.0,
        "defines a density wave operator");

  m.def("explicit_operator",
        [](const std::string &name, nb::object elem_obj, int tau, int sigma,
           const std::string &type) {
          PyObject *eo = elem_obj.ptr();
          if (!PySequence_Check(eo))
            qcm_throw("argument 2 of 'explicit_operator' is not a python list");
          size_t n = PySequence_Size(eo);
          vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>>
              elem(n);
          for (size_t i = 0; i < n; i++) {
            PyObject *t = PySequence_GetItem(eo, i);
            if (!PyTuple_Check(t) || PyTuple_Size(t) != 3) {
              Py_DECREF(t);
              qcm_throw("element " + to_string(i) +
                        " passed to explicit_operator is not a 3-tuple");
            }
            get<0>(elem[i]) =
                intvector_from_Py((PyArrayObject *)PyTuple_GetItem(t, 0));
            get<1>(elem[i]) =
                intvector_from_Py((PyArrayObject *)PyTuple_GetItem(t, 1));
            Py_complex z = PyComplex_AsCComplex(PyTuple_GetItem(t, 2));
            get<2>(elem[i]) = complex<double>(z.real, z.imag);
            Py_DECREF(t);
          }
          QCM::explicit_operator(name, type, elem, tau, sigma);
        },
        "name"_a, "elements"_a, "tau"_a = 1, "sigma"_a = 0,
        "type"_a = "one-body", "defines an explicit operator");

  //---------------------------------------------------------------- parameters
  m.def("set_parameter",
        [](const std::string &name, double value) {
          QCM::set_parameter(name, value);
        },
        "name"_a, "value"_a, "sets the value of a parameter");

  m.def("set_multiplier",
        [](const std::string &name, double value) {
          QCM::set_multiplier(name, value);
        },
        "name"_a, "value"_a, "sets the multiplier of a dependent parameter");

  m.def("set_parameters",
        [](nb::object py_values) {
          PyObject *pv = py_values.ptr();
          if (!PySequence_Check(pv))
            qcm_throw("argument of 'set_parameters' should be a list");
          size_t n = PySequence_Size(pv);
          vector<pair<string, double>> values;
          values.reserve(n);
          vector<tuple<string, double, string>> equiv;
          equiv.reserve(n);
          for (size_t i = 0; i < n; i++) {
            PyObject *pkey = PySequence_GetItem(pv, i);
            if (PyTuple_Size(pkey) == 2) {
              values.push_back({Py2string(PyTuple_GetItem(pkey, 0)),
                                PyFloat_AsDouble(PyTuple_GetItem(pkey, 1))});
            } else if (PyTuple_Size(pkey) == 3) {
              equiv.push_back({Py2string(PyTuple_GetItem(pkey, 0)),
                               PyFloat_AsDouble(PyTuple_GetItem(pkey, 1)),
                               Py2string(PyTuple_GetItem(pkey, 2))});
            } else {
              Py_DECREF(pkey);
              qcm_throw("element " + to_string(i) +
                        " of argument of 'set_parameters' should be a 2- or "
                        "3-tuple");
            }
            Py_DECREF(pkey);
          }
          QCM::set_parameters(values, equiv);
        },
        "values"_a, "sets the values/dependences of multiple parameters");

  m.def("set_target_sectors",
        [](nb::object py_sectors) {
          PyObject *ps = py_sectors.ptr();
          if (!PySequence_Check(ps))
            qcm_throw("argument of 'set_target_sectors' should be a sequence");
          size_t n = PySequence_Size(ps);
          if (n != qcm_model->nsys)
            qcm_throw("The number of strings in argument of "
                      "'set_target_sectors' (" +
                      to_string(n) +
                      ") should be the number of systems in the repeated unit (" +
                      to_string(qcm_model->nsys) + ")");
          qcm_model->sector_strings.resize(n);
          for (size_t i = 0; i < n; i++) {
            PyObject *item = PySequence_GetItem(ps, i);
            qcm_model->sector_strings[i] = Py2string(item);
            Py_DECREF(item);
          }
        },
        "sectors"_a, "defines the target Hilbert space sectors to be used");

  m.def("parameters", []() { return QCM::parameters(); },
        "returns the values of the parameters in the parameter set (dict)");

  m.def("instance_parameters",
        [](int label) { return QCM::instance_parameters(label); },
        "label"_a = 0,
        "returns the values of the parameters in a given instance (dict)");

  m.def("print_parameter_set",
        []() { qcm_model->param_set->print(cout); },
        "prints the parameter set to the screen");

  m.def("parameter_set",
        []() {
          if (qcm_model->param_set == nullptr)
            qcm_throw("The parameter set has not been defined yet");
          nb::dict d;
          for (auto &x : qcm_model->param_set->param) {
            nb::object overlord = nb::none();
            nb::object mult = nb::none();
            if (x.second->ref) {
              overlord = nb::cast(x.second->ref->name);
              mult = nb::cast(x.second->multiplier);
            }
            d[nb::str(x.first.c_str())] =
                nb::make_tuple(x.second->value, overlord, mult);
          }
          return d;
        },
        "returns the content of the parameter set (dict of (value, overlord, "
        "multiplier))");

  //------------------------------------------------------------------- globals
  m.def("set_global_parameter",
        [](const std::string &name, nb::object obj) {
          if (obj.is_none()) {
            set_global_bool(name, true);
            cout << "global parameter " << name << " set to true" << endl;
          } else {
            PyObject *o = obj.ptr();
            if (PyLong_Check(o)) {
              size_t I = (size_t)PyLong_AsUnsignedLong(o);
              if (GP_bool.find(name) != GP_bool.end()) {
                set_global_bool(name, I != 0);
                cout << "global parameter " << name << " set to "
                     << (I ? "true" : "false") << endl;
              } else {
                set_global_int(name, I);
                cout << "global parameter " << name << " set to " << I << endl;
              }
            } else if (PyFloat_Check(o)) {
              double I = PyFloat_AsDouble(o);
              set_global_double(name, I);
              cout << "global parameter " << name << " set to " << I << endl;
            } else if (PyUnicode_Check(o)) {
              const char *op_char = PyUnicode_AsUTF8(o);
              set_global_char(name, op_char[0]);
              cout << "global parameter " << name << " set to " << op_char[0]
                   << endl;
            } else
              qcm_throw("unknown type of global_parameter");
          }
        },
        "name"_a, "value"_a = nb::none(),
        "sets the value of a global parameter");

  m.def("get_global_parameter",
        [](const std::string &name) -> nb::object {
          if (is_global_bool(name)) return nb::cast((int)global_bool(name));
          if (is_global_int(name)) return nb::cast((int)global_int(name));
          if (is_global_double(name)) return nb::cast(global_double(name));
          if (is_global_char(name)) {
            char C[2] = {global_char(name), '\0'};
            return nb::cast(std::string(C));
          }
          qcm_throw(name + " is not the name of a global_parameter");
          return nb::none();
        },
        "name"_a, "gets the value of a global parameter");

  m.def("print_options",
        [](int to_file) {
          QCM::global_parameter_init();
          print_options(to_file);
        },
        "to_file"_a,
        "prints the list of global options and parameters (0:screen, 1:latex, "
        "2:RST)");

  //----------------------------------------------------------- simple queries
  m.def("mixing", []() { return QCM::mixing(); },
        "returns the mixing state of the system (integer code)");

  m.def("spatial_dimension", []() { return QCM::spatial_dimension(); },
        "returns the spatial dimension (0,1,2 or 3) of the model");

  m.def("complex_HS", [](int label) { return (int)QCM::complex_HS(label); },
        "label"_a = 0,
        "True if the model instance requires a complex Hilbert space");

  m.def("Green_function_dimension",
        []() { return (int)QCM::Green_function_dimension(); },
        "returns the dimension of the CPT Green function matrix");

  m.def("reduced_Green_function_dimension",
        []() { return (int)QCM::reduced_Green_function_dimension(); },
        "returns the dimension of the reduced Green function");

  m.def("model_size",
        []() {
          return nb::make_tuple((int)qcm_model->sites.size(),
                                (int)qcm_model->n_band);
        },
        "returns (size of the supercell, number of lattice orbitals)");

  m.def("model_is_closed", []() { return (int)qcm_model->is_closed; },
        "True if the model is no longer modifiable");

  m.def("cluster_info", []() { return QCM::cluster_info(); },
        "list of 5-tuples (name, n_sites, n_bath, dim_GF, n_sym) per cluster");

  m.def("ground_state", [](int label) { return QCM::ground_state(label); },
        "label"_a = 0,
        "computes the ground state; list of (energy, sector) per cluster");

  m.def("Green_function_solve",
        [](int label) { QCM::Green_function_solve(label); }, "label"_a,
        "forces (non-lazy) computation of cluster Green functions");

  m.def("averages",
        [](nb::object lst, int label) {
          vector<string> ops;
          if (!lst.is_none() && PyList_Size(lst.ptr()) > 0)
            ops = strings_from_PyList(lst.ptr());
          auto ave = QCM::averages(ops, label);
          nb::dict d;
          for (auto &x : ave) d[nb::str(x.first.c_str())] = x.second;
          return d;
        },
        "ops"_a = nb::none(), "label"_a = 0,
        "returns the average values of operators in a model instance (dict)");

  //----------------------------------------------------------------- energies
  m.def("Potthoff_functional",
        [](int label) { return QCM::Potthoff_functional(label); },
        "label"_a = 0, "computes the Potthoff functional for a given instance");

  m.def("potential_energy",
        [](int label) { return QCM::potential_energy(label); }, "label"_a = 0,
        "computes the potential energy for a given instance");

  m.def("kinetic_energy",
        [](int label) { return QCM::kinetic_energy(label); }, "label"_a = 0,
        "returns the kinetic energy for a given instance");

  m.def("spectral_average",
        [](const std::string &name, complex<double> z, int label) {
          return QCM::spectral_average(name, z, label);
        },
        "name"_a, "z"_a, "label"_a = 0,
        "contribution of a frequency to the average of an operator");

  //--------------------------------------------------------- cluster matrices
  m.def("cluster_Green_function",
        [](int clus, complex<double> z, int spin_down, int label, int blocks) {
          auto g = QCM::cluster_Green_function((size_t)clus, z, (bool)spin_down,
                                               label, (bool)blocks);
          return nb_cmat(g);
        },
        "clus"_a, "z"_a, "spin_down"_a = 0, "label"_a = 0, "blocks"_a = 0,
        "computes the cluster Green function at a given frequency");

  m.def("cluster_self_energy",
        [](int clus, complex<double> z, int spin_down, int label) {
          auto g =
              QCM::cluster_self_energy((size_t)clus, z, (bool)spin_down, label);
          return nb_cmat(g);
        },
        "clus"_a, "z"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the cluster self energy at a given frequency");

  m.def("cluster_hopping_matrix",
        [](int clus, int spin_down, int label) {
          auto g =
              QCM::cluster_hopping_matrix((size_t)clus, (bool)spin_down, label);
          return nb_cmat(g);
        },
        "clus"_a, "spin_down"_a = 0, "label"_a = 0,
        "returns the one-body matrix of a cluster");

  m.def("Green_integral",
        [](int spin_down, int clus, int label) {
          auto g = QCM::Green_integral((bool)spin_down, clus, label);
          return nb_cmat(g);
        },
        "spin_down"_a = 0, "clus"_a = 0, "label"_a = 0,
        "frequency integral of the local lattice Green function");

  m.def("hybridization_function",
        [](complex<double> z, int spin_down, int clus, int label) {
          auto g = QCM::hybridization_function(z, (bool)spin_down, (size_t)clus,
                                               label);
          return nb_cmat(g);
        },
        "z"_a, "spin_down"_a = 0, "clus"_a = 0, "label"_a = 0,
        "returns the hybridization function for a cluster and instance");

  m.def("hybridization_function_sys",
        [](complex<double> z, int spin_down, int sys, int label) {
          auto g = ED::hybridization_function(
              z, (bool)spin_down, (size_t)(qcm_model->nsys * label + sys));
          return nb_cmat(g);
        },
        "z"_a, "spin_down"_a = 0, "sys"_a = 0, "label"_a = 0,
        "returns the hybridization function for a system and instance");

  m.def("projected_Green_function",
        [](complex<double> z, int spin_down, int label) {
          auto g = QCM::projected_Green_function(z, (bool)spin_down, label);
          return nb_cmat(g);
        },
        "z"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the projected Green function (as used in CDMFT)");

  m.def("compact_tiling",
        [](nb::object A, nb::object k) {
          size_t d = QCM::Green_function_dimension();
          PyObject *Ao = A.ptr();
          if (!PyArray_Check(Ao) || PyArray_NDIM((PyArrayObject *)Ao) != 2 ||
              (size_t)PyArray_DIM((PyArrayObject *)Ao, 0) != d ||
              (size_t)PyArray_DIM((PyArrayObject *)Ao, 1) != d)
            qcm_throw("argument A of compact_tiling must be a square complex "
                      "array of size dim_GF");
          matrix<complex<double>> Amat(d);
          memcpy(Amat.v.data(), PyArray_DATA((PyArrayObject *)Ao),
                 d * d * sizeof(complex<double>));
          auto Act =
              QCM::compact_tiling(Amat, vector_from_Py((PyArrayObject *)k.ptr()));
          return nb_cmat(Act);
        },
        "A"_a, "k"_a,
        "symmetrizes a dim_GF matrix w.r.t. cluster translations at k");

  m.def("combined_mcf",
        [](int label, nb::object k_obj) {
          ED::CombinedMCF_data D;
          if (k_obj.is_none()) {
            D = ED::get_combined_mcf(false, label);
          } else {
            PyObject *ko = k_obj.ptr();
            if (!PyArray_Check(ko))
              qcm_throw("k argument of combined_mcf must be a numpy array or "
                        "None");
            D = QCM::get_combined_mcf_k(vector_from_Py((PyArrayObject *)ko),
                                        false, label);
          }
          nb::list A, B;
          for (auto &a : D.A) A.append(nb_cmat(a));
          for (auto &b : D.B) B.append(nb_cmat(b));
          return nb::make_tuple(nb_cmat(D.W), A, B);
        },
        "label"_a = 0, "k"_a = nb::none(),
        "combined matrix continued fraction (W, A, B) for the cluster Green "
        "function");

  //------------------------------------------------------- lattice quantities
  m.def("CPT_Green_function",
        [](complex<double> z, nb::object k, int spin_down,
           int label) -> nb::ndarray<nb::numpy, complex<double>> {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "CPT_Green_function");
          if (ndim > 2)
            qcm_throw("Argument 2 of 'CPT_Green_function' should be of "
                      "dimension 1 or 2");
          size_t d = QCM::Green_function_dimension();
          if (ndim == 1) {
            auto g = QCM::CPT_Green_function(
                z, vector_from_Py((PyArrayObject *)ko), (bool)spin_down, label);
            return nb_cmat(g);
          } else {
            auto g = QCM::CPT_Green_function(
                z, many_vectors_from_Py((PyArrayObject *)ko), (bool)spin_down,
                label);
            return nb_cstack(g, d);
          }
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the CPT Green function at a frequency and wavevector(s)");

  m.def("CPT_Green_function_grid",
        [](int iw, int ik, int label) {
          auto g = QCM::CPT_Green_function(iw, ik, label);
          return nb_cmat(g);
        },
        "iw"_a, "ik"_a, "label"_a = 0,
        "CPT Green function at a frequency/wavevector indexed in the external "
        "grids");

  m.def("CPT_Green_function_inverse",
        [](complex<double> z, nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "CPT_Green_function_inverse");
          if (ndim != 2)
            qcm_throw("Argument 2 of 'CPT_Green_function_inverse' should be of "
                      "dimension 2");
          size_t d = QCM::Green_function_dimension();
          auto g = QCM::CPT_Green_function_inverse(
              z, many_vectors_from_Py((PyArrayObject *)ko), (bool)spin_down,
              label);
          return nb_cstack(g, d);
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the inverse CPT Green function at a frequency and "
        "wavevectors");

  m.def("periodized_Green_function",
        [](complex<double> z, nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "periodized_Green_function");
          if (ndim > 2)
            qcm_throw("Argument 2 of 'periodized_Green_function' should be of "
                      "dimension 1 or 2");
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::periodized_Green_function(z, kk, (bool)spin_down, label);
          return nb_cstack(g, QCM::reduced_Green_function_dimension());
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the periodized Green function at a frequency and "
        "wavevectors");

  m.def("band_Green_function",
        [](complex<double> z, nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "band_Green_function");
          if (ndim > 2)
            qcm_throw("Argument 2 of 'band_Green_function' should be of "
                      "dimension 1 or 2");
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::band_Green_function(z, kk, (bool)spin_down, label);
          return nb_cstack(g, QCM::reduced_Green_function_dimension());
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "periodized Green function in the band basis");

  m.def("self_energy",
        [](complex<double> z, nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "self_energy");
          if (ndim > 2)
            qcm_throw("Argument 2 of 'self_energy' should be of dimension 1 or "
                      "2");
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::self_energy(z, kk, (bool)spin_down, label);
          return nb_cstack(g, QCM::reduced_Green_function_dimension());
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "self-energy of the periodized Green function");

  m.def("periodized_Green_function_element",
        [](int r, int c, complex<double> z, nb::object k, int spin_down,
           int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "periodized_Green_function_element");
          if (ndim > 2)
            qcm_throw("Argument 4 of 'periodized_Green_function_element' should "
                      "be of dimension 1 or 2");
          auto g = QCM::periodized_Green_function_element(
              r, c, z, many_vectors_from_Py((PyArrayObject *)ko),
              (bool)spin_down, label);
          return nb_array_<complex<double>>(g.data(), {g.size()});
        },
        "r"_a, "c"_a, "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "a matrix element of the periodized Green function over wavevectors");

  m.def("V_matrix",
        [](complex<double> z, nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "V_matrix");
          if (ndim > 1)
            qcm_throw("Argument 2 of 'V_matrix' should be of dimension 1");
          auto g = QCM::V_matrix(z, vector_from_Py((PyArrayObject *)ko),
                                 (bool)spin_down, label);
          return nb_cmat(g);
        },
        "z"_a, "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the V matrix at a frequency and wavevector");

  m.def("tk",
        [](nb::object k, int spin_down,
           int label) -> nb::ndarray<nb::numpy, complex<double>> {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "tk");
          if (ndim > 2)
            qcm_throw("Argument 1 of 'tk' should be of dimension 1 or 2");
          size_t d = QCM::Green_function_dimension();
          if (ndim == 1) {
            auto g = QCM::tk(vector_from_Py((PyArrayObject *)ko),
                             (bool)spin_down, label);
            return nb_cmat(g);
          } else {
            auto g = QCM::tk(many_vectors_from_Py((PyArrayObject *)ko),
                             (bool)spin_down, label);
            return nb_cstack(g, d);
          }
        },
        "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the k-dependent one-body matrix t(k)");

  m.def("dispersion",
        [](nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "dispersion");
          if (ndim > 2)
            qcm_throw("Argument 1 of 'dispersion' should be of dimension 1 or "
                      "2");
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::dispersion(kk, (bool)spin_down, label);
          size_t N = g.size();
          size_t d = QCM::reduced_Green_function_dimension();
          vector<double> buf(N * d);
          for (size_t j = 0; j < N; j++)
            std::copy(g[j].data(), g[j].data() + d, buf.data() + j * d);
          return nb_array_<double>(buf.data(), {N, d});
        },
        "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "computes the dispersion relation (band energies)");

  m.def("epsilon",
        [](nb::object k, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "epsilon");
          if (ndim > 2)
            qcm_throw("Argument 1 of 'epsilon' should be of dimension 1 or 2");
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::epsilon(kk, (bool)spin_down, label);
          return nb_cstack(g, QCM::reduced_Green_function_dimension());
        },
        "k"_a, "spin_down"_a = 0, "label"_a = 0,
        "dispersion relation in the orbital basis");

  m.def("dos",
        [](complex<double> z, int label, int use_grid) {
          auto g = QCM::dos(z, label, (bool)use_grid);
          return nb_array_<double>(g.data(), {g.size()});
        },
        "z"_a, "label"_a = 0, "use_grid"_a = 0,
        "computes the density of states at a given frequency");

  m.def("momentum_profile",
        [](const std::string &name, nb::object k, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "momentum_profile");
          if (ndim != 2)
            qcm_throw("Argument 2 of 'momentum_profile' should be of dimension "
                      "2");
          auto g = QCM::momentum_profile(
              name, many_vectors_from_Py((PyArrayObject *)ko), label);
          return nb_array_<double>(g.data(), {g.size()});
        },
        "name"_a, "k"_a, "label"_a = 0,
        "momentum-resolved average of an operator");

  m.def("Lehmann_Green_function",
        [](nb::object k, int orb, int spin_down, int label) {
          PyObject *ko = k.ptr();
          int ndim = require_array_ndim(ko, "Lehmann_Green_function");
          if (ndim > 2)
            qcm_throw("Argument 1 of 'Lehmann_Green_function' should be of "
                      "dimension 1 or 2");
          orb -= 1;
          vector<vector3D<double>> kk;
          if (ndim == 1)
            kk.assign(1, vector_from_Py((PyArrayObject *)ko));
          else
            kk = many_vectors_from_Py((PyArrayObject *)ko);
          auto g = QCM::Lehmann_Green_function(kk, orb, (bool)spin_down, label);
          nb::list lst;
          for (auto &x : g)
            lst.append(nb::make_tuple(
                nb_array_<double>(x.first.data(), {x.first.size()}),
                nb_array_<double>(x.second.data(), {x.second.size()})));
          return lst;
        },
        "k"_a, "orb"_a = 1, "spin_down"_a = 0, "label"_a = 0,
        "Lehmann representation of the periodized Green function");

  //--------------------------------------------------------------- topological
  m.def("Berry_flux",
        [](nb::object k, int orb, int label) {
          auto kk = many_vectors_from_Py((PyArrayObject *)k.ptr());
          return QCM::Berry_flux(kk, orb, label);
        },
        "k"_a, "orb"_a = 0, "label"_a = 0,
        "Berry flux around a closed loop of wavevectors");

  m.def("Berry_curvature",
        [](nb::object k1, nb::object k2, int nk, int orb, int rec, int dir,
           int label) {
          vector3D<double> a = vector_from_Py((PyArrayObject *)k1.ptr());
          vector3D<double> b = vector_from_Py((PyArrayObject *)k2.ptr());
          auto g = QCM::Berry_curvature(a, b, nk, orb, (bool)rec, dir, label);
          return nb_array_<double>(g.data(), {(size_t)nk, (size_t)nk});
        },
        "k1"_a, "k2"_a, "nk"_a, "orb"_a = 0, "rec"_a = 0, "dir"_a = 3,
        "label"_a = 0, "Berry curvature on a 2D region of the Brillouin zone");

  m.def("monopole",
        [](nb::object k, double a, int nk, int orb, int rec, int label) {
          vector3D<double> kv = vector_from_Py((PyArrayObject *)k.ptr());
          return QCM::monopole(kv, a, nk, orb, (bool)rec, label);
        },
        "k"_a, "a"_a, "nk"_a, "orb"_a = 0, "rec"_a = 0, "label"_a = 0,
        "topological charge of a node in a Weyl semi-metal");

  //----------------------------------------------------------------- profiles
  m.def("site_and_bond_profile",
        [](int label) {
          auto prof = QCM::site_and_bond_profile(label);
          auto a = nb_array_<double>(
              reinterpret_cast<const double *>(prof.first.data()),
              {prof.first.size(), (size_t)9});
          auto b = nb_array_<complex<double>>(
              reinterpret_cast<const complex<double> *>(prof.second.data()),
              {prof.second.size(), (size_t)11});
          return nb::make_tuple(a, b);
        },
        "label"_a = 0,
        "site and bond profiles in all clusters of the repeated unit");

  //--------------------------------------------------------------------- CDMFT
  m.def("CDMFT_variational_set",
        [](nb::object v) {
          if (qcm_model->param_set == nullptr)
            qcm_throw("The parameters have not been specified yet.");
          qcm_model->param_set->CDMFT_variational_set(
              strings_from_DoublePyList(v.ptr()));
        },
        "varia"_a, "defines the set of CDMFT variational parameters (per cluster)");

  m.def("CDMFT_variational_sys_set",
        [](nb::object v) {
          if (qcm_model->param_set == nullptr)
            qcm_throw("The parameters have not been specified yet.");
          qcm_model->param_set->CDMFT_variational_sys_set(
              strings_from_DoublePyList(v.ptr()));
        },
        "varia"_a, "defines the set of CDMFT variational parameters per system (for the per-system distance)");

  m.def("CDMFT_host",
        [](nb::object freqs, nb::object weights, int label) {
          QCM::CDMFT_host(doubles_from_Py(freqs.ptr()),
                          doubles_from_Py(weights.ptr()), label);
        },
        "freqs"_a, "weights"_a, "label"_a = 0,
        "sets the CDMFT host function from frequencies and weights");

  m.def("set_CDMFT_host",
        [](int label, nb::object freqs, nb::object H, int clus, int spin_down) {
          vector<double> f = doubles_from_Py(freqs.ptr());
          vector<matrix<complex<double>>> Hv =
              complex_array3_from_Py((PyArrayObject *)H.ptr());
          QCM::set_CDMFT_host(label, f, clus, Hv, (bool)spin_down);
        },
        "label"_a, "freqs"_a, "H"_a, "clus"_a = 0, "spin_down"_a = 0,
        "sets the CDMFT host function from precomputed matrix data");

  m.def("get_CDMFT_host",
        [](int clus, int spin_down, int label) {
          auto g = QCM::get_CDMFT_host(spin_down, label);
          size_t d = qcm_model->GF_dims[clus];
          size_t N = g.size();
          vector<complex<double>> H(N * d * d);
          size_t l = 0;
          for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < d; j++)
              for (size_t k = 0; k < d; k++) H[l++] = g[i][clus](j, k);
          return nb_array_<complex<double>>(H.data(), {N, d, d});
        },
        "clus"_a = 0, "spin_down"_a = 0, "label"_a = 0,
        "retrieves the CDMFT host function (per-cluster matrices)");

  m.def("CDMFT_distance",
        [](nb::object val, int idx, int label, int by_system) {
          return QCM::CDMFT_distance(doubles_from_Py(val.ptr()), idx, label,
                                     (bool)by_system);
        },
        "p"_a, "clus"_a, "label"_a = 0, "by_system"_a = 0,
        "computes the CDMFT distance function at a variational point (per cluster, or per system if by_system)");

  m.def("CDMFT_residuals",
        [](nb::object val, int clus, int label) {
          auto r = QCM::CDMFT_residuals(doubles_from_Py(val.ptr()), clus, label);
          return nb_array_<double>(r.data(), {r.size()});
        },
        "p"_a, "clus"_a, "label"_a = 0,
        "real-valued CDMFT residual vector (for least_squares optimizers)");

  m.def("CDMFT_gradient",
        [](nb::object val, int clus, int label) {
          vector<double> p = doubles_from_Py(val.ptr());
          auto J = QCM::CDMFT_gradient(p, clus, label);
          size_t Nparams = p.size();
          size_t Nrows = Nparams ? J.size() / Nparams : 0;
          return nb_array_<double>(J.data(), {Nrows, Nparams});
        },
        "p"_a, "clus"_a, "label"_a = 0,
        "analytical Jacobian dr/dp of the CDMFT residual vector");

  m.def("CDMFT_residuals_sys",
        [](nb::object val, int sys, int label) {
          auto r = QCM::CDMFT_residuals_sys(doubles_from_Py(val.ptr()), sys, label);
          return nb_array_<double>(r.data(), {r.size()});
        },
        "p"_a, "sys"_a, "label"_a = 0,
        "system-based CDMFT residual vector (non-remixed ED hybridization)");

  m.def("CDMFT_gradient_sys",
        [](nb::object val, int sys, int label) {
          vector<double> p = doubles_from_Py(val.ptr());
          auto J = QCM::CDMFT_gradient_sys(p, sys, label);
          size_t Nparams = p.size();
          size_t Nrows = Nparams ? J.size() / Nparams : 0;
          return nb_array_<double>(J.data(), {Nrows, Nparams});
        },
        "p"_a, "sys"_a, "label"_a = 0,
        "system-based Jacobian dr/dp of the CDMFT residual vector");

  //------------------------------------------------------------------ printing
  m.def("print_model",
        [](const std::string &file) { QCM::print_model(file); }, "file"_a,
        "prints a description of the model into a file");

  m.def("print_statistics",
        []() {
          cout << "Number of recycled evaluations of the cluster Green "
                  "function : "
               << STATS.n_GF_recycled << endl;
          cout << "Number of computed evaluations of the cluster Green "
                  "function : "
               << STATS.n_GF_computed << endl;
        },
        "prints run statistics on the screen");

  //----------------------------------------------------------- frequency grids
  m.def("frequency_grid",
        [](nb::object freqs, nb::object weights) {
          grid_freqs = doubles_from_Py(freqs.ptr());
          grid_weights = doubles_from_Py(weights.ptr());
        },
        "freqs"_a, "weights"_a,
        "sets the fixed imaginary-frequency grid for discrete integrals");

  m.def("set_wavevector_grid",
        [](int nkx, int nky, int nkz) {
          if (nkx <= 0 || nky <= 0 || nkz <= 0)
            qcm_throw("set_wavevector_grid: nkx, nky and nkz must all be "
                      "strictly positive");
          QCM::set_wavevector_grid(nkx, nky, nkz);
        },
        "nkx"_a, "nky"_a, "nkz"_a,
        "sets the number of wavevectors along each direction of the fixed grid");
}

#endif
