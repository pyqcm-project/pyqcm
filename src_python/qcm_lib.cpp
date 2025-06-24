#define PY_ARRAY_UNIQUE_SYMBOL QCM_ARRAY_API

#include "qcm_ED_wrap.hpp"
#include "qcm_wrap.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

//==============================================================================
// doc string
const char *qcm_help =
    R"{(
qcm performs many tasks associated with quantum cluster methods. The following
functions are available (type help(<function> or <function>? in ipython) for help on any of them):
){";

static PyMethodDef methods[] = {
    //-------------------- QCM_ED
    //--------------------------------------------------------------
    {"cluster_averages", cluster_averages_python, METH_VARARGS, cluster_averages_help},
    {"ED_complex_HS", ED_complex_HS_python, METH_VARARGS, ED_complex_HS_help},
    {"density_matrix", density_matrix_python, METH_VARARGS, density_matrix_help},
    {"fidelity", fidelity_python, METH_VARARGS, fidelity_help},
    {"Green_function_average", Green_function_average_python, METH_VARARGS, Green_function_average_help},
    {"Green_function_density", Green_function_density_python, METH_VARARGS, Green_function_density_help},
    {"Green_function_dimensionC", Green_function_dimensionC_python, METH_VARARGS, Green_function_dimensionC_help},
    {"Green_function_solveC", Green_function_solveC_python, METH_VARARGS, Green_function_solveC_help},
    {"Green_function", Green_function_python, METH_VARARGS, Green_function_help},
    {"ground_state_solve", ground_state_solve_python, METH_VARARGS, ground_state_solve_help},
    {"hopping_matrix", hopping_matrix_python, METH_VARARGS, hopping_matrix_help},
    {"interactions", interactions_python, METH_VARARGS, interactions_help},
    {"hybridization_functionC", hybridization_functionC_python, METH_VARARGS, hybridization_functionC_help},
    {"hybridization", hybridization_python, METH_VARARGS, hybridization_help},
    {"matrix_elements", matrix_elements_python, METH_VARARGS, matrix_elements_help},
    {"mixingC", mixingC_python, METH_VARARGS, mixingC_help},
    {"model_sizeC", model_sizeC_python, METH_VARARGS, model_sizeC_help},
    {"new_model_instanceC", new_model_instanceC_python, METH_VARARGS, new_model_instanceC_help},
    {"new_model", new_model_python, METH_VARARGS, new_model_help},
    {"new_operator_complex", new_operator_complex_python, METH_VARARGS, new_operator_complex_help},
    {"new_operator", new_operator_python, METH_VARARGS, new_operator_help},
    {"parametersC", parametersC_python, METH_VARARGS, parametersC_help},
    {"print_graph", print_graph_python, METH_VARARGS, print_graph_help},
    {"print_models", print_models_python, METH_VARARGS, print_models_help},
    {"print_wavefunction", print_wavefunction_python, METH_VARARGS, print_wavefunction_help},
    {"qmatrix", qmatrix_python, METH_VARARGS, qmatrix_help},
    {"read_instance", read_instance_python, METH_VARARGS, read_instance_help},
    {"self_energyC", self_energyC_python, METH_VARARGS, self_energyC_help},
    {"set_global_parameterC", set_global_parameterC_python, METH_VARARGS, set_global_parameterC_help},
    {"susceptibility_poles", susceptibility_poles_python, METH_VARARGS, susceptibility_poles_help},
    {"susceptibility", susceptibility_python, METH_VARARGS, susceptibility_help},
    {"write_instance_to_file", write_instance_to_file_python, METH_VARARGS, write_instance_to_file_help},
    {"write_instance", write_instance_python, METH_VARARGS, write_instance_help},
    //-------------------- QCM
    //--------------------------------------------------------------
    {"add_cluster", add_cluster_python, METH_VARARGS, add_cluster_help},
    {"anomalous_operator", (PyCFunction)anomalous_operator_python, METH_VARARGS | METH_KEYWORDS, anomalous_operator_help},
    {"averages", averages_python, METH_VARARGS, averages_help},
    {"band_Green_function", band_Green_function_python, METH_VARARGS, band_Green_function_help},
    {"Berry_curvature", Berry_curvature_python, METH_VARARGS, Berry_curvature_help},
    {"Berry_flux", Berry_flux_python, METH_VARARGS, Berry_flux_help},
    {"CDMFT_distance", CDMFT_distance_python, METH_VARARGS, CDMFT_distance_help},
    {"CDMFT_host", CDMFT_host_python, METH_VARARGS, CDMFT_host_help},
    {"set_CDMFT_host", set_CDMFT_host_python, METH_VARARGS, set_CDMFT_host_help},
    {"CDMFT_variational_set", CDMFT_variational_set_python, METH_VARARGS, CDMFT_variational_set_help},
    {"cluster_Green_function", cluster_Green_function_python, METH_VARARGS, cluster_Green_function_help},
    {"cluster_hopping_matrix", cluster_hopping_matrix_python, METH_VARARGS, cluster_hopping_matrix_help},
    {"cluster_info", cluster_info_python, METH_VARARGS, cluster_info_help},
    {"cluster_self_energy", cluster_self_energy_python, METH_VARARGS, cluster_self_energy_help},
    {"complex_HS", complex_HS_python, METH_VARARGS, complex_HS_help},
    {"CPT_Green_function_inverse", CPT_Green_function_inverse_python, METH_VARARGS, CPT_Green_function_inverse_help},
    {"CPT_Green_function", CPT_Green_function_python, METH_VARARGS, CPT_Green_function_help},
    {"current_operator", (PyCFunction)current_operator_python, METH_VARARGS | METH_KEYWORDS, current_operator_help},
    {"density_wave", (PyCFunction)density_wave_python, METH_VARARGS | METH_KEYWORDS, density_wave_help},
    {"dispersion", dispersion_python, METH_VARARGS, dispersion_help},
    {"dos", dos_python, METH_VARARGS, dos_help},
    {"epsilon", epsilon_python, METH_VARARGS, epsilon_help},
    {"erase_model_instance", erase_model_instance_python, METH_VARARGS, erase_model_instance_help},
    {"explicit_operator", (PyCFunction)explicit_operator_python, METH_VARARGS | METH_KEYWORDS, explicit_operator_help},
    {"get_global_parameter", get_global_parameter_python, METH_VARARGS, get_global_parameter_help},
    {"get_CDMFT_host", get_CDMFT_host_python, METH_VARARGS, get_CDMFT_host_help},
    {"great_reset", great_reset_python, METH_VARARGS, great_reset_help},
    {"Green_function_dimension", Green_function_dimension_python, METH_VARARGS, Green_function_dimension_help},
    {"Green_function_solve", Green_function_solve_python, METH_VARARGS, Green_function_solve_help},
    {"ground_state", ground_state_python, METH_VARARGS, ground_state_help},
    {"hopping_operator", (PyCFunction)hopping_operator_python, METH_VARARGS | METH_KEYWORDS, hopping_operator_help},
    {"hybridization_function", hybridization_function_python, METH_VARARGS, hybridization_function_help},
    {"interaction_operator", (PyCFunction)interaction_operator_python, METH_VARARGS | METH_KEYWORDS, interaction_operator_help},
    {"kinetic_energy", kinetic_energy_python, METH_VARARGS, kinetic_energy_help},
    {"lattice_model", lattice_model_python, METH_VARARGS, lattice_model_help},
    {"Lehmann_Green_function", Lehmann_Green_function_python, METH_VARARGS, Lehmann_Green_function_help},
    {"mixing", mixing_python, METH_VARARGS, mixing_help},
    {"model_size", model_size_python, METH_VARARGS, model_size_help},
    {"model_is_closed", model_is_closed_python, METH_VARARGS, model_is_closed_help},
    {"momentum_profile", momentum_profile_python, METH_VARARGS, momentum_profile_help},
    {"monopole", monopole_python, METH_VARARGS, monopole_help},
    {"new_model_instance", new_model_instance_python, METH_VARARGS, new_model_instance_help},
    {"parameter_set", parameter_set_python, METH_VARARGS, parameter_set_help},
    {"parameters", parameters_python, METH_VARARGS, parameters_help},
    {"instance_parameters", instance_parameters_python, METH_VARARGS, instance_parameters_help},
    {"periodized_Green_function_element", periodized_Green_function_element_python, METH_VARARGS, periodized_Green_function_element_help},
    {"periodized_Green_function", periodized_Green_function_python, METH_VARARGS, periodized_Green_function_help},
    {"potential_energy", potential_energy_python, METH_VARARGS, potential_energy_help},
    {"Potthoff_functional", Potthoff_functional_python, METH_VARARGS, Potthoff_functional_help},
    {"print_model", (PyCFunction)print_model_python, METH_VARARGS | METH_KEYWORDS, print_model_help},
    {"print_options", print_options_python, METH_VARARGS, print_options_help},
    {"print_parameter_set", print_parameter_set_python, METH_VARARGS, print_parameter_set_help},
    {"print_statistics", print_statistics_python, METH_VARARGS, print_statistics_help},
    {"projected_Green_function", projected_Green_function_python, METH_VARARGS, projected_Green_function_help},
    {"reduced_Green_function_dimension", reduced_Green_function_dimension_python, METH_VARARGS, reduced_Green_function_dimension_help},
    {"self_energy", self_energy_python, METH_VARARGS, self_energy_help},
    {"set_basis", set_basis_python, METH_VARARGS, set_basis_help},
    {"set_global_parameter", set_global_parameter_python, METH_VARARGS, set_global_parameter_help},
    {"set_multiplier", set_multiplier_python, METH_VARARGS, set_multiplier_help},
    {"set_parameter", set_parameter_python, METH_VARARGS, set_parameter_help},
    {"set_parameters", set_parameters_python, METH_VARARGS, set_parameters_help},
    {"set_target_sectors", set_target_sectors_python, METH_VARARGS, set_target_sectors_help},
    {"site_and_bond_profile", site_and_bond_profile_python, METH_VARARGS, site_and_bond_profile_help},
    {"spatial_dimension", spatial_dimension_python, METH_VARARGS, spatial_dimension_help},
    {"spectral_average", spectral_average_python, METH_VARARGS, spectral_average_help},
    {"switch_cluster_model", switch_cluster_model_python, METH_VARARGS, switch_cluster_model_help},
    {"tk", tk_python, METH_VARARGS, tk_help},
    {"V_matrix", V_matrix_python, METH_VARARGS, V_matrix_help},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef qcm = {
    PyModuleDef_HEAD_INIT,
    "qcm",    // name of module
    qcm_help, // module documentation, may be NULL */
    -1, // size of per-interpreter state of the module, or -1 if the module
   // keeps state in global variables. */
    methods};

PyMODINIT_FUNC PyInit_qcm(void) {
  import_array();
  QCM::qcm_init();

  PyObject *module = PyModule_Create(&qcm);
  if (module == NULL)
    return NULL;

  qcm_Error = PyErr_NewException("qcm.error", NULL, NULL);
  Py_INCREF(qcm_Error);
  if (PyModule_AddObject(module, "error", qcm_Error) < 0) {
    Py_XDECREF(qcm_Error);
    Py_CLEAR(qcm_Error);
    Py_DECREF(module);
    return NULL;
  };

  qcm_ED_Error = PyErr_NewException("qcm_ED.error", NULL, NULL);
  Py_INCREF(qcm_ED_Error);
  if (PyModule_AddObject(module, "error", qcm_ED_Error) < 0) {
    Py_XDECREF(qcm_ED_Error);
    Py_CLEAR(qcm_ED_Error);
    Py_DECREF(module);
    return NULL;
  };

  return module;
}

/**
  initialization of the library. Must be called before the first call to the
  library.
  */
void QCM::qcm_init() {
  qcm_model = make_shared<lattice_model>();
  // Initialize environment variable
  // May be overwrite by the user
  setenv("CUBACORES", "0", 0);
  // setenv("OMP_NUM_THREADS","1",0);
  QCM::global_parameter_init();

#ifdef _OPENMP
  omp_set_max_active_levels(2);
  cout << "Number of OpenMP threads = " << omp_get_max_threads() << endl;
#endif
}

void qcm_catch(const std::string &s) { PyErr_SetString(qcm_Error, s.c_str()); }
void qcm_ED_catch(const std::string &s) {
  PyErr_SetString(qcm_Error, s.c_str());
}
