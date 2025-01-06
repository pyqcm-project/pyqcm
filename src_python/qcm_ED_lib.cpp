#define PY_ARRAY_UNIQUE_SYMBOL QCM_ARRAY_API

#include "qcm_ED_only.hpp"
#include "qcm_ED_wrap.hpp"

//==============================================================================
// doc string
const char *qcm_ED_help =
    R"(
qcm performs many tasks associated with quantum cluster methods. The following
functions are available (type help(<function> or <function>? in ipython) for help on any of them):
fidelity(string name, dict values1, dict values2, string sector)
Green_function_dimension(int label = 0)
Green_function_solve(int label = 0)
Green_function(complex w, bool spin_down, int label = 0)
cluster_averages(int label = 0)
ground_state_solve(int label = 0)
hopping_matrix(bool spin_down, int label = 0)
interactions(int label = 0)
hybridization_function(complex w, bool spin_down, int label = 0)
mixing(int label = 0)
model_size(string name)
new_model(string name, int nc, int nb, array symmetry_generators)
new_model_instance(string cluster_name, dict values, string sectors, int label = 0)
new_operator_complex(string cluster_name, string operator_name, string type, matrix_elements)
new_operator(string cluster_name, string operator_name, string type, matrix_elements)
print_models()
self_energy(complex w, bool spin_down, int label = 0)
set_global_parameter(string name, value)
susceptibility_poles(string operator_name, int label = 0)
susceptibility(string operator_name, frequencies, int label = 0)
)";

static PyMethodDef methods[] = {
    //-------------------- QMC_ED
    //--------------------------------------------------------------
    {"ED_complex_HS", ED_complex_HS_python, METH_VARARGS, ED_complex_HS_help},
    {"fidelity", fidelity_python, METH_VARARGS, fidelity_help},
    {"Green_function_dimensionC", Green_function_dimensionC_python,
     METH_VARARGS, Green_function_dimensionC_help},
    {"Green_function_solveC", Green_function_solveC_python, METH_VARARGS,
     Green_function_solveC_help},
    {"Green_function", Green_function_python, METH_VARARGS,
     Green_function_help},
    {"cluster_averages", cluster_averages_python, METH_VARARGS,
     cluster_averages_help},
    {"ground_state_solve", ground_state_solve_python, METH_VARARGS,
     ground_state_solve_help},
    {"hopping_matrix", hopping_matrix_python, METH_VARARGS,
     hopping_matrix_help},
    {"interactions", interactions_python, METH_VARARGS, interactions_help},
    {"hybridization", hybridization_python, METH_VARARGS, hybridization_help},
    {"hybridization_functionC", hybridization_functionC_python, METH_VARARGS,
     hybridization_functionC_help},
    {"matrix_elements", matrix_elements_python, METH_VARARGS,
     matrix_elements_help},
    {"mixingC", mixingC_python, METH_VARARGS, mixingC_help},
    {"model_sizeC", model_sizeC_python, METH_VARARGS, model_sizeC_help},
    {"new_model_instanceC", new_model_instanceC_python, METH_VARARGS,
     new_model_instanceC_help},
    {"new_model", new_model_python, METH_VARARGS, new_model_help},
    {"new_operator_complex", new_operator_complex_python, METH_VARARGS,
     new_operator_complex_help},
    {"new_operator", new_operator_python, METH_VARARGS, new_operator_help},
    {"parametersC", parametersC_python, METH_VARARGS, parametersC_help},
    {"print_graph", print_graph_python, METH_VARARGS, print_graph_help},
    {"print_models", print_models_python, METH_VARARGS, print_models_help},
    {"print_options", print_options_python, METH_VARARGS, print_options_help},
    {"print_wavefunction", print_wavefunction_python, METH_VARARGS,
     print_wavefunction_help},
    {"qmatrix", qmatrix_python, METH_VARARGS, qmatrix_help},
    {"read_instance", read_instance_python, METH_VARARGS, read_instance_help},
    {"self_energyC", self_energyC_python, METH_VARARGS, self_energyC_help},
    {"set_global_parameterC", set_global_parameterC_python, METH_VARARGS,
     set_global_parameterC_help},
    {"susceptibility_poles", susceptibility_poles_python, METH_VARARGS,
     susceptibility_poles_help},
    {"susceptibility", susceptibility_python, METH_VARARGS,
     susceptibility_help},
    {"write_instance_to_file", write_instance_to_file_python, METH_VARARGS,
     write_instance_to_file_help},
    {"write_instance", write_instance_python, METH_VARARGS,
     write_instance_help},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef qcm_ED = {
    PyModuleDef_HEAD_INIT, "qcm_ED", /* name of module */
    qcm_ED_help,                     /* module documentation, may be NULL */
    -1, /* size of per-interpreter state of the module,
         or -1 if the module keeps state in global variables. */
    methods};

PyMODINIT_FUNC PyInit_qcm_ED(void) {
  import_array();
  ED::qcm_ED_init();
  PyObject *module = PyModule_Create(&qcm_ED);

  qcm_ED_Error = PyErr_NewException("qcm_ED.error", NULL, NULL);
  Py_INCREF(qcm_ED_Error);
  PyModule_AddObject(module, "error", qcm_ED_Error);

  return module;
}

void qcm_ED_catch(const std::string &s) {
  PyErr_SetString(qcm_ED_Error, s.c_str());
}

/**
 initialization
 */
void ED::qcm_ED_init() {
  // Initialize environment variable
  // May be overwrite by the user
  setenv("CUBACORES", "0", 0);
  if (std::strlen(std::getenv("OPENBLAS_NUM_THREADS")) == 0)
    setenv("OPENBLAS_NUM_THREADS", "1", 0);
  if (std::strlen(std::getenv("BLIS_NUM_THREADS")) == 0)
    setenv("BLIS_NUM_THREADS", "1", 0);

  global_parameter_init();
}
