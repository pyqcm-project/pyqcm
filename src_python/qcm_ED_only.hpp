//==============================================================================
const char *print_options_help =
    R"{(
Prints the list of global options and parameters on the screen
argument:
1. int to_file : 0 -> prints to screen. 1 -> prints to latex. 2-> prints to RST
){";
//------------------------------------------------------------------------------
static PyObject *print_options_python(PyObject *self, PyObject *args) {
  int to_file = 0;
  ED::global_parameter_init();
  try {
    if (!PyArg_ParseTuple(args, "|i", &to_file))
      qcm_ED_throw("failed to read parameters in call to print_model (python)");
  } catch (const string &s) {
    qcm_ED_catch(s);
  }

  print_options(to_file);
  return Py_BuildValue("");
}
