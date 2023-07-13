#ifndef Heisenberg_operator_h
#define Heisenberg_operator_h

#include "Hermitian_operator.hpp"

//! Hermitian operator representing a Heisenberg spin interaction
struct Heisenberg_operator : Hermitian_operator
{
  vector<matrix_element<double>> elements; //!< matrix elements
  map<index_pair, double> element_map; //!< temporary map form of elements, for checks
  char dir; //!< specific type: 'X', 'Y', 'Z' or 'H' (default) for the sum of all three

  Heisenberg_operator(const string &_name, shared_ptr<model> _the_model, const vector<matrix_element<double>>& _elements, char _dir='H');
  void set_target(vector<bool> &in_bath);
  shared_ptr<HS_Hermitian_operator> build_HS_operator(sector sec, bool complex_Hilbert_space);
  void set_hopping_matrix(double value, matrix<double>& tc, bool spin_down, int sys_mixing){}
  void set_hopping_matrix(double value, matrix<Complex>& tc, bool spin_down, int sys_mixing){}
  double average_from_GF(matrix<Complex>& Gave, bool spin_down){return NAN;}
  void print(ostream& fout);
  vector<matrix_element<Complex>> matrix_elements();
  string type() {return string("Heisenberg");}
  void multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B);
  void multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B);
};


#endif
