#ifndef general_interaction_operator_h
#define general_interaction_operator_h

#include "Hermitian_operator.hpp"
#include "HS_general_interaction_operator.hpp"

template<typename op_field>
struct general_interaction_operator : Hermitian_operator
{
  vector<matrix_element<op_field>> elements; //!< matrix elements

  general_interaction_operator(const string &_name, shared_ptr<model> the_model, const vector<matrix_element<op_field>>& _elements);
  void check_spin_symmetry();
  // void diag(vector<double> &d, double z);

  double average_from_GF(matrix<Complex>& Gave, bool spin_down);
  shared_ptr<HS_Hermitian_operator> build_HS_operator(sector sec, bool complex_Hilbert_space);
  string type();
  vector<matrix_element<Complex>> matrix_elements();
  void multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B);
  void multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B);
  void print(ostream& fout);
  void set_hopping_matrix(double value, matrix<Complex>& tc, bool spin_down, int sys_mixing);
  void set_hopping_matrix(double value, matrix<double>& tc, bool spin_down, int sys_mixing);
  void set_target(vector<bool> & in_bath);
};

//==============================================================================
// implementation of general_interaction_operator

/**
 Constructor from name and matrix elements
 @param _name   name of the operator
 @param _the_model   model
 @param _elements   nonzero one-body matrix elements
 */
template<typename op_field>
general_interaction_operator<op_field>::general_interaction_operator(const string &_name, shared_ptr<model> _the_model, const vector<matrix_element<op_field>>& _elements)
: Hermitian_operator(_name, _the_model), elements(_elements)
{
  set_target(the_model->in_bath);
  is_interaction = true;
  // checks on _elements not done here. Should be done.
}



template<typename op_field>
double general_interaction_operator<op_field>::average_from_GF(matrix<Complex>& Gave, bool spin_down){return NAN;}



/**
 returns a pointer to, and constructs the associated HS operator in the sector with basis B.
 */
template<typename op_field>
shared_ptr<HS_Hermitian_operator>  general_interaction_operator<op_field>::build_HS_operator(sector sec, bool complex_Hilbert_space)
{
  if(complex_Hilbert_space) return make_shared<HS_general_interaction_operator<Complex>>(the_model, name, sec, elements);
  else return make_shared<HS_general_interaction_operator<double>>(the_model, name, sec, elements);
}



template<typename op_field>
string general_interaction_operator<op_field>::type() {return string("general_interaction");}




/**
 returns a list of complexified matrix elements. Just a dummy program for interaction terms. Returns an empty list.
 */
template<typename op_field>
vector<matrix_element<Complex>> general_interaction_operator<op_field>::matrix_elements()
{
  return vector<matrix_element<Complex>>(0);
}




template<typename op_field>
void general_interaction_operator<op_field>::multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  qcm_ED_throw("on the fly computation impossible with a general interaction operator");
}




template<typename op_field>
void general_interaction_operator<op_field>::multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  qcm_ED_throw("on the fly computation impossible with a general interaction operator");
}



/**
 prints definition to a file
 @param fout output stream
 */
template<typename op_field>
void general_interaction_operator<op_field>::print(ostream& fout)
{
  fout << "\ninteraction operator " << name << "\t (target " << target << ")" << endl;
  for(auto& x : elements) fout << x << endl;
}



template<typename op_field>
void general_interaction_operator<op_field>::set_hopping_matrix(double value, matrix<double>& tc, bool spin_down, int sys_mixing){return;}


template<typename op_field>
void general_interaction_operator<op_field>::set_hopping_matrix(double value, matrix<Complex>& tc, bool spin_down, int sys_mixing){return;}


template<typename op_field>
void general_interaction_operator<op_field>::set_target(vector<bool> &in_bath){this->target = 1;}


/**
 determines whether the operator is symmetric under the exchange of up and down spins
 */
template<typename op_field>
void general_interaction_operator<op_field>::check_spin_symmetry()
{
  return;
}



#endif
