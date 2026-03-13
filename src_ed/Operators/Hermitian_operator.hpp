#ifndef Hermitian_operator_h
#define Hermitian_operator_h

#include <mutex>
#include <vector>
#include <map>
#include <string>

#include "matrix.hpp"
#include "qcm_ED.hpp"
#include "sector.hpp"
#include "ED_basis.hpp"
#include "HS_Hermitian_operator.hpp"
#include "matrix_element.hpp"
#include "state.hpp"
#include "model.hpp"
#include <algorithm>

enum op_type {one_body, anomalous, interaction, Hund, Heisenberg, X, Y, Z, general_interaction};

//! Abstract class representing a Hermitian operator, i.e. a term in the Hamiltonian
struct Hermitian_operator
{
  bool is_active; //!< true is the operator is active (nonzero)
  bool is_complex; //!< true is the operator has complex matrix elements, which requires a complex Hilbert space
  bool is_factorizable; // true if the operator is a tensor product (spins up and down) or diagonal
  bool is_interaction; //!< true if the operator is an interaction
  double nambu_correction_full; //!< Nambu correction to the averages in the full Nambu mixing case
  double nambu_correction; //!< Nambu correction to the averages in the restricted Nambu mixing case
  double norm; //!< multiplicative factor on the average (e.g. to divide by the number of sites)
  int mixing; //!< mixing caused by this operator (default = normal)
  mutable std::mutex hs_op_mutex; //!< protects HS_operator map from concurrent insertions
  map<sector, shared_ptr<HS_Hermitian_operator>> HS_operator; //!< Hilbert space realizations, organized by sector
  shared_ptr<model> the_model; //!< backtrace to the model
  size_t target; //!< target of the operator: 1:cluster, 2:bath, or 3:hybrid (binary coding).
  string name; //!< name of the operator
 
  Hermitian_operator(const string &_name, shared_ptr<model> _the_model)
  : name(_name), the_model(_the_model)
  {
    target = 0;
    norm = 1.0/the_model->n_sites;
    if(name == "mu") norm *= -1.0;
    is_complex = false;
    is_interaction = false;
    mixing = HS_mixing::normal;
    nambu_correction=0.0;
    nambu_correction_full=0.0;
    is_active = false;
    is_factorizable = false;
  }


  virtual double average_from_GF(matrix<Complex>& Gave, bool spin_down) = 0;
  virtual shared_ptr<HS_Hermitian_operator> build_HS_operator(sector sec, bool complex_Hilbert_space) = 0;
  virtual string type() = 0;
  virtual vector<matrix_element<Complex>> matrix_elements() = 0;
  virtual void multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B) = 0;
  virtual void multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B) = 0;
  virtual void print(ostream& fout) = 0;
  virtual void set_hopping_matrix(double value, matrix<Complex>& tc, bool spin_down, int sys_mixing) = 0;
  virtual void set_hopping_matrix(double value, matrix<double>& tc, bool spin_down, int sys_mixing) = 0;
  virtual void set_target(vector<bool> & in_bath) = 0;

  bool shift_label(size_t label, size_t &shifted_label, bool bath, bool nambu, bool spin_down);

  template<typename HS_field>
  void expectation_value(const state<HS_field> &gs, double &average, double &average2){
    if(Hamiltonian_format == H_FORMAT::H_format_onthefly){
      auto B = the_model->provide_basis(gs.sec);
      vector<HS_field> tmp_gs(B->dim);
      multiply_add_OTF(gs.psi, tmp_gs, 1.0, B);
      average += gs.weight*real(tmp_gs*gs.psi);
      average2 += gs.weight*norm2(tmp_gs);
    }
    else{
      HS_operator.at(gs.sec)->expectation_value(gs, average, average2);
    }
  }

  template<typename HS_field>
  void dense_form(matrix<HS_field> &h, double z, sector sec){
    auto B = the_model->provide_basis(sec);
    for(int i=0; i<B->dim; i++){
      vector<HS_field> x(B->dim);
      vector<HS_field> y(B->dim);
      x[i] = 1.0;
      multiply_add_OTF(x, y, z, B);
      for(int j=0; j<B->dim; j++) h(i,j) += y[j];
    }
  }
};

#endif
