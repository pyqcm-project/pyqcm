#ifndef interaction_operator_h
#define interaction_operator_h

#include "Hermitian_operator.hpp"
#include "HS_interaction_operator.hpp"

//! Density-density interaction operator
struct interaction_operator : Hermitian_operator
{
  vector<matrix_element<double>> elements; //!< matrix elements
  map<index_pair, double> element_map; //!< temporary map form of elements, for checks

  interaction_operator(const string &_name, shared_ptr<model> the_model, const vector<matrix_element<double>>& _elements);
  void check_spin_symmetry();
  void set_target(vector<bool> &in_bath);
  shared_ptr<HS_Hermitian_operator> build_HS_operator(sector sec, bool complex_Hilbert_space);
  void set_hopping_matrix(double value, matrix<double>& tc, bool spin_down, int sys_mixing){}
  void set_hopping_matrix(double value, matrix<Complex>& tc, bool spin_down, int sys_mixing){}
  double average_from_GF(matrix<Complex>& Gave, bool spin_down){return NAN;}
  void diag(vector<double> &d, double z);
  void print(ostream& fout);
  vector<matrix_element<Complex>> matrix_elements();
  string type() {return string("interaction");}
  void multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B);
  void multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B);
};

//==============================================================================
// implementation

/**
 Constructor from name and matrix elements
 @param _name   name of the operator
 @param _the_model   model
 @param _elements   nonzero one-body matrix elements
 */
interaction_operator::interaction_operator(const string &_name, shared_ptr<model> _the_model, const vector<matrix_element<double>>& _elements)
: Hermitian_operator(_name, _the_model)
{
  is_interaction = true;
  is_factorizable = true;
  // fixes the matrix elements regarding symmetry
  elements.clear();
  elements.reserve(_elements.size());
  for(auto& x : _elements){
    if(x.r <= x.c) elements.push_back(matrix_element<double>(x.r, x.c, x.v));
    else elements.push_back(matrix_element<double>(x.c, x.r, x.v));
  }
  
  // checks
  for(auto& x : elements) element_map[{x.r, x.c}] = x.v;
  set_target(the_model->in_bath);
  check_spin_symmetry();
  the_model->group->check_invariance<double>(element_map, name, true);
  element_map.clear();

}




/**
 determines whether the operator is symmetric under the exchange of up and down spins
 */
void interaction_operator::check_spin_symmetry()
{
  if(mixing&HS_mixing::spin_flip) return;
  for(auto &x : element_map){
    index_pair p = x.first;
    if(p.r < the_model->n_orb) p.r += the_model->n_orb;
    else p.r -= the_model->n_orb;
    if(p.c < the_model->n_orb) p.c += the_model->n_orb;
    else p.c -= the_model->n_orb;
    index_pair p2 = p.swap();
    auto y = element_map.find(p);
    auto y2 = element_map.find(p2);
    if((y == element_map.end() or y->second != x.second) and (y2 == element_map.end() or y2->second != x.second)){
      mixing |= HS_mixing::up_down;
      break;
    };
  }
}





/**
 set the target of an operator
 1 : cluster
 2 : bath only
 3 : hybridization
 @param in_bath vector of bool defining the status of each site
 */
void interaction_operator::set_target(vector<bool> &in_bath){
  this->target = 1;
  for(auto& x : this->elements){
    if(in_bath[x.r] or in_bath[x.c])
      qcm_ED_throw("interaction operator "+this->name+" must be defined on the cluster only");
  }
}




/**
 returns a pointer to, and constructs the associated HS operator in the sector with basis B.
 */
shared_ptr<HS_Hermitian_operator> interaction_operator::build_HS_operator(sector sec, bool complex_Hilbert_space)
{
  return make_shared<HS_interaction_operator>(the_model, name, sec, elements);
}




/**
 prints definition to a file
 @param fout output stream
 */
void interaction_operator::print(ostream& fout)
{
  fout << "\ninteraction operator " << name << "\t (target " << target << ")" << endl;
  for(auto& x : elements) fout << x.r+1 << '\t' << x.c+1 << '\t' << x.v << endl;
}

/**
 returns a list of complexified matrix elements
 */
vector<matrix_element<Complex>> interaction_operator::matrix_elements()
{
  vector<matrix_element<Complex>> celem(elements.size());
  for(int i=0; i<elements.size(); i++){
    auto el = elements[i];
    celem[i] = {el.r, el.c, Complex(el.v)};
  }
  return celem;
}


void interaction_operator::multiply_add_OTF(const vector<double> &x, vector<double> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  for(size_t I=0; I<B->dim; ++I){
    for(auto &E : elements){
      uint64_t mask = binary_state::mask(E.r,B->L) + binary_state::mask(E.c,B->L);
      if((B->bin(I).b & mask) == mask) y[I] += z*E.v*x[I];
    }
  }
}

void interaction_operator::multiply_add_OTF(const vector<Complex> &x, vector<Complex> &y, double z, shared_ptr<ED_mixed_basis> B)
{
  for(size_t I=0; I<B->dim; ++I){
    for(auto &E : elements){
      uint64_t mask = binary_state::mask(E.r,B->L) + binary_state::mask(E.c,B->L);
      if((B->bin(I).b & mask) == mask) y[I] += z*E.v*x[I];
    }
  }
}

#endif /* interaction_operator_h */
