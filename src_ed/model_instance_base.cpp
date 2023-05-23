#include <fstream>
#include "model_instance_base.hpp"
#include "Operators/Hermitian_operator.hpp"

//==============================================================================
// implementation of model_instance_base

model_instance_base::model_instance_base(size_t _label, shared_ptr<model> _the_model, const map<string,double> _value, const string &_sectors)
: label(_label), value(_value), the_model(_the_model), mixing(0),
  gs_solved(false), gf_read(false), hopping_solved(false), gf_solved(false), complex_Hilbert(false),
  E0(0.0), SEF_bath(0.0), GS_energy(0.0), is_correlated(false)
{
  // checking that the values are associated with actual operators, and setting the mixing field
  bool nonzero_anomalous = false;
  if(value.size() == 0) qcm_ED_throw("ED model instance (label = "+to_string(_label)+") for model "+_the_model->name+" has no parameters!");

  static bool first_instance = true;

  GF_solver = global_bool("continued_fraction") ? GF_format_CF : GF_format_BL;
  
  if(first_instance){
    first_instance = false;
    auto S = global_char("Hamiltonian_format");
    if(S == 'O') Hamiltonian_format = H_FORMAT::H_format_ops;
    else if(S == 'S') Hamiltonian_format = H_FORMAT::H_format_csr;
    else if(S == 'N') Hamiltonian_format = H_FORMAT::H_format_onthefly;
    else if(S == 'F') Hamiltonian_format = H_FORMAT::H_format_factorized;
    else if(S == 'E') Hamiltonian_format = H_FORMAT::H_format_eigen;
    else qcm_ED_throw("Hamiltonian_format is not one of : N, S, O, F, E");
  }

  for(auto& v : value){
    if((the_model->term.find(v.first) == the_model->term.end()) and global_bool("verb_warning")){
      cout << "ED WARNING : operator " << v.first << " does not exist in cluster model " << the_model->name << endl;
    }
    Hermitian_operator& op = *the_model->term.at(v.first);
    op.is_active = true;
    if(op.mixing & HS_mixing::anomalous) nonzero_anomalous = true;
    mixing |= op.mixing;
    if(op.is_interaction) is_correlated = true;
  }
  if(mixing>5) mixing = mixing&HS_mixing::full;
  if(mixing==5) mixing = mixing&HS_mixing::anomalous;

  // makes sure the mixing stays the same as the first instance
  if(the_model->mixing == -1) the_model->mixing = mixing;
  else if((mixing | the_model->mixing) != the_model->mixing){
    qcm_ED_throw("The new instance mixing state of model "+the_model->name+" ("+to_string(mixing)+") is not compatible with the first instance mixing ("+to_string(the_model->mixing)+")");
  }
  else mixing = the_model->mixing;

  spin_down = false;
  n_mixed = 1;
  if(mixing & HS_mixing::anomalous) n_mixed *= 2;
  if(mixing & HS_mixing::spin_flip) n_mixed *= 2;
  dim_GF = the_model->n_sites*n_mixed;
  // checking whether anomalous operators are nonzero in the full Nambu case
  if(n_mixed==4 and  nonzero_anomalous == false)
    qcm_ED_throw("cluster anomalous operators cannot all be zero in the full Nambu case.");
  
  // checking the factorized option
  bool factorizable = false;
  if(Hamiltonian_format == H_FORMAT::H_format_factorized and the_model->group->g != 1)
    qcm_ED_throw("the Hamiltonian format 'factorized' is not compatible with a nontrivial point group symmetry");
  if(Hamiltonian_format == H_FORMAT::H_format_factorized and (mixing == HS_mixing::normal or mixing == HS_mixing::up_down)){
    factorizable = true;
    for(auto &s : value){
      if(s.second != 0.0 and the_model->term.at(s.first)->is_factorizable == false)
      factorizable = false;
    }
  }
  if(the_model->is_factorized and factorizable == false) qcm_ED_throw("conflict between the factorizable character of different instances of the model");
  the_model->is_factorized = factorizable;

  // extracting the sectors from the string _sectors, which is a slash-separated list
  vector<string> sec = split_string(_sectors, '/');
  for(auto& x : sec){
    auto S = sector(x);
    if(the_model->group->sector_is_valid(S) == false) qcm_ED_throw("sector "+x+" is not valid");
    if(S.S and mixing == HS_mixing::normal) mixing = HS_mixing::up_down;
    target_sectors.insert(S);
  }
  if(target_sectors.size() == 0){
    qcm_ED_throw("no target sectors were specified for model "+the_model->name);
  }
  
  try{
    for(auto& s : target_sectors){
      if(s.S >= sector::odd and (mixing&HS_mixing::spin_flip) == 0){
        qcm_ED_throw("sector string " + s.name() + " defines a non conserved spin, but spin is conserved in the model");
      }
      if(s.N >= sector::odd and (mixing&HS_mixing::anomalous) == 0){
        qcm_ED_throw("sector string " + s.name() + " defines a non conserved particle number, but particle number is conserved in the model");
      }
      if(s.S < sector::odd and (mixing&HS_mixing::spin_flip)){
        qcm_ED_throw("sector string " + s.name() + " defines a conserved spin, but spin is not conserved in the model");
      }
      if(s.N < sector::odd and (mixing&HS_mixing::anomalous)){
        qcm_ED_throw("sector string " + s.name() + " defines a conserved particle number, but particle number is not conserved in the model");
      }
    }
  }catch(const string& s) {qcm_ED_catch(s);}
  
  #ifdef QCM_DEBUG
  cout << "cluster model instance " << label << " created" << endl;
  #endif

}


model_instance_base::~model_instance_base()
{
  #ifdef QCM_DEBUG
    cout << "cluster model instance #" << label << " deleted." << endl;
  #endif
}


/**
 returns a string with an identifier
 */
string model_instance_base::full_name()
{
  return the_model->name+'_'+to_string(label);
}



size_t model_instance_base::dimension()
{
  return the_model->n_sites * n_mixed;
}





