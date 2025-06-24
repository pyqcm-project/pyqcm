/** @file
 This file is the main header for the library, included in other projects
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <tuple>
#include "qcm_ED.hpp"
#include "model_instance.hpp"
#include "Operators/one_body_operator.hpp"
#include "Operators/anomalous_operator.hpp"
#include "Operators/interaction_operator.hpp"
#include "Operators/Hund_operator.hpp"
#include "Operators/Heisenberg_operator.hpp"
#include "Operators/general_interaction_operator.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif
using namespace ED;

// global variables
map<string, shared_ptr<model>> models;
map<size_t, shared_ptr<model_instance_base>> model_instances;

double max_gap; //!< maximum excitation energy compatible with minimum_weight and temperature
run_statistics STATS;

//------------------------------------------------------------------------------


// template specializations exposed
// template class matrix_element<double>;
// template class matrix_element<Complex>;

// template class model_instance<double>;
// template class model_instance<Complex>;

namespace ED{
  
  void new_model(const string &name, const size_t L, const size_t nb, const vector<vector<int>> &gen, bool bath_irrep)
  {
    if(global_bool("verb_Hilbert")) ED_basis::verb = true;

    if(models.find(name) != models.end()){
      qcm_warning("The name "+name+" has already been used for a model. Ignoring (using same model as before)");
      return;
    }
    if(nb>32) qcm_ED_throw("number of bath orbitals exceeds limits or negative!");
    if(L>1024) qcm_ED_throw("number of cluster sites  exceeds limits or negative!");
    
    models[name] = make_shared<model>(name, L+nb, nb, gen, bath_irrep);
    
    max_gap = -log(global_double("minimum_weight"))*global_double("temperature");
    if(max_gap < MIN_GAP) max_gap = MIN_GAP;
  }
  
  void check_instance(int label){
    if(model_instances.find(label) == model_instances.end()) 
    qcm_ED_throw("The cluster instance label "+to_string(label)+" is out of range.");
  }
  
  void erase_model_instance(size_t label){
    // if(lattice_model_instances.find(label) != lattice_model_instances.end()) lattice_model_instances.erase(label);
    model_instances.erase(label);
}



  
  void new_operator(const string &model_name, const string &_name, const string &_type, const vector<matrix_element<double>> &elements)
  {
    if(!elements.size()) return;
    if(models.find(model_name) == models.end())
      qcm_ED_throw("The model "+model_name+" is not defined. Check spelling.");
    shared_ptr<model> M = models.at(model_name);
    if(M->is_closed){
      qcm_warning("model " + model_name + " has already been instantiated and is closed for modifications. Ignoring.");
    }
    if(_type == "one-body") M->term[_name] = make_shared<one_body_operator<double>>(_name, M, elements);
    else if(_type == "anomalous") M->term[_name] = make_shared<anomalous_operator<double>>(_name, M, elements);
    else if(_type == "interaction") M->term[_name] = make_shared<interaction_operator>(_name, M, elements);
    else if(_type == "Hund") M->term[_name] = make_shared<Hund_operator>(_name, M, elements);
    else if(_type == "Heisenberg") M->term[_name] = make_shared<Heisenberg_operator>(_name, M, elements);
    else if(_type == "X") M->term[_name] = make_shared<Heisenberg_operator>(_name, M, elements, 'X');
    else if(_type == "Y") M->term[_name] = make_shared<Heisenberg_operator>(_name, M, elements, 'Y');
    else if(_type == "Z") M->term[_name] = make_shared<Heisenberg_operator>(_name, M, elements, 'Z');
    else if(_type == "general_interaction") M->term[_name] = make_shared<general_interaction_operator<double>>(_name, M, elements);
    else qcm_throw("type of operator "+_name+" is not yet implemented");
  }
  
  
  void new_operator(const string &model_name, const string &_name, const string &_type, const vector<matrix_element<Complex>> &elements)
  {
    if(!elements.size()) return;
    if(models.find(model_name) == models.end()) qcm_ED_throw("The model "+model_name+" is not defined. Check spelling.");
    shared_ptr<model> M = models.at(model_name);
    if(M->is_closed){
      qcm_warning("model " + model_name + " has already been instantiated and is closed for modifications. Ignoring.");
    }
    if(_type == "one-body") M->term[_name] = make_shared<one_body_operator<Complex>>(_name, M, elements);
    else if(_type == "anomalous") M->term[_name] = make_shared<anomalous_operator<Complex>>(_name, M, elements);
    else if(_type == "general_interaction") M->term[_name] = make_shared<general_interaction_operator<Complex>>(_name, M, elements);
    else cout << "ED_WARNING : type of operator " << _name << " is not yet implemented" << endl;
  }
  
  
  
  bool exists(const string &model_name, const string &name)
  {
    if(models.find(model_name) == models.end()) qcm_ED_throw("model "+model_name+" does not exist!");
    model& M = *models.at(model_name);
    if(M.term.find(name) != M.term.end()) return true;
    else return false;
  }
  

  tuple<size_t,size_t,size_t>  model_size(const string &name)
  {
    if(models.find(name) == models.end()) qcm_ED_throw("model "+name+" does not exist!");
    return {models.at(name)->n_sites, models.at(name)->n_bath, models.at(name)->group->e.size()};
  }
  

  void new_model_instance(const string &model_name, map<string, double> &param, const string &sec, size_t label)
  {
    bool need_complex = false;
    if(models.find(model_name) == models.end()) qcm_ED_throw("The model "+model_name+" is not defined and so no model instance based on it is allowed.");
    model& mod = *models.at(model_name);
    if(mod.is_closed == false) mod.is_closed = true;

    model_instances.erase(label);
    
    // first, remove values associated with non existent operators
    auto it = param.begin();
    while(it != param.end()){
      if(mod.term.find(it->first) == mod.term.end()){
        param.erase(it++);
      }
      else ++it;
    }
    
    // need to know whether the instance is complex or real
    for(auto& v : param){
      if(v.second != 0 and mod.term.at(v.first)->is_complex){
        need_complex = true;
        break;
      }
    }
    // decides whether the sector set requires a complex Hilbert space
    if(mod.group->has_complex_irrep) need_complex = true;
    
    if(need_complex) 
      model_instances[label] = make_shared<model_instance<Complex>>(label, models.at(model_name), param, sec);
    else model_instances[label] = make_shared<model_instance<double>>(label, models.at(model_name), param, sec);
  }
  
  
  
  int mixing(size_t label)
  {
    return model_instances.at(label)->mixing;
  }
  
  
    
  bool complex_HS(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->complex_Hilbert;
  }
  


  pair<double, string> ground_state_solve(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    
    pair<double, string> gs;
    gs = model_instances.at(label)->low_energy_states();
    return gs;
  }
  
  
  
  vector<tuple<string,double,double>>  cluster_averages(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    
    auto& M = model_instances.at(label);
    if(!M->gs_solved) M->low_energy_states();
    return M->averages;
  }
  
  
  
  pair<matrix<Complex>, vector<uint64_t>>  density_matrix(const vector<int> &sites, size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);

    assert(M->complex_Hilbert == false);

    if(M->the_model->is_factorized) return M->density_matrix_factorized(sites);
    else return M->density_matrix_mixed(sites);
  }



  void Green_function_solve(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    model_instances.at(label)->Green_function_solve();
  }
  
  
  
  matrix<complex<double>> Green_function(const Complex &z, bool spin_down, const size_t label, bool blocks)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->Green_function(z, spin_down, blocks);
  }

  

  matrix<complex<double>> Green_function_average(bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    Green_function_solve(label);
    if(spin_down) return model_instances.at(label)->M_down;
    else return model_instances.at(label)->M;
  }

  
  double Green_function_density(const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->GF_density;
  }


  matrix<complex<double>> self_energy(const Complex &z, bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->self_energy(z, spin_down);
  }
  
  
  matrix<complex<double>> hopping_matrix(bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->hopping_matrix(spin_down);
  }
  
  
  matrix<complex<double>> hopping_matrix_full(bool spin_down, bool diag, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->hopping_matrix_full(spin_down, diag);
  }
  
  
  vector<tuple<int,int,double>> interactions(const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->interactions();
  }

  

  matrix<complex<double>> hybridization_function(const Complex w, bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return model_instances.at(label)->hybridization_function(w, spin_down);
  }
  
  
  
  vector<complex<double>> susceptibility(const string &op, const vector<Complex> &w, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    if(M->the_model->term.find(op) == M->the_model->term.end()) qcm_ED_throw("Operator "+op+" is not defined in the model.");
    return model_instances.at(label)->susceptibility(M->the_model->term.at(op), w);
  }
  
  
  
  vector<pair<double,double>> susceptibility_poles(const string &op, const size_t label){
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    if(M->the_model->term.find(op) == M->the_model->term.end()) qcm_ED_throw("Operator "+op+" is not defined in the model.");
    return model_instances.at(label)->susceptibility_poles(M->the_model->term.at(op));
  }
  
  
  
  double Potthoff_functional(const size_t label){
    model_instance_base& M = *model_instances.at(label);
    if(!M.gf_solved) M.Green_function_solve();
    return M.SEF_bath + M.E0;
  }
  
  
  
  size_t Green_function_dimension(size_t label)
  {
    return model_instances.at(label)->dimension();
  }
  
  
  
  double tr_sigma_inf(const size_t label)
  {
    return model_instances.at(label)->tr_sigma_inf();
  }
  
  
  void print_models(ostream& fout)
  {
    for(auto& x : models) x.second->print(fout);
    banner('=', "model instances",fout);
    for(auto& x : model_instances) x.second->print(fout);
  }
  
  
  
  double fidelity(int label1, int label2)
  {
    check_instance(label1);
    check_instance(label2);
    auto I1 = model_instances.at(label1);
    auto I2 = model_instances.at(label2);
    if(!I1->complex_Hilbert) return dynamic_pointer_cast<model_instance<double>>(I1)->fidelity(*dynamic_pointer_cast<model_instance<double>>(I2));
    else return dynamic_pointer_cast<model_instance<Complex>>(I1)->fidelity(*dynamic_pointer_cast<model_instance<Complex>>(I2));
  }
  
  
  
  
  void write_instance(ostream& fout, int label)
  {
    if(model_instances.find(label) == model_instances.end()) qcm_ED_throw("The label "+to_string(label)+" is out of range.");
    auto& M = model_instances.at(label);
    M->write(fout);
  }
  
  
  
  void read_instance(istream& fin, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    M->read(fin);
    #ifdef QCM_DEBUG
    cout << "cluster instance " << label << " read from file" << endl;
    #endif
  }
  
  
  pair<vector<double>, vector<complex<double>>> qmatrix(bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    if(!M->complex_Hilbert){
      return dynamic_pointer_cast<model_instance<double>>(M)->qmatrix(spin_down);
    }
    return dynamic_pointer_cast<model_instance<complex<double>>>(M)->qmatrix(spin_down);
  }

  
  pair<vector<double>, vector<complex<double>>> hybridization(bool spin_down, const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    if(!M->complex_Hilbert){
      dynamic_pointer_cast<model_instance<double>>(M)->hybridization(spin_down);
    }
    return dynamic_pointer_cast<model_instance<complex<double>>>(M)->hybridization(spin_down);
  }

  string print_wavefunction(const size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = model_instances.at(label);
    ostringstream sout;
    M->print_wavefunction(sout);
    return sout.str();
  }

  pair<string, vector<matrix_element<Complex>>> matrix_elements(const string& model_name, const string& op_name)
  {
    if(models.find(model_name) == models.end())
      qcm_ED_throw("The model "+model_name+" is not defined. Check spelling.");
    shared_ptr<model> M = models.at(model_name);
    if(M->term.find(op_name) == M->term.end())
      qcm_ED_throw("operator "+op_name+" is not defined in model . Check spelling.");
    return {M->term[op_name]->type(), M->term[op_name]->matrix_elements()};
  }
}

