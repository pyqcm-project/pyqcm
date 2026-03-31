/** @file
 This file is the main header for the library, included in other projects
 */

#include <iostream>
#include <fstream>
#include <memory>
#include <tuple>
#include <H5Cpp.h>
#include <sys/stat.h>
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

    QCM_ASSERT(M->complex_Hilbert == false);

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
  
  
  
  
  void write_instance_hdf5(const string& filename, int label, const string& group)
  {
    if(model_instances.find(label) == model_instances.end())
      qcm_ED_throw("The label " + to_string(label) + " is out of range.");
    try {
      // Open file in read-write mode if it exists, create otherwise
      struct stat st{};
      H5::H5File file(filename,
        (stat(filename.c_str(), &st) == 0) ? H5F_ACC_RDWR : H5F_ACC_TRUNC);
      // Remove the group if it already exists (e.g. overwriting a previous run)
      if(file.nameExists(group)) file.unlink(group);
      H5::Group grp = file.createGroup(group);
      model_instances.at(label)->write_hdf5(grp);
    } catch(H5::Exception& e){
      qcm_ED_throw("HDF5 error while writing instance: " + string(e.getDetailMsg()));
    }
  }


  void read_instance_hdf5(const string& filename, int label, const string& group)
  {
    if(model_instances.find(label) == model_instances.end())
      qcm_ED_throw("The label " + to_string(label) + " is out of range.");
    try {
      H5::H5File file(filename, H5F_ACC_RDONLY);
      H5::Group grp = file.openGroup(group);
      model_instances.at(label)->read_hdf5(grp);
    } catch(H5::Exception& e){
      qcm_ED_throw("HDF5 error while reading instance: " + string(e.getDetailMsg()));
    }
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


  //============================================================================


  template<typename HilbertField>
  static CombinedMCF_data assemble_combined_mcf_site_data(
      shared_ptr<model_instance<HilbertField>> inst, bool spin_down)
  {
    auto& group   = inst->the_model->group;
    int   n_mixed = inst->n_mixed;
    int   dim_GF  = inst->dim_GF;

    // Use the first ground state only (same convention as qmatrix())
    for(auto& x : inst->states){
      shared_ptr<mcf_set<HilbertField>> mcf =
        dynamic_pointer_cast<mcf_set<HilbertField>>(spin_down ? x->gf_down : x->gf);
      if(!mcf) continue;

      // Find the first irrep block that has a combined MCF
      int M = 0;
      for(size_t r = 0; r < group->g; ++r)
        M = max(M, mcf->combined[r].floors());
      if(M == 0)
        qcm_ED_throw("get_combined_mcf: combined MCF is empty. "
                     "Set combine_mcf=True before solving.");

      CombinedMCF_data result;
      result.A.resize(M);
      result.B.resize(M);

      for(int j = 0; j < M; ++j){
        block_matrix<Complex> A_blk(group->site_irrep_dim * n_mixed);
        block_matrix<Complex> B_blk(group->site_irrep_dim * n_mixed);
        for(size_t r = 0; r < group->g; ++r){
          if(j < mcf->combined[r].floors()){
            A_blk.block[r] = to_complex_matrix(mcf->combined[r].A[j]);
            B_blk.block[r] = to_complex_matrix(mcf->combined[r].B[j]);
          }
        }
        result.A[j] = matrix<Complex>(dim_GF);
        result.B[j] = matrix<Complex>(dim_GF);
        group->to_site_basis(A_blk, result.A[j], n_mixed);
        group->to_site_basis(B_blk, result.B[j], n_mixed);
      }

      // Weight matrix W (may be non-square: p_actual rows × p columns per block)
      // Use the first non-empty block to determine the column count of W
      size_t W_cols = 0;
      for(size_t r = 0; r < group->g; ++r)
        if(mcf->combined[r].floors() > 0){ W_cols = mcf->combined[r].W.c; break; }

      block_matrix<Complex> W_blk(group->site_irrep_dim * n_mixed);
      for(size_t r = 0; r < group->g; ++r)
        if(mcf->combined[r].floors() > 0) W_blk.block[r] = mcf->combined[r].W;
      result.W = matrix<Complex>(dim_GF);
      group->to_site_basis(W_blk, result.W, n_mixed);

      return result;
    }
    qcm_ED_throw("get_combined_mcf: no valid ground state found.");
    return {};  // unreachable
  }


  CombinedMCF_data get_combined_mcf(bool spin_down, size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& inst_base = model_instances.at(label);
    if(inst_base->GF_solver != GF_format_MCF &&
       inst_base->GF_solver != GF_format_Q_to_MCF)
      qcm_ED_throw("get_combined_mcf: the instance at label "
                   + to_string(label) + " does not use the MCF format "
                   "(GF_method must be 'M', or 'L' with combine_mcf=True).");
    if(!global_bool("combine_mcf"))
      qcm_ED_throw("get_combined_mcf: the combined MCF is not built unless "
                   "the global parameter 'combine_mcf' is set to True.");
    if(!inst_base->gf_solved) inst_base->Green_function_solve();

    if(inst_base->complex_Hilbert)
      return assemble_combined_mcf_site_data(
        dynamic_pointer_cast<model_instance<Complex>>(inst_base), spin_down);
    return assemble_combined_mcf_site_data(
      dynamic_pointer_cast<model_instance<double>>(inst_base), spin_down);
  }
}

