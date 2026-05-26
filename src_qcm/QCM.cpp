//
//  QCM.cpp
//  qcm3
//
//  Created by David Sénéchal on 16-11-22.
//  Copyright © 2016 David Sénéchal. All rights reserved.
//

/**
 * @file QCM.cpp
 * @brief Implementation of the public QCM-namespace interface.
 */

#include <memory>
#include <iostream>
#include <fstream>
#ifdef _OPENMP
  #include <omp.h>
#endif

#include "QCM.hpp"
#include "global_parameter.hpp"
#include "lattice_model.hpp"
#include "lattice_model_instance.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"
#include "parameter_set.hpp"
#include "model.hpp"
#include "model_instance_base.hpp"

//==============================================================================
// global variables

map<string, shared_ptr<lattice_model>> qcm_models; // registry of named lattice models
map<int, unique_ptr<lattice_model_instance>> lattice_model_instances; // list of instances

extern map<string, shared_ptr<model>> models;
extern map<size_t, shared_ptr<model_instance_base>> model_instances;

namespace QCM {
  // Resolves a model name to a registered lattice_model.
  // If 'name' is empty, falls back to the single registered model (error if
  // none or more than one). Throws if the name is not found.
  shared_ptr<lattice_model> resolve_lattice_model(const string& name)
  {
    if(name.empty()){
      if(qcm_models.empty()) qcm_throw("no lattice model has been created yet.");
      if(qcm_models.size() > 1)
        qcm_throw("more than one lattice model is defined; you must specify the model name.");
      return qcm_models.begin()->second;
    }
    auto it = qcm_models.find(name);
    if(it == qcm_models.end())
      qcm_throw("the lattice model \""+name+"\" does not exist.");
    return it->second;
  }
}

using QCM::resolve_lattice_model;

// Returns the model entry for 'name', creating an empty lattice_model if it
// does not yet exist. Used by add_cluster() so that the first cluster added
// implicitly creates the model. The name "" is allowed.
static shared_ptr<lattice_model> get_or_create_lattice_model(const string& name)
{
  auto& mod = qcm_models[name];
  if(!mod) mod = make_shared<lattice_model>();
  return mod;
}

vector<double> grid_freqs; // optional imaginary frequency grid for discrete integrals
vector<double> grid_weights; // weights associated with grid_freqs

// per-direction wavevector grid sizes set via QCM::set_wavevector_grid().
// A value of 0 means "not set: fall back to global_int(\"kgrid_side\")".
int wavevector_grid_nx = 0;
int wavevector_grid_ny = 0;
int wavevector_grid_nz = 0;

//==============================================================================
namespace QCM{


/**
Sets the per-direction wavevector grid sizes used by the grid integration
routines. A value of zero in any direction is treated as "not set" and the
global parameter "kgrid_side" is used as the fallback for that direction.
*/
void set_wavevector_grid(int nkx, int nky, int nkz){
  wavevector_grid_nx = nkx;
  wavevector_grid_ny = nky;
  wavevector_grid_nz = nkz;
}


/**
Resolves the wavevector grid sizes used by the grid integration routines:
falls back to the global parameter "kgrid_side" for any direction that has
not been explicitly set via set_wavevector_grid().
*/
void get_wavevector_grid(int& nkx, int& nky, int& nkz){
  int def = (int)global_int("kgrid_side");
  nkx = wavevector_grid_nx > 0 ? wavevector_grid_nx : def;
  nky = wavevector_grid_ny > 0 ? wavevector_grid_ny : def;
  nkz = wavevector_grid_nz > 0 ? wavevector_grid_nz : def;
}


/**
resets all the models and global variables
*/
void great_reset(){
  for(auto& x:lattice_model_instances) x.second.reset();
  lattice_model_instances.clear();
  for(auto& x:models) x.second.reset();
  models.clear();
  for(auto& x: qcm_models) x.second.reset();
  qcm_models.clear();
  parameter_set::parameter_set_defined = false;
}


void erase_lattice_model_instance(size_t label){
    auto it = lattice_model_instances.find(label);
    if(it == lattice_model_instances.end()) return;
    size_t nsys = it->second->model->nsys;
    for(size_t i=0; i < nsys; i++){
      model_instances.erase(label*nsys+i);
    }
    lattice_model_instances.erase(label);
}




/**
 * prints the definition of the model in a file
 * @param filename name of the output file
 * @param asy_operators true if asymptote files for the various operators are produced
 * @param asy_labels true if the site labels are written in these files
 * @param asy_orb true if the lattice orbital labels are written in these files
 * @param asy_neighbors true if the neighboring clusters are drawn
 * @param asy_working_basis true if the drawing is done in the working basis instead of the physical basis
 */
  void check_instance(int label)
  {
    if(lattice_model_instances.find(label) == lattice_model_instances.end()) 
      qcm_throw("The instance # "+to_string(label)+" does not exist.");
  }



  void print_model(const string& name, const string& filename, bool asy_operators, bool asy_labels, bool asy_orb, bool asy_neighbors, bool asy_working_basis)
  {
    auto mod = resolve_lattice_model(name);
    ofstream fout(filename);
    if (!fout.good()) qcm_throw("failed to open file " + filename);
    mod->print(fout, asy_operators, asy_labels, asy_orb, asy_neighbors, asy_working_basis);
    fout.close();
  }
  
  
  
  
  /**
   sets the model parameters to the values specified by the parameter_set No 'prm_label'
   @param label label to be given to the new instance, to identify it in memory (the library is stateful)
   */
  void new_model_instance(const string& model_name, int label)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    lattice_model_instances[label] = unique_ptr<lattice_model_instance>(new lattice_model_instance(mod, label));
  }




  /**
   defines a new parameter set
   @param val array of pairs (name, value) for the independent parameters
   @param equiv array of 3-tuples (name, multiplier, reference parameter) for the dependent parameters
\   */
  void set_parameters(const string& model_name, vector<pair<string,double>>& val, vector<tuple<string, double, string>>& equiv)
  {
    auto mod = resolve_lattice_model(model_name);
    mod->param_set = make_shared<parameter_set>(mod, val, equiv);
  }


  /**
   sets the value of the  parameter 'param_name' in the parameter set of lattice model 'model_name'
   */
  void set_parameter(const string& model_name, const string& param_name, double value)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    mod->param_set->set_value(param_name, value);
  }



  /**
   sets the multiplier of the parameter 'param_name' in the parameter set of lattice model 'model_name'
   */
  void set_multiplier(const string& model_name, const string& param_name, double value)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    mod->param_set->set_multiplier(param_name, value);
  }



  /**
   outputs a map of the parameters of lattice model 'model_name'
   */
  map<string,double> parameters(const string& model_name)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    return mod->param_set->value_map();
  }
  
  
  /**
   outputs a map of the parameters for model instance 'label'
   */
  map<string,double> instance_parameters(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->params;
  }

  
  
  /**
   computes the momentum-resolved average of an operator
   */
  vector<double> momentum_profile(const string& op, const vector<vector3D<double>> &k_set, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& mod = *lattice_model_instances.at(label)->model;
    if(mod.term.find(op) == mod.term.end()) qcm_throw("The lattice operator "+op+" does not exist.");
    return lattice_model_instances.at(label)->momentum_profile_per(*mod.term.at(op), k_set);
  }
  
  
  
  /**
   solves for the ground state (GS) of the clusters
   returns an array of pairs of values, one pair per cluster, giving the GS energy and the GS sector
   */
  vector<pair<double,string>> ground_state(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->ground_state();
  }
  
  
  
  /**
   returns the cluster Green function for cluster # i at frequency w
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> cluster_Green_function(size_t i, complex<double> w, bool spin_down, int label, bool blocks)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->cluster_Green_function(i, w, spin_down, blocks);
  }
  
  
  /**
   returns the cluster Green function for cluster # i at frequency w
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> cluster_self_energy(size_t i, complex<double> w, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->cluster_self_energy(i, w, spin_down);
  }
  
  

  /**
   returns the cluster hopping matrix for cluster # i
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> cluster_hopping_matrix(size_t i, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->cluster_hopping_matrix(i, spin_down);
  }
  
  
  
  
  
  /**
   returns the cluster hybridization for cluster # i
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> hybridization_function(complex<double> w, bool spin_down, size_t i, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->hybridization_function(i, w, spin_down);
  }
  
  


    /**
   returns the frequency integral of the local lattice Green function
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> Green_integral(bool spin_down, int clus, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->Green_integral(spin_down, clus);
  }
  





  /**
   returns the CPT Green function at a given frequency and wavevector
   * @param spin_down true if the spin-down sector is covered
   */
  matrix<complex<double>> CPT_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector3D<double> K = mod.superdual.to(mod.physdual.from(k));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    Green_function_k M(G, K);
    lattice_model_instances.at(label)->set_Gcpt(M);
    return M.Gcpt;
  }
  
  
  /**
   returns the CPT Green function at a given frequency and for an array of wavevectors
   * @param spin_down true if the spin-down sector is covered
   */
  vector<matrix<complex<double>>> CPT_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances[label]->model;
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    vector<matrix<Complex>> R(k.size());
    for(size_t i = 0; i< k.size(); i++) {
      Green_function_k M(G, K[i]);
      lattice_model_instances.at(label)->set_Gcpt(M);
      R[i] = M.Gcpt;
    }
    return R;
  }
  
  
  /**
   returns the CPT Green function at a frequency and wavevector indexed within the external hybridization grids.
   Requires that the model was created with a non-empty hybrid_file.
   * @param iw index of the frequency in the external hybridization frequency array
   * @param ik index of the wavevector in the external hybridization wavevector array
   */
  matrix<complex<double>> CPT_Green_function(int iw, int ik, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    if(mod.hybrid == nullptr) qcm_throw("CPT_Green_function(iw, ik, ...) requires an external hybridization (hybrid_file)");
    lattice_hybrid& H = *mod.hybrid;
    if(iw < 0 || (size_t)iw >= H.nw) qcm_throw("frequency index iw out of range");
    if(ik < 0 || (size_t)ik >= H.nk) qcm_throw("wavevector index ik out of range");
    Complex w(0.0, H.w[iw]);
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, false);
    G.iw = iw;
    Green_function_k M(G, H.k[ik], ik);
    lattice_model_instances.at(label)->set_Gcpt(M);
    cout << "k (read) = " << H.k[ik] << "\tw = " << w << endl; // TEMPO
    cout << "gamma = \n" << mod.lattice_hybridization(M.G.iw, M.ik) << endl; // TEMPO
    return M.Gcpt;
  }


  /**
   returns the V matrix at a given frequency and wavevector
   * @param spin_down true if the spin-down sector is covered
   * @param w complex frequency
   * @param k wavevector in the physdual basis
   */
  matrix<complex<double>> V_matrix(const complex<double> w, const vector3D<double> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector3D<double> K = mod.superdual.to(mod.physdual.from(k));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    Green_function_k M(G, K);
    lattice_model_instances.at(label)->set_V(M);
    return M.V;
  }
  
  
/**
   returns the inverse CPT Green function at a given frequency and for an array of wavevectors
   * @param spin_down true if the spin-down sector is covered
   */
  vector<matrix<complex<double>>> CPT_Green_function_inverse(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances[label]->model;
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    G.G.inverse();
    vector<matrix<Complex>> R(k.size());
    for(size_t i = 0; i< k.size(); i++) {
      Green_function_k M(G, K[i]);
      lattice_model_instances.at(label)->inverse_Gcpt(G.G, M);
      R[i] = M.V;
    }
    return R;
  }


/**
 * returns the k-dependent lattice one-body matrix $t(k)$, for a single wavevector input
 * @param k reduced wavevector, in the domain [0,1], like in integrals
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
 * @param label label of the instance
 */
  matrix<complex<double>> tk(const vector3D<double> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(Complex(0,0.1), false, spin_down);
    Green_function_k M(G, k);
    lattice_model_instances.at(label)->set_V(M);
    return M.t;
  }
  
  
/**
 * returns the k-dependent lattice one-body matrix $t(k)$, for a multiple wavevector input
 * @param k array of reduced wavevectors, each in the domain [0,1], like in integrals
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
 * @param label label of the instance
 */
  vector<matrix<complex<double>>> tk(const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances[label]->model;
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(Complex(0., 0.1), false, spin_down);
    vector<matrix<Complex>> R(k.size());
    for(size_t i = 0; i< k.size(); i++) {
      Green_function_k M(G, k[i]);
      lattice_model_instances.at(label)->set_Gcpt(M);
      R[i] = M.t;
    }
    return R;
  }

  
  /**
   returns the density of states at a given frequency for the different lattice orbitals
   */
  vector<double> dos(const complex<double> w, int label, bool use_grid)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->dos(w, use_grid);
  }
  
  
  /**
   returns the contribution of a frequency w to the average of an operator named 'name'
   */
  double spectral_average(const string& name, const complex<double> w, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->spectral_average(name, w);
  }

  
  
  /**
   returns the periodized Green function at a given frequency and wavevector
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  matrix<complex<double>> periodized_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector3D<double> K = mod.superdual.to(mod.physdual.from(k));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    Green_function_k M(G, K);
    lattice_model_instances.at(label)->periodized_Green_function(M);
    return M.g;
  }


  /**
   returns the band Green function at a given frequency and wavevector
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  matrix<complex<double>> band_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector3D<double> K = mod.superdual.to(mod.physdual.from(k));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    Green_function_k M(G, K);
    lattice_model_instances.at(label)->periodized_Green_function(M);

    return lattice_model_instances.at(label)->band_Green_function(M);
  }
  
  
  
  
  /**
   returns the periodized Green function at a given frequency and for an array of wavevectors
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  vector<matrix<complex<double>>> periodized_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    vector<matrix<Complex>> R;
    R.reserve(K.size());
    for(size_t i = 0; i< K.size(); i++) {
      Green_function_k M(G, K[i]);
      lattice_model_instances.at(label)->periodized_Green_function(M);
      R.push_back(M.g);
    }
    return R;
  }


  /**
   returns the periodized Green function at a given frequency and for an array of wavevectors
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  vector<matrix<complex<double>>> band_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    vector<matrix<Complex>> R;
    R.reserve(K.size());
    for(size_t i = 0; i< K.size(); i++) {
      Green_function_k M(G, K[i]);
      lattice_model_instances.at(label)->periodized_Green_function(M);
      R.push_back(lattice_model_instances.at(label)->band_Green_function(M));
    }
    return R;
  }


/**
 returns the periodized Green function at a given frequency and for an array of wavevectors
* @param spin_down true if the spin-down sector is covered (mixing = 4)
  */
vector<complex<double>> periodized_Green_function_element(int r, int c, const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
{
  #ifdef QCM_DEBUG
check_instance(label);
#endif
  lattice_model& mod = *lattice_model_instances.at(label)->model;
  vector<vector3D<double>> K(k.size());
  for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
  Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
  vector<Complex> R;
  R.reserve(K.size());
  for(size_t i = 0; i< K.size(); i++) {
    Green_function_k M(G, K[i]);
    lattice_model_instances.at(label)->periodized_Green_function(M);
    R.push_back(M.g(r,c));
  }
  return R;
}
  
  
  

  /**
   returns the dispersion relation for an array of wavevectors
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  vector<vector<double>> dispersion(const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instance& inst = *lattice_model_instances.at(label);
    
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = inst.cluster_Green_function({0.0,0.05}, false, spin_down);
    
    vector<vector<double>> R(K.size());
    for(size_t i = 0; i< K.size(); i++) {
      Green_function_k M(G, K[i]);
      R[i] = inst.dispersion(M);
    }
    return R;
  }
  
  
  
  
  /**
   Applies compact_tiling to an arbitrary dim_GF matrix at wavevector k.
   @param A  input matrix (dim_GF x dim_GF)
   @param k  wavevector in the same convention as M.k and set_V(): k_phys * a / (2*pi),
             i.e. in units of the inverse primitive lattice constant / (2*pi).
             The inter-cluster Bloch phase is exp(i * k * neighbor * 2*pi) where
             neighbor is the wrapping vector in primitive lattice units.
   */
  matrix<complex<double>> compact_tiling(const string& model_name, const matrix<complex<double>>& A, const vector3D<double>& k)
  {
    lattice_model& mod = *resolve_lattice_model(model_name);
    return mod.compact_tiling(A, k);
  }


  /**
   Returns the combined MCF (W, A[j], B[j]) periodized into the band basis at wavevector k.
   Implements the 'L' periodization scheme: adds inter-cluster hopping V to A[0],
   applies compact_tiling to A[j>=1] and B[j>=1], then periodizes all blocks and W.
   Only valid with a single cluster per lattice model and with GF_method='M' or
   GF_method='L' + combine_mcf=True.
   @param k          wavevector in the physical reciprocal basis (same convention as periodized_Green_function)
   @param spin_down  true for the spin-down sector (mixing = 4)
   @param label      lattice model instance label
   */
  ED::CombinedMCF_data get_combined_mcf_k(const vector3D<double>& k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    QCM_ASSERT(mod.clusters.size() == 1);

    // Convert to superdual basis (same convention as set_V / periodize)
    vector3D<double> K = mod.superdual.to(mod.physdual.from(k));

    // Build a dummy Green_function_k to retrieve V(k)
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(Complex(0, 0.1), false, spin_down);
    Green_function_k M(G, K);
    lattice_model_instances.at(label)->set_V(M);

    // Get combined MCF from the ED layer
    size_t sys_label = mod.nsys * label + mod.clusters[0].sys_start;
    auto D = ED::get_combined_mcf(spin_down, sys_label);

    // Add inter-cluster hopping into the first diagonal block
    D.A[0].v += M.V.v;

    int nA = (int)D.A.size();
    int nB = (int)D.B.size();

    // Apply compact_tiling to A[j>=1] and B[j>=1]
    for(int j = 1; j < nA; ++j)
      D.A[j] = mod.compact_tiling(D.A[j], K);
    for(int j = 1; j < nB; ++j)
      D.B[j] = mod.compact_tiling(D.B[j], K);

    // Periodize all blocks into the band basis
    ED::CombinedMCF_data Dper;
    Dper.A.resize(nA);
    Dper.B.resize(nB);
    for(int j = 0; j < nA; ++j)
      Dper.A[j] = mod.periodize(K, D.A[j]);
    for(int j = 0; j < nB; ++j)
      Dper.B[j] = mod.periodize(K, D.B[j]);
    Dper.W = mod.periodize(K, D.W);

    return Dper;
  }


  /**
   returns the dispersion relation for an array of wavevectors
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  vector<matrix<Complex>> epsilon(const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instance& inst = *lattice_model_instances.at(label);
    
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = inst.cluster_Green_function({0.0,0.05}, false, spin_down);
    
    vector<matrix<Complex>> R(K.size());
    for(size_t i = 0; i< K.size(); i++) {
      Green_function_k M(G, K[i]);
      R[i] = inst.epsilon(M);
    }
    return R;
  }


  
  /**
   returns the self-energy associated with periodized Green function at a given frequency and for an array of wavevectors
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  vector<matrix<complex<double>>> self_energy(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = mod.superdual.to(mod.physdual.from(k[i]));
    Green_function G = lattice_model_instances.at(label)->cluster_Green_function(w, false, spin_down);
    vector<matrix<Complex>> R;
    R.reserve(K.size());
    for(size_t i = 0; i< K.size(); i++) {
      Green_function_k M(G, K[i]);
      lattice_model_instances.at(label)->self_energy(M);
      R.push_back(M.sigma);
    }
    return R;
  }
  
  
  
  
  /**
   computes the Potthoff functional
   */
  double Potthoff_functional(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->Potthoff_functional();
  }
  
  
  
  /**
   computes the potential energy (Tr (Sigma.G))
   */
  double potential_energy(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->potential_energy();
  }
  
  
  
  /**
   computes the kinetic energy (sum of averages of one-body operators)
   */
  double kinetic_energy(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->E_kin;
  }



  /**
   computes the lattice averages
   */
  vector<pair<string,double>> averages(const vector<string> &_ops, int label)
  {
    return lattice_model_instances.at(label)->averages(_ops);
  }
  
  
  
  /**
   returns the projected Green function, as used in CDMFT:
   \bar G = \sum_\kv (G_0^{-1}(\kv,\omega) - \Sigma(\omega))^{-1}
   @param w frequency
 * @param spin_down true if the spin-down sector is covered (mixing = 4)
   */
  matrix<complex<double>> projected_Green_function(const complex<double> w, bool spin_down, int label)
  {
    return lattice_model_instances.at(label)->projected_Green_function(w, spin_down);
  }
  
  
  
  /**
   returns the dimensions
   */
  size_t Green_function_dimension(const string& model_name)
  {
    return resolve_lattice_model(model_name)->dim_GF;
  }


  /**
   * returns the dimension of the periodized Green function.
   * Essentially: the number of bands, times the mixing number (2 or 4)
   */
  size_t reduced_Green_function_dimension(const string& model_name)
  {
    auto mod = resolve_lattice_model(model_name);
    char periodization = global_char("periodization");
    if(periodization == 'N') return mod->dim_GF;
    else return mod->dim_reduced_GF;
  }


  /**
   * returns the spatial dimension of the model: 0, 1, 2 or 3
   */
  int spatial_dimension(const string& model_name)
  {
    return resolve_lattice_model(model_name)->spatial_dimension;
  }



  /**
   returns the mixing
   0: normal, 1: anomalous, 2: spin-flip, 3:anomalous and spin-flip, 4: up and down spin separate
   */
  int mixing(const string& model_name)
  {
    return resolve_lattice_model(model_name)->mixing;
  }
  
  

  /**
   computes the Berry flux for model_instance label, through a contour specified by wavevectors k
   @param k array of wavevectors in the Brillouin zone ($\times\pi$)
   @param open true if the path defined by k is "open" in the Brillouin zone.
   @param orb lattice orbital label
   @param label label of the model instance
   */
  double Berry_flux(vector<vector3D<double>>& k, int orb, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->Berry_flux(k, orb, false);
  }





  /**
   computes the Berry curvature for model_instance label, with a nk x nk grid.
   Works in two dimensions only.
   @param k1 lowest-left corner of the region in the Brillouin zone ($\times\pi$)
   @param k2 upper-right corner of the region in the Brillouin zone ($\times\pi$)
   @param nk number of wavevectors on the side of the grid
   @param orb lattice orbital label
   @param label label of the model instance
   */
  vector<double> Berry_curvature(vector3D<double>& k1, vector3D<double>& k2, int nk, int orb, bool rec, int dir, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->Berry_curvature(k1, k2, nk, orb, rec, dir);
  }
  
  
  
  /**
   * Adds a cluster to the supercluster (or repeated unit)
   * @param cpos base position of the cluster
   * @param pos array of the positions of the different sites of the cluster, with respect to the base position of the cluster
   * @param ref if cluster is equivalent to another cluster, index of that cluster. If not, -1.
   * @param conj true if cluster is equivalent to the complex conjugate of its master
   */
  void add_cluster(const string& model_name, const vector3D<int64_t> &cpos, const vector<vector3D<int64_t>> &pos, int ref, bool conj)
  {
    auto mod = get_or_create_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created. Ignoring.");
      return;
    }
    for(size_t i=0; i<pos.size(); i++) mod->sites.push_back({mod->clusters.size(), i, 0, pos[i]+cpos});
    if(ref == -1) ref = mod->clusters.size();
    mod->clusters.push_back({pos.size(), mod->sites.size(), cpos, ref, 0, conj, 0, 0});
    // n_sites, n_bath, offset, name, position, ref, mixing, sys_start, nsys
  }


  /**
   * Adds a system to the lattice model (or repeated unit)
   * @param model_name name of the lattice model the system belongs to
   * @param name name of the cluster model
   * @param clus cluster this model is associated to (starts at 0)
   */
  void add_system(const string& model_name, const string &name, const int clus)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created. Ignoring.");
      return;
    }
    if(clus > mod->clusters.size())
      qcm_throw("This system cannot be added to the cluster (out of range, or maybe clusters not yet added to lattice model)");

    auto tmp = ED::model_size(name);
    if(get<0>(tmp) != mod->clusters[clus].n_sites) qcm_throw("The number of sites of cluster "+name+" is inconsistent with the cluster model");
    if(get<1>(tmp) > 0) mod->bath_exists = true;
    // n_sites, n_bath, name, clus, mixing, n_sym
    mod->systems.push_back({get<0>(tmp), get<1>(tmp), name, clus, 0, get<2>(tmp)});
  }



  /**
   * Finalizes the geometry of a lattice model. Must be called after all
   * clusters and systems have been added (via add_cluster/add_system).
   * @param name name of the model (must match the name used in add_cluster)
   * @param superlattice array of D superlattice vectors, defining the superlattice of the model and the spatial dimension D
   * @param unit_cell array of D lattice vectors, defining the unit cell of the model (determines the number of bands)
   * @param latt_hybrid name of HDF5 file containing the lattice hybridization data
   */
  void new_lattice_model(const string &name, vector<int64_t> &superlattice, vector<int64_t> &unit_cell, const string &latt_hybrid)
  {
    auto mod = resolve_lattice_model(name);
    if(mod->is_closed){
      qcm_warning("model already created. Ignoring.");
      return;
    }
    if(mod->clusters.empty())
      qcm_throw("no cluster has been added to model \""+name+"\"!");
    mod->name = name;
    mod->superlattice = lattice3D(superlattice);
    mod->unit_cell = lattice3D(unit_cell);
    mod->phys.trivial();
    mod->hybrid_file = latt_hybrid;
    mod->hybrid = nullptr;
    mod->pre_operator_consolidate();
  }


  /**
   * defines the physical basis of the unit cell of the model.
   * By default, the cartesian basis is used and this function does not always need to be called.
   * @param model_name name of the lattice model
   * @param basis flattened array of the basis vectors (0, 3, 6, or 9 components). These are the components of the Bravais lattice vectors in the 'physical' basis, i.e., as they appear in space.
   */
  void set_basis(const string& model_name, vector<double> &basis)
  {
    auto mod = resolve_lattice_model(model_name);
    mod->phys = basis3D(basis);
    mod->phys.inverse();
    mod->phys.dual(mod->physdual);
    mod->physdual.init();
  }


  /**
   * defines an interaction operator
   * @param model_name name of the lattice model
   * @param name name of the operator
   */
  void interaction_operator(const string& model_name, const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->interaction_operator(name, link, amplitude, orb1-1, orb2-1, type);
  }


  /**
   * Defines a hopping operator on the lattice.
   */
  void hopping_operator(const string& model_name, const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->hopping_operator(name, link, amplitude, orb1-1, orb2-1, tau, sigma);
  }


  /**
   * Defines a current operator on the lattice.
   */
  void current_operator(const string& model_name, const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int dir, bool pau)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->current_operator(name, link, amplitude, orb1-1, orb2-1, dir, pau);
  }


  /**
   * Defines an anomalous operator on the lattice.
   */
  void anomalous_operator(const string& model_name, const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb1, int orb2, const string& type)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->anomalous_operator(name, link, amplitude, orb1-1, orb2-1, type);
  }

  /**
   * Defines a density wave on the lattice.
   */
  void density_wave(const string& model_name, const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->density_wave(name, link, amplitude, orb-1, Q, phase, type);
  }

  /**
   * Defines an operator on the lattice with explicit lattice elements
   */
  void explicit_operator(const string& model_name, const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau, int sigma)
  {
    auto mod = resolve_lattice_model(model_name);
    if(mod->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    mod->explicit_operator(name, type, elem, tau, sigma);
  }

  
  /**
   * returns some information about the clusters in an array of 4-tuples
   * for each cluster of the repeated unit: 1. the number of sites, 2. the number of systems, 3. the index of the first system, 4. the dimension of the Green function
   */
  vector<tuple<string, int, int, int, int>> cluster_info(const string& model_name)
  {
    auto mod = resolve_lattice_model(model_name);
    vector<tuple<string, int, int, int, int>> info(mod->clusters.size());
    for(int i=0; i<info.size(); i++){
      get<0>(info[i]) = (int)mod->clusters[i].n_sites;
      get<1>(info[i]) = (int)mod->clusters[i].nsys;
      get<2>(info[i]) = (int)mod->clusters[i].sys_start;
      get<3>(info[i]) = (int)mod->clusters[i].n_sites*mod->n_mixed;
    }
    return info;
  }



  /**
   * returns some information about the systems in an array of 4-tuples
   * for each system: 1. the name, 2. the number of sites, 3. the number of bath orbitals, 4. the label of the cluster
   */
  vector<tuple<string, int, int, int>> systems_info(const string& model_name)
  {
    auto mod = resolve_lattice_model(model_name);
    vector<tuple<string, int, int, int>> info(mod->systems.size());
    for(int i=0; i<info.size(); i++){
      get<0>(info[i]) = mod->systems[i].name;
      get<1>(info[i]) = (int)mod->systems[i].n_sites;
      get<2>(info[i]) = (int)mod->systems[i].n_bath;
      get<3>(info[i]) = (int)mod->systems[i].clus;
    }
    return info;
  }



  /**
   * returns information about the averages of local operators on sites and bonds
   * @param label label of the model instance
   * the first element of the returned object is a vector of quantities for each site
   * the second element of the returned object is a vectors of quantities for each NN bond
   */
  pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> site_and_bond_profile(int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    auto& M = *lattice_model_instances[label];
    return M.site_and_bond_profile();
  }

  

/**
   * @param spin_down true if the spin-down sector is covered
 */
  vector<pair<vector<double>, vector<double>>> Lehmann_Green_function(vector<vector3D<double>> &k, int orb, bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model& mod = *lattice_model_instances.at(label)->model;
    for(size_t i = 0; i< k.size(); i++) k[i] = mod.superdual.to(mod.physdual.from(k[i]));
    return lattice_model_instances[label]->Lehmann_Green_function(k, orb, spin_down);
  }


  void Green_function_solve(int label){
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instances.at(label)->Green_function_solve();  
  }

  void CDMFT_variational_set(const string& model_name, vector<vector<string>>& varia){
    auto mod = resolve_lattice_model(model_name);
    if(mod->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    mod->param_set->CDMFT_variational_set(varia);
  }

  void CDMFT_host(const vector<double>& freqs, const vector<double>& weights, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instances.at(label)->CDMFT_host(freqs, weights); 
  }

  void set_CDMFT_host(int label, const vector<double>& freqs, const int clus, const vector<matrix<Complex>>& H, const bool spin_down)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instances.at(label)->set_CDMFT_host(freqs, clus, H, spin_down);
  }


  vector<vector<matrix<Complex>>> get_CDMFT_host(bool spin_down, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->get_CDMFT_host(spin_down); 
  }

  double CDMFT_distance(const vector<double>& p, int clus, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->CDMFT_distance(p, clus);
  }

  vector<double> CDMFT_residuals(const vector<double>& p, int clus, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->CDMFT_residuals(p, clus);
  }

  vector<double> CDMFT_gradient(const vector<double>& p, int clus, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->CDMFT_gradient(p, clus);
  }

  double monopole(vector3D<double>& k, double a, int nk, int orb, bool rec, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->monopole(k, a, nk, orb, rec); 
  }


  bool complex_HS(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->complex_HS;
  }

}

