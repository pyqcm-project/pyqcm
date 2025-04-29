//
//  QCM.cpp
//  qcm3
//
//  Created by David Sénéchal on 16-11-22.
//  Copyright © 2016 David Sénéchal. All rights reserved.
//
#include <memory>
#include <iostream>
#include <fstream>
#ifdef _OPENMP
  #include <omp.h>
#endif

#include "QCM.hpp"
#include "lattice_model.hpp"
#include "lattice_model_instance.hpp"
#include "parser.hpp"
#include "qcm_ED.hpp"
#include "parameter_set.hpp"
#include "model.hpp"
#include "model_instance_base.hpp"

//==============================================================================
// global variables

shared_ptr<lattice_model> qcm_model = nullptr; // pointer to the unique lattice model of the library
map<int, unique_ptr<lattice_model_instance>> lattice_model_instances; // list of instances

extern map<string, shared_ptr<model>> models;
extern map<size_t, shared_ptr<model_instance_base>> model_instances;

//==============================================================================
namespace QCM{
  
/**
resets all the models and global variables
*/
void great_reset(){
  for(auto& x:lattice_model_instances) x.second.reset();
  lattice_model_instances.clear();
  for(auto& x:models) x.second.reset();
  models.clear();
  qcm_model->sector_strings.clear();
  qcm_model->param_set.reset();
  parameter_set::parameter_set_defined = false;
  lattice_model::model_consolidated = false;
  qcm_model.reset();
  qcm_model = make_shared<lattice_model>();
}


void erase_lattice_model_instance(size_t label){
    size_t n_clus = qcm_model->clusters.size();
    for(size_t i=0; i < n_clus; i++){
      model_instances.erase(label*n_clus+i);
      // cout << "cluster instance " << label*n_clus+i << " deleted" << endl;
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



  void print_model(const string& filename, bool asy_operators, bool asy_labels, bool asy_orb, bool asy_neighbors, bool asy_working_basis)
  {
    ofstream fout(filename);
    qcm_model->print(fout, asy_operators, asy_labels, asy_orb, asy_neighbors, asy_working_basis);
    fout.close();
  }
  
  
  
  
  /**
   sets the model parameters to the values specified by the parameter_set No 'prm_label'
   @param label label to be given to the new instance, to identify it in memory (the library is stateful)
   */
  void new_model_instance(int label)
  {
    if(qcm_model->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    lattice_model_instances[label] = unique_ptr<lattice_model_instance>(new lattice_model_instance(qcm_model, label));
  }
  

  
  
  /**
   defines a new parameter set
   @param val array of pairs (name, value) for the independent parameters
   @param equiv array of 3-tuples (name, multiplier, reference parameter) for the dependent parameters
\   */
  void set_parameters(vector<pair<string,double>>& val, vector<tuple<string, double, string>>& equiv)
  {
    qcm_model->param_set = make_shared<parameter_set>(qcm_model, val, equiv);
  }

  
  /**
   sets the value of the  parameter 'name' in the parameter set with label 'label' to 'value'
   */
  void set_parameter(const string& name, double value)
  {
    if(qcm_model->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    qcm_model->param_set->set_value(name, value);
  }
  
  
  
  /**
   sets the value of the  parameter 'name' in the parameter set with label 'label' to 'value'
   */
  void set_multiplier(const string& name, double value)
  {
    if(qcm_model->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    qcm_model->param_set->set_multiplier(name, value);
  }
  
  
  
  /**
   outputs a map of the parameters for model instance 'label'
   */
  map<string,double> parameters()
  {
    return qcm_model->param_set->value_map();
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
    if(qcm_model->term.find(op) == qcm_model->term.end()) qcm_throw("The lattice operator "+op+" does not exist.");
    return lattice_model_instances.at(label)->momentum_profile_per(*qcm_model->term.at(op), k_set);
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
   returns the V matrix at a given frequency and wavevector
   * @param spin_down true if the spin-down sector is covered
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
  vector<double> dos(const complex<double> w, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->dos(w);
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
    vector3D<double> K = qcm_model->superdual.to(qcm_model->physdual.from(k));
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
    vector3D<double> K = qcm_model->superdual.to(qcm_model->physdual.from(k));
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
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = qcm_model->superdual.to(qcm_model->physdual.from(k[i]));
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
    vector<vector3D<double>> K(k.size());
    for(size_t i = 0; i< K.size(); i++) K[i] = qcm_model->superdual.to(qcm_model->physdual.from(k[i]));
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
  vector<vector3D<double>> K(k.size());
  for(size_t i = 0; i< K.size(); i++) K[i] = qcm_model->superdual.to(qcm_model->physdual.from(k[i]));
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
  size_t Green_function_dimension()
  {
    return qcm_model->dim_GF;
  }
  
  
  /**
   * returns the dimension of the periodized Green function.
   * Essentially: the number of bands, times the mixing number (2 or 4)
   */
  size_t reduced_Green_function_dimension()
  {
    char periodization = global_char("periodization");
    if(periodization == 'N') return qcm_model->dim_GF;
    else return qcm_model->dim_reduced_GF;
  }
  
  
  /**
   * returns the spatial dimension of the model: 0, 1, 2 or 3
   */
  int spatial_dimension()
  {
    return qcm_model->spatial_dimension;
  }
  
  
  
  /**
   returns the mixing
   0: normal, 1: anomalous, 2: spin-flip, 3:anomalous and spin-flip, 4: up and down spin separate
   */
  int mixing()
  {
    return qcm_model->mixing;
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
   * @param name name of the cluster model
   * @param cpos base position of the cluster
   * @param pos array of the positions of the different sites of the cluster, with respect to the base position of the cluster
   */
  void add_cluster(const string &name, const vector3D<int64_t> &cpos, const vector<vector3D<int64_t>> &pos, int ref)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created. Ignoring.");
      return;
    }
    auto tmp = ED::model_size(name);
    try{
      if(get<0>(tmp) != pos.size()) qcm_throw("The number of sites of cluster "+name+" is inconsistent with the cluster model");
    } catch(const string& s) {qcm_catch(s);}
    if(get<1>(tmp) > 0) qcm_model->bath_exists = true;
    for(size_t i=0; i<pos.size(); i++){
      qcm_model->sites.push_back({qcm_model->clusters.size(), i, 0, pos[i]+cpos});
    }
    if (ref == 0){
      ref = qcm_model->clusters.size()+1;
      qcm_model->inequiv.push_back(qcm_model->clusters.size());
    }
    qcm_model->clusters.push_back({get<0>(tmp), get<1>(tmp), qcm_model->sites.size(), name, cpos, ref-1, 0, get<2>(tmp)});
    // n_sites, n_bath, offset, name, position, ref, mixing, n_sym

  }
  
  
  /**
   * defines a new lattice model
   * In the current version, there is a single lattice model in memory
   * @param name name of the model (for reference purposes)
   * @param superlattice array of D superlattice vectors, defining the superlattice of the model and the spatial dimension D
   * @param unit_cell array of D lattice vectors, defining the unit cell of the model (determines the number of bands)
   */
  void new_lattice_model(const string &name, vector<int64_t> &superlattice, vector<int64_t> &unit_cell)
  {
    try{
      if(qcm_model==nullptr) qcm_throw("no cluster has been added to the model!");
    } catch(const string& s) {qcm_catch(s);}
    if(qcm_model->is_closed){
      qcm_warning("model already created. Ignoring.");
      return;
    }
    qcm_model->name = name;
    qcm_model->superlattice = lattice3D(superlattice);
    qcm_model->unit_cell = lattice3D(unit_cell);
    qcm_model->phys.trivial();
    qcm_model->pre_operator_consolidate();
  }
  
  
  /**
   * defines the physical basis of the unit cell of the model.
   * By default, the cartesian basis is used and this function does not always need to be called.
   * @param basis flattened array of the basis vectors (0, 3, 6, or 9 components). These are the components of the Bravais lattice vectors in the 'physical' basis, i.e., as they appear in space.
   */
  void set_basis(vector<double> &basis)
  {
    qcm_model->phys = basis3D(basis);
    qcm_model->phys.inverse();
    qcm_model->phys.dual(qcm_model->physdual);
    qcm_model->physdual.init();
  }
  
  
  /**
   * defines an interaction operator
   * @param name name of the operator
   * @param link bond vector on which the operator is defined
   * @param amplitude default amplitude of the operator, that multiplies all matrix elements and its given value
   * @param orb1 index of the first orbital (from 1 to nband)
   * @param orb2 index of the second orbital (from 1 to nband)
   * @param type type of interaction operator
   */
  void interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->interaction_operator(name, link, amplitude, orb1-1, orb2-1, type);
    } catch(const string& s) {qcm_catch(s);}
  }
  
  
  /**
   * Defines a hopping operator on the lattice.
   * The operator, given two sites labelled 1 and 2, is defined as: 
   * $c^\dagger_{is}\tau^a_{ij}\sigma^b_{ss'}c_{js'}$
   * @param name name given to the operator
   * @param link bond vector on which the operator is defined
   * @param amplitude default amplitude of the operator, that multiplies all matrix elements and its given value
   * @param orb1 index of the first orbital (from 1 to nband)
   * @param orb2 index of the second orbital (from 1 to nband)
   * @param tau label (0,1,2,3) of the Pauli matrix defining the orbital component (a in the formula above)
   * @param sigma label (0,1,2,3) of the Pauli matrix defining the spin component (b in the formula above)
   */
  void hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->hopping_operator(name, link, amplitude, orb1-1, orb2-1, tau, sigma);
    } catch(const string& s) {qcm_catch(s);}
  }
  
  
    /**
   * Defines a current operator on the lattice.
   * The operator, given two sites labelled 1 and 2, is defined as: 
   * $(r_1-r_2)c^\dagger_{is}\tau^2_{ij}c_{js}$
   * @param name name given to the operator
   * @param link bond vector on which the operator is defined
   * @param amplitude default amplitude of the operator, that multiplies all matrix elements and its given value
   * @param orb1 index of the first orbital (from 1 to nband)
   * @param orb2 index of the second orbital (from 1 to nband)
   * @param dir component of the current (0,1,2) standing for (x,y,z)
   */
  void current_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int dir)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->current_operator(name, link, amplitude, orb1-1, orb2-1, dir);
    } catch(const string& s) {qcm_catch(s);}
  }


  /**
   * Defines an anomalous operator on the lattice.
   * @param name name given to the operator
   * @param link bond vector on which the operator is defined
   * @param amplitude default amplitude of the operator, that multiplies all matrix elements and its given value
   * @param orb1 index of the first orbital (from 1 to nband)
   * @param orb2 index of the second orbital (from 1 to nband)
   * @param type type of pairing: singlet, dx, dy, dz
   */
  void anomalous_operator(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb1, int orb2, const string& type)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->anomalous_operator(name, link, amplitude, orb1-1, orb2-1, type);
    } catch(const string& s) {qcm_catch(s);}
  }
  
  /**
   * Defines a density wave on the lattice.
   * @param name name given to the operator
   * @param link bond vector on which the operator is defined
   * @param amplitude default amplitude of the operator, that multiplies all matrix elements and its given value
   * @param orb index of the first site of the pair (from 1 to nband, the number of lattice orbitals). 0 if all orbitals.
   * @param Q wavevector ($\times\pi$) of the density wave
   * @param phase constant phase (see general documentation for the formula)
   * @param type type of pairing: cdw, X, Z, singlet, dx, dy, dz
   */
  void density_wave(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->density_wave(name, link, amplitude, orb-1, Q, phase, type);
    } catch(const string& s) {qcm_catch(s);}
  }

  /**
   * Defines an operator on the lattice with explicit lattice elements
   * @param name name given to the operator
   * @param type type of pairing: Hubbard, Hund, Heisenberg, X, Y, Z, singlet, dx, dy, dz (by default: hopping)
   * @param elem a liste of pairs (site, bond)
   * @param tau label (0,1,2,3) of the Pauli matrix defining the orbital component (a in the formula above)
   * @param sigma label (0,1,2,3) of the Pauli matrix defining the spin component (b in the formula above)
   */
  void explicit_operator(const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau, int sigma)
  {
    if(qcm_model->is_closed){
      qcm_warning("model already created and closed. Ignoring operator creation.");
      return;
    }
    try{
      qcm_model->explicit_operator(name, type, elem, tau, sigma);
    } catch(const string& s) {qcm_catch(s);}
  }

  
  /**
   * returns some information about the clusters in an array of 4-tuples
   * for each cluster of the repeated unit: 1. the name of the cluster model, 2. the number of sites, 3. the number of bath sites, 4. the dimension of the Green function
   */
  vector<tuple<string, int, int, int, int>> cluster_info()
  {
    vector<tuple<string, int, int, int, int>> info(qcm_model->clusters.size());
    for(int i=0; i<info.size(); i++){
      get<0>(info[i]) = qcm_model->clusters[i].name;
      get<1>(info[i]) = (int)qcm_model->clusters[i].n_sites;
      get<2>(info[i]) = (int)qcm_model->clusters[i].n_bath;
      get<3>(info[i]) = (int)qcm_model->clusters[i].n_sites*qcm_model->n_mixed;
      get<4>(info[i]) = (int)qcm_model->clusters[i].n_sym;
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
    for(size_t i = 0; i< k.size(); i++) k[i] = qcm_model->superdual.to(qcm_model->physdual.from(k[i]));
    return lattice_model_instances[label]->Lehmann_Green_function(k, orb, spin_down);
  }


  void Green_function_solve(int label){
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    lattice_model_instances.at(label)->Green_function_solve();  
  }

  void CDMFT_variational_set(vector<string>& varia){
    if(qcm_model->param_set == nullptr) qcm_throw("The parameters have not been specified yet.");
    qcm_model->param_set->CDMFT_variational_set(varia);
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

  double CDMFT_distance(const vector<double>& p, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->CDMFT_distance(p); 
  }

  double monopole(vector3D<double>& k, double a, int nk, int orb, bool rec, int label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->monopole(k, a, nk, orb, rec); 
  }

  /**
   * replace the cluster model with another one with the same number of sites (for DCA)
   * @param name name of the new cluster model
   */
  void switch_cluster_model(const string &name)
  {
    if(models.find(name)==models.end()) qcm_throw("cluster model "+name+" does not exist!");
    qcm_model->clusters[0].name = name;
    if (!models[name]->is_closed) qcm_model->close_model(true);
  }


    bool complex_HS(size_t label)
  {
    #ifdef QCM_DEBUG
    check_instance(label);
    #endif
    return lattice_model_instances.at(label)->complex_HS;
  }

}

