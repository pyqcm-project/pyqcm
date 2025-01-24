//
//  QCM.hpp
//  qcm3
//
//  Created by David Sénéchal on 16-11-22.
//  Copyright © 2016 David Sénéchal. All rights reserved.
//

#ifndef QCM_hpp
#define QCM_hpp

/**
 Interface for using the QCM library
 */

#include <string>
#include <vector>
#include <complex>
#include <map>
#include <functional>
#include "vector3D.hpp"

using namespace std;

struct lattice_matrix_element;

/**
 Interface per se
 */
namespace QCM{
  void check_instance(int label);
  void great_reset();
  void erase_lattice_model_instance(size_t label);
  bool complex_HS(size_t label);
  double Berry_flux(vector<vector3D<double>>& k, int orb, int label);
  double monopole(vector3D<double>& k, double a, int nk, int orb, bool rec, int label);
  double Potthoff_functional(int label);
  double potential_energy(int label);
  double kinetic_energy(int label);
  double spectral_average(const string& name, const complex<double> w, int label);
  int mixing();
  int spatial_dimension();
  map<string,double> parameters();
  map<string,double> instance_parameters(int label);
  matrix<complex<double>> cluster_Green_function(size_t i, complex<double> w, bool spin_down, int label, bool blocks);
  matrix<complex<double>> cluster_self_energy(size_t i, complex<double> w, bool spin_down, int label);
  matrix<complex<double>> cluster_hopping_matrix(size_t i, bool spin_down, int label);
  matrix<complex<double>> CPT_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  matrix<complex<double>> hybridization_function(complex<double> w, bool spin_down, size_t i, int label);
  matrix<complex<double>> periodized_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  matrix<complex<double>> band_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  matrix<complex<double>> projected_Green_function(const complex<double> w, bool spin_down, int label);
  matrix<complex<double>> tk(const vector3D<double> &k, bool spin_down, int label);
  matrix<complex<double>> V_matrix(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> site_and_bond_profile(int label);
  size_t Green_function_dimension();
  size_t reduced_Green_function_dimension();
  vector<double> Berry_curvature(vector3D<double>& k1, vector3D<double>& k2, int nk, int orb, bool rec, int dir, int label);
  vector<double> dos(const complex<double> w, int label);
  vector<double> momentum_profile(const string& op, const vector<vector3D<double>> &k_set, int label);
  vector<matrix<complex<double>>> CPT_Green_function_inverse(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<matrix<complex<double>>> CPT_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<matrix<complex<double>>> periodized_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<matrix<complex<double>>> band_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<complex<double>> periodized_Green_function_element(int r, int c, const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<matrix<complex<double>>> self_energy(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<matrix<complex<double>>> tk(const vector<vector3D<double>> &k, bool spin_down, int label);
  vector<pair<double,string>> ground_state(int label);
  vector<pair<string,double>> averages(const vector<string> &_ops = {}, int label=0);
  vector<pair<vector<double>, vector<double>>> Lehmann_Green_function(vector<vector3D<double>> &k, int orb, bool spin_down, int label);
  vector<tuple<string, int, int, int, int>> cluster_info();
  vector<vector<double>> dispersion(const vector<vector3D<double>> &k, bool spin_down, int label);
  void add_cluster(const string &name, const vector3D<int64_t> &cpos, const vector<vector3D<int64_t>> &pos, int ref=0);
  void anomalous_operator(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb1, int orb2, const string& type);
  void density_wave(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type);
  void explicit_operator(const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau=1, int sigma=0);
  void global_parameter_init();
  void hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma);
  void interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type);
  void k_integral(int dim, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb=false);
  void new_lattice_model(const string &name, vector<int64_t> &superlattice, vector<int64_t> &lattice);
  void new_model_instance(int label);
  void print_model(const string& filename, bool asy_operators=false, bool asy_labels=false, bool asy_orb=false, bool asy_neighbors=false, bool asy_working_basis=false);
  void qcm_init();
  void set_basis(vector<double> &basis);
  void set_parameter(const string& name, double value);
  void set_parameters(vector<pair<string,double>>&, vector<tuple<string, double, string>>&);
  void wk_integral(int dim, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy,bool verb=false);
  void Green_function_solve(int label);
  void CDMFT_variational_set(vector<string>& varia);
  void CDMFT_host(const vector<double>& freqs, const vector<double>& weights, int label);
  void set_CDMFT_host(int label, const vector<double>& freqs, const int clus, const vector<matrix<Complex>>& H, const bool spin_down);
  double CDMFT_distance(const vector<double>& p, int label);
  void switch_cluster_model(const string &name);
  vector<vector<matrix<Complex>>> get_CDMFT_host(bool spin_down, int label);
};

#endif /* QCM_hpp */







