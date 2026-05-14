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
 * @file QCM.hpp
 * @brief Public C++ interface of the QCM library (free functions in the QCM namespace).
 */

#include <string>
#include <optional>
#include <vector>
#include <complex>
#include <map>
#include <functional>
#include "vector3D.hpp"
#include "qcm_ED.hpp"

using namespace std;

struct lattice_matrix_element;
struct System;

/**
 Interface per se
 */
namespace QCM{
  //! Throws if no model instance with this label exists.
  void check_instance(int label);
  //! Resets the whole system, allowing the model to be redefined.
  void great_reset();
  //! Erases the model instance with the given label.
  void erase_lattice_model_instance(size_t label);
  //! Returns true if the model instance requires a complex Hilbert space.
  bool complex_HS(size_t label);
  //! Computes the Berry flux around a closed loop of wavevectors.
  double Berry_flux(vector<vector3D<double>>& k, int orb, int label);
  //! Returns the topological charge of a node (e.g. Weyl point) by integrating the Berry flux on a small cube around it.
  double monopole(vector3D<double>& k, double a, int nk, int orb, bool rec, int label);
  //! Computes the Potthoff functional for a given instance.
  double Potthoff_functional(int label);
  //! Returns the potential energy for a given instance.
  double potential_energy(int label);
  //! Returns the kinetic energy for a given instance.
  double kinetic_energy(int label);
  //! Returns the contribution of a frequency to the average of an operator.
  double spectral_average(const string& name, const complex<double> w, int label);
  //! Returns the integer code for the mixing state (0=normal, 1=anomalous, 2=spin-flip, 3=full Nambu, 4=up/down separate).
  int mixing();
  //! Returns the spatial dimension (0, 1, 2 or 3) of the model.
  int spatial_dimension();
  //! Returns the values of all parameters in the parameter set.
  map<string,double> parameters();
  //! Returns the values of all parameters for a given instance.
  map<string,double> instance_parameters(int label);
  //! Computes the cluster Green function at a given frequency.
  matrix<complex<double>> cluster_Green_function(size_t i, complex<double> w, bool spin_down, int label, bool blocks);
  //! Computes the cluster self-energy at a given frequency.
  matrix<complex<double>> cluster_self_energy(size_t i, complex<double> w, bool spin_down, int label);
  //! Returns the cluster one-body (hopping) matrix.
  matrix<complex<double>> cluster_hopping_matrix(size_t i, bool spin_down, int label);
  //! Symmetrizes a dim_GF matrix with respect to cluster translations at wavevector k.
  matrix<complex<double>> compact_tiling(const matrix<complex<double>>& A, const vector3D<double>& k);
  //! Returns the combined matrix continued fraction (W, A, B) for the cluster Green function at k.
  ED::CombinedMCF_data get_combined_mcf_k(const vector3D<double>& k, bool spin_down, int label);
  //! Computes the CPT Green function at a given frequency and wavevector.
  matrix<complex<double>> CPT_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  //! Computes the CPT Green function at a frequency and wavevector indexed in the external hybridization grid.
  matrix<complex<double>> CPT_Green_function(int iw, int ik, int label);
  //! Returns the hybridization function for a given cluster and instance.
  matrix<complex<double>> hybridization_function(complex<double> w, bool spin_down, size_t i, int label);
  //! Returns the frequency integral of the local lattice Green function.
  matrix<complex<double>> Green_integral(bool spin_down, int clus, int label);
  //! Computes the periodized Green function at a frequency and wavevector.
  matrix<complex<double>> periodized_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  //! Computes the periodized Green function in the band basis at a frequency and wavevector.
  matrix<complex<double>> band_Green_function(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  //! Computes the projected Green function at a given frequency, as used in CDMFT.
  matrix<complex<double>> projected_Green_function(const complex<double> w, bool spin_down, int label);
  //! Returns the k-dependent one-body matrix t(k) at a given wavevector.
  matrix<complex<double>> tk(const vector3D<double> &k, bool spin_down, int label);
  //! Computes the V matrix at a given frequency and wavevector.
  matrix<complex<double>> V_matrix(const complex<double> w, const vector3D<double> &k, bool spin_down, int label);
  //! Computes site and bond profiles in all clusters of the repeated unit.
  pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> site_and_bond_profile(int label);
  //! Returns the dimension of the CPT Green function matrix.
  size_t Green_function_dimension();
  //! Returns the reduced Green function dimension n (n, 2n or 4n depending on mixing state).
  size_t reduced_Green_function_dimension();
  //! Computes the Berry curvature on a 2D region of the Brillouin zone (works in 2D only).
  vector<double> Berry_curvature(vector3D<double>& k1, vector3D<double>& k2, int nk, int orb, bool rec, int dir, int label);
  //! Computes the density of states at a given frequency.
  vector<double> dos(const complex<double> w, int label, bool use_grid=false);
  //! Computes the momentum-resolved average of an operator on a set of wavevectors.
  vector<double> momentum_profile(const string& op, const vector<vector3D<double>> &k_set, int label);
  //! Computes the inverse CPT Green function at a frequency and an array of wavevectors.
  vector<matrix<complex<double>>> CPT_Green_function_inverse(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the CPT Green function at a frequency and an array of wavevectors.
  vector<matrix<complex<double>>> CPT_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the periodized Green function at a frequency and an array of wavevectors.
  vector<matrix<complex<double>>> periodized_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the band-basis periodized Green function at a frequency and an array of wavevectors.
  vector<matrix<complex<double>>> band_Green_function(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Returns one matrix element of the periodized Green function at a frequency and array of wavevectors.
  vector<complex<double>> periodized_Green_function_element(int r, int c, const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the self-energy of the periodized Green function at a frequency and array of wavevectors.
  vector<matrix<complex<double>>> self_energy(const complex<double> w, const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Returns the k-dependent one-body matrix t(k) at an array of wavevectors.
  vector<matrix<complex<double>>> tk(const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the ground state of the cluster(s) of a given instance.
  vector<pair<double,string>> ground_state(int label);
  //! Returns the average values of all (or a selected list of) operators in a model instance.
  vector<pair<string,double>> averages(const vector<string> &_ops = {}, int label=0);
  //! Computes a Lehmann representation of the periodized Green function at given wavevectors.
  vector<pair<vector<double>, vector<double>>> Lehmann_Green_function(vector<vector3D<double>> &k, int orb, bool spin_down, int label);
  //! Returns information about each cluster: (name, n_sites, n_bath, dim_GF, n_sym).
  vector<tuple<string, int, int, int, int>> cluster_info();
  //! Returns information about each system in the lattice.
  vector<tuple<string, int, int, int>> systems_info();
  //! Computes the dispersion relation (band energies) for an array of wavevectors.
  vector<vector<double>> dispersion(const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Computes the dispersion relation in the orbital basis for an array of wavevectors.
  vector<matrix<Complex>> epsilon(const vector<vector3D<double>> &k, bool spin_down, int label);
  //! Adds a cluster to the repeated unit.
  void add_cluster(const vector3D<int64_t> &cpos, const vector<vector3D<int64_t>> &pos, int ref=0, bool conj=false);
  //! Adds a system (group of clusters) to the lattice.
  void add_system(const string &name, const int clus=0);
  //! Defines an anomalous (pairing) operator.
  void anomalous_operator(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb1, int orb2, const string& type);
  //! Defines a density-wave operator.
  void density_wave(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type);
  //! Defines an explicit operator from a list of matrix elements.
  void explicit_operator(const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau=1, int sigma=0);
  //! Initializes the global parameter table.
  void global_parameter_init();
  //! Defines a hopping (one-body) operator.
  void hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma);
  //! Defines a current operator.
  void current_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int dir, bool re=true);
  //! Defines an interaction operator (Hubbard, Hund, Heisenberg, or X/Y/Z).
  void interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type);
  //! Performs an adaptive Brillouin-zone integral of a vector-valued function.
  void k_integral(int dim, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb=false);
  //! Performs a fixed-grid Brillouin-zone integral of a vector-valued function.
  void k_integral_grid(int dim, int nkx, int nky, int nkz, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv);
  //! Sets the number of wavevectors along each direction of the fixed wavevector grid.
  void set_wavevector_grid(int nkx, int nky, int nkz);
  //! Returns the current sizes of the fixed wavevector grid.
  void get_wavevector_grid(int& nkx, int& nky, int& nkz);
  //! Initiates the lattice model.
  void new_lattice_model(const string &name, vector<int64_t> &superlattice, vector<int64_t> &lattice, const string &latt_hybrid="");
  //! Creates a new instance of the lattice model with values associated to terms of the Hamiltonian.
  void new_model_instance(int label);
  //! Prints a description of the model into a file (optionally with Asymptote markup).
  void print_model(const string& filename, bool asy_operators=false, bool asy_labels=false, bool asy_orb=false, bool asy_neighbors=false, bool asy_working_basis=false);
  //! Initializes the qcm library.
  void qcm_init();
  //! Sets the working basis (a Dx3 real matrix).
  void set_basis(vector<double> &basis);
  //! Sets the value of a parameter in the parameter set.
  void set_parameter(const string& name, double value);
  //! Sets the multiplier of a dependent parameter.
  void set_multiplier(const string& name, double value);
  //! Sets the values and dependences of multiple parameters at once.
  void set_parameters(vector<pair<string,double>>&, vector<tuple<string, double, string>>&);
  //! Performs an adaptive integral over frequencies and the Brillouin zone.
  void wk_integral(int dim, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy,bool verb=false);
  //! Forces eager (non-lazy) computation of cluster Green functions in all clusters.
  void Green_function_solve(int label);
  //! Defines the set of CDMFT variational parameters.
  void CDMFT_variational_set(vector<vector<string>>& varia);
  //! Sets the CDMFT host function from a list of frequencies and weights.
  void CDMFT_host(const vector<double>& freqs, const vector<double>& weights, int label);
  //! Sets the CDMFT host function from precomputed matrix data.
  void set_CDMFT_host(int label, const vector<double>& freqs, const int clus, const vector<matrix<Complex>>& H, const bool spin_down);
  //! Computes the CDMFT distance function at the given variational point.
  double CDMFT_distance(const vector<double>& p, int clus, int label);
  //! Computes the CDMFT residual vector r(p) used by least-squares optimizers.
  vector<double> CDMFT_residuals(const vector<double>& p, int clus, int label);
  //! Computes the analytical Jacobian dr/dp of the CDMFT residual vector.
  vector<double> CDMFT_gradient(const vector<double>& p, int clus, int label);
  //! Retrieves the CDMFT host function (per-cluster matrices over all frequencies).
  vector<vector<matrix<Complex>>> get_CDMFT_host(bool spin_down, int label);
};

#endif /* QCM_hpp */







