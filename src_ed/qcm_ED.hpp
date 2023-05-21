#ifndef qcm_ED_h
#define qcm_ED_h

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <complex>
#include <array>
#include <tuple>
#include <functional>

#include "matrix_element.hpp"
#include "matrix.hpp"

using namespace::std;

namespace ED{
  
  /**
   initialization
   */
  void qcm_ED_init();
  
  
  /**
   function that defines the groundwork for a system

   name : name of the system (=cluster)
   n_sites : number of sites for which the Green function will be computed
   n_bath : number of bath sites
   gen : array of the generators of the symmetry group of the system
   bath_irrep : if true, the labels in 'gen', for the bath section, are irrep labels for each bath instead of permutations of bath labels
   */
  void new_model(const string &name, const size_t n_sites, const size_t n_bath, const vector<vector<int>> &gen, bool bath_irrep);
  
  
  
  /**
   retrieves basic data for model from its name
   returns: n_sites, n_bath
   */
  tuple<size_t,size_t,size_t> model_size(const string &name);
  
  
  
  
  
  /**
   function that adds an operator to the Hamiltonian

   model_name : name of the model the operator belongs to
   name : name of the operator
   type : type of the operator (e.g. one-body, anomalous, interaction, etc.)
   elements : a vector of one-body matrix elements that defines the operator
   */
  void new_operator(const string &model_name, const string &name, const string &type, const vector<matrix_element<double>> &elements);
  void new_operator(const string &model_name, const string &name, const string &type, const vector<matrix_element<complex<double>>> &elements);

  
  
  /**
   checks whether operator named 'name' already exists within the model 'model_name'

   returns true is the operator exists already
   */
  bool exists(const string &model_name, const string &name);
  
  
  
  /**
  defines an instance of a model, i.e., with particular values of the coefficients of the operators in the Hamiltonian,
  and with a set of Hilbert space sectors.

  model_name : name of the model the Hamiltonian is based upon
  param : values of the coefficients of the operators making up the Hamiltonian
  opt : string of options, to be interpreted differently by each solver.
  sec : The sectors of Hilbert space to build the Hamiltonian(s) upon (in string format).
  label : label of the instance (globally, not only for the model in question)
   */
  void new_model_instance(const string &model_name, map<string, double> &param, const string &sec, size_t label);
  
  void erase_model_instance(size_t label);

  
  /**
   function that returns the mixing state of the instance
   0: normal, 1: anomalous, 2: spin-flip, 3:anomalous and spin-flip, 4: up and down spin separate
   */
  int mixing(size_t label);
  
  /**
   function that returns 1 if the Hilbert space is complex, 0 if it is real
   */
  bool complex_HS(size_t label);
  
  
  
  
  /**
   function that finds the ground state or density matrix, without computing the Green functions.
   label : label of the model instance to solve
   output : ground state energy and a string representing the sector information
   */
  pair<double, string> ground_state_solve(size_t label);
  
  
  
  /**
   function that returns the ground state averages of the operators for the model instance # label
   label : label of the model instance
   output : array of string/double pairs : name and ground state average of the operators of the model
   */
  vector<tuple<string,double,double>> cluster_averages(size_t label);
  
  
  
  
  /**
   function that solves for the re-usable representation of the one-body Green function.
   label : label of the model instance to solve (in case there are many)
   */
  void Green_function_solve(size_t label);
  
  
  
  /**
   function that evaluates the one-body Green function at a given complex frequency
   z : complex frequency
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   blocks : true if in irreducible representations basis
   output : the Green function matrix
   */
  matrix<complex<double>> Green_function(const complex<double> &z, bool spin_down, const size_t label, bool blocks);
  
  
  /**
   function that evaluates the one-body Green function integrated around the negative real axis
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   output : the integrated Green function matrix
   */
  matrix<complex<double>> Green_function_average(bool spin_down, const size_t label);
  
  
  
    /**
   function that evaluates the cluster density from the integrated Green function
   label : label of the model instance (in case there are many)
   output : the density
   */
  double Green_function_density(const size_t label);




  /**
   function that evaluates the self-energy at a given complex frequency
   z : complex frequency
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   output : the self-energy matrix
   */
  matrix<complex<double>> self_energy(const complex<double> &z, bool spin_down, const size_t label);

  
  
  /**
   function that evaluates the hopping matrix
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   output : the hopping matrix
   */
  matrix<complex<double>> hopping_matrix(bool spin_down, const size_t label);
  matrix<complex<double>> hopping_matrix_full(bool spin_down, bool diag, const size_t label);
  
  /**
   function that evaluates the matrix of density-density interactions
   */
  vector<tuple<int,int,double>> interactions(const size_t label);

  
  /**
   function that evaluates the hybridization function (in case there is a bath)
   w : a complex frequency
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   output : the hybridization function matrix
   */
  matrix<complex<double>> hybridization_function(const complex<double> w, bool spin_down, const size_t label);
  
  
  
  
  /**
   function that evaluates the dynamic susceptibility of an operator
   op : the name of the operator
   w : the array of frequencies at which to compute the susceptibility
   label : label of the model instance (in case there are many)
   output : the corresponding values of the dynamic susceptibility
   */
  vector<complex<double>> susceptibility(const string &op, const vector<complex<double>> &w, const size_t label);
  
  
  
  
  /**
   function that evaluates the dynamic susceptibility of an operator and returns a Lehmann representation
   op : the name of the operator
   label : label of the model instance (in case there are many)
   output : array of pole/residue pairs
   */
  vector<pair<double,double>> susceptibility_poles(const string &op, const size_t label);
  
  
  
  
  /**
   function that evaluates the contribution of the cluster ground states and bath to the Potthoff functional
   label : label of the model instance (in case there are many)
   */
  double Potthoff_functional(const size_t label);
  
  
  
  
  
  /**
   function that returns the dimension of the Green function matrix
   label : label of the model instance (in case there are many)
   output : a size_t
   */
  size_t Green_function_dimension(size_t label);
  
  
  
  /**
   computes the overlap squared of the ground states of two instances of the same model: tr(rho_1 rho_2)
   model_name : name of the model
   param1 : first set of parameters
   param2 : second set of parameters
   sec : Hilbert space sector
   */
  double fidelity(const int label1, const int label2);
  
  
  
  /**
   computes the asymptotic values of the trace of the self-energy (taking into account Nambu signs)
   label : label of the model instance (in case there are many)
   */
  double tr_sigma_inf(const size_t label);
  
  
  /**
   writes the solved model instance in a stream, for re-use later
   fout : output stream
   label : label of the model instance (in case there are many)
   */
  void write_instance(ostream& fout, int label);
    

  /**
   reads the solved model instance from a text file
   fin : input stream
   label : label of the model instance (in case there are many)
   */
  void read_instance(istream& fin, int label);

  
  
  /**
   prints the global model on a file
   */
  void print_models(ostream& fout);
  

  /**
  initializes and defines the global parameters
  */
  void global_parameter_init();
  

  /**
   returns the Lehman representation of the Green function
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   */
  pair<vector<double>, vector<complex<double>>> qmatrix(bool spin_down, const size_t label);


  /**
   returns the Lehman representation of the hybridization function
   spin_down : true if we want the spin-down sector (if the mixing=4)
   label : label of the model instance (in case there are many)
   */
  pair<vector<double>, vector<complex<double>>> hybridization(bool spin_down, const size_t label);


  void w_integral(function<void (Complex w, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb=false);

  /**
  writes a string (output) of the ground state wavefunction, is dimension is small enough
  */
  string print_wavefunction(const size_t label);

  /**
  gives the matrix elements of an operator
  model_name : name of model
  op_name : name of operator
  */
  pair<string, vector<matrix_element<Complex>>> matrix_elements(const string& model_name, const string& op_name);

  /**
  computes the density matrix of subsystem A by tracing over subsystem B
  sites : sites definin subsystem A
  label : label of instance
  */
  pair<matrix<Complex>, vector<uint64_t>> density_matrix(const vector<int> &sites, size_t label);
};

#endif
