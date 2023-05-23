#ifndef model_instance_base_h
#define model_instance_base_h

#include "model.hpp"
#include "Green_function_set.hpp"

struct sector_data
{
  double E0; //!< GS energy in that sector
  double weight; //!< weight of that sector: (dim/total_dim)*exp(-(E0-GS_energy)/T)
};

//! Abstract class for an instance of a model
struct model_instance_base
{
  
  // members
	double SEF_bath; //!< contribution to the Potthoff functional
  bool complex_Hilbert; //!< true if a complex Hilbert space is used
  bool gf_read;
  bool gf_solved;
  bool gs_solved;
  bool hopping_solved;
  bool is_correlated; // true if interactions are non zero on the cluster
  bool spin_down; //!< true if currently dealing with the spin-down sector, in cases where it is separate and different
  double E0; //!< Cluster grand potential
  double GS_energy; //!< Ground state energy of the cluster
  GF_FORMAT GF_solver;
  int dim_GF; //!< dimension of the Green function (number of sites x n_mixed)
  int mixing; //!< mixing state of the cluster itself
  map<sector, sector_data> sec_data; //!< GS energy for each sector;
  map<string, double> value; //!< value of the parameters
  matrix<Complex> M_down; //!< average of G_down (Green function) : M_{ab} = \L c^\dagger_b c_a \R. Used for uncorrelated instances or solutions read from files.
  matrix<Complex> M; //!< average of G (Green function) : M_{ab} = \L c^\dagger_b c_a \R. Used for uncorrelated instances or solutions read from files.
  sector GS_sector; //!< actual symmetry sector of the GS
  set<sector> sector_set; //!< set of sectors containing the low energy states
  set<sector> target_sectors; //!< set of sectors to probe for low energy states
  shared_ptr<model> the_model;
  size_t label;
  size_t n_mixed; //!< 1:normal, 2:anomalous or spin-flip, 4: anomalous and spin-flip
  size_t total_dim; //!> total HS dimension of all sectors
  vector<tuple<string,double,double>> averages; //!> averages and variances of operators
  double GF_density; //!> density from the Green function

  model_instance_base(size_t _label, shared_ptr<model> _the_model, const map<string,double> _value, const string &_sectors);
  ~model_instance_base();

  string full_name();
  size_t dimension();
  
  virtual pair<double, string> low_energy_states() = 0;
  virtual pair<double, double> cluster_averages(shared_ptr<Hermitian_operator> h) = 0;
  virtual void Green_function_solve() = 0;
  virtual pair<double, string> one_body_solve() = 0;
  virtual matrix<Complex> Green_function(const Complex &z, bool spin_down, bool blocks) = 0;
  virtual void Green_function_average() = 0;
  virtual void Green_function_density() = 0;
  virtual matrix<Complex> self_energy(const Complex &z, bool spin_down) = 0;
  virtual matrix<Complex> hopping_matrix(bool spin_down) = 0;
  virtual vector<tuple<int,int,double>> interactions() = 0;
  virtual matrix<Complex> hopping_matrix_full(bool spin_down, bool diag) = 0;
  virtual vector<Complex> susceptibility(shared_ptr<Hermitian_operator> h, const vector<Complex> &w) = 0;
  virtual matrix<Complex> hybridization_function(Complex w, bool spin_down) = 0;
  virtual vector<pair<double,double>> susceptibility_poles(shared_ptr<Hermitian_operator> h) = 0;
  virtual void print(ostream& fout) = 0;
  virtual double tr_sigma_inf() = 0;
  virtual void write(ostream& fout) = 0;
  virtual void read(istream& fin) = 0;
  virtual void print_wavefunction(ostream& fout) = 0;
  virtual pair<matrix<Complex>, vector<uint64_t>>  density_matrix_mixed(vector<int> sites) = 0;
  virtual pair<matrix<Complex>, vector<uint64_t>>  density_matrix_factorized(vector<int> sites) = 0;
};





#endif /* model_instance_base_hpp */
