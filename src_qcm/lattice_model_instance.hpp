#ifndef lattice_model_instance_hpp
#define lattice_model_instance_hpp

#include <array>
#include "lattice_model.hpp"
#include "block_matrix.hpp"

//! instance of a model (i.e. with specific values of the parameters in the Hamiltonian)
struct lattice_model_instance{
	int label; //!< label used to distinguish different, concurrent instances stored in memory
	bool gs_solved; //!< true if the ground state has been already solved  for that instance
	bool gf_solved; //!< true if the Green function has been already solved  for that instance
	bool average_solved; //!< true if the lattice averages have already been computed for that instance
	bool SEF_solved; //!< true if the Potthoff functional (SEF) has already been computed for that instance
	bool PE_solved; //!< true if the potential energy has already been computed for that instance
	bool complex_HS; //!< true if a complex Hilbert space is needed in one of the clusters
	shared_ptr<lattice_model> model; //!< backtrace to the lattice model
	matrix<Complex> H; //!< one-body  matrix (k-independent part)
	matrix<Complex> H_down; //!< one-body  matrix (k-independent part) for the spin-down part, if mixing=4
	block_matrix<Complex> Hc; //!< one-body  matrix of the clusters
	block_matrix<Complex> Hc_down; //!< one-body  matrix of the clusters (spin-down part, if mixing=4)
	map<string, double> params; //!< values of the lattice and cluster parameters
	vector<pair<double,string>> gs; //!< ground state energies and sectors of the clusters
	vector<vector<tuple<string,double,double>>> clus_ave; //!< cluster averages and variances of all operators
	vector<pair<string,double>> ave; //!< lattice averages of all operators
	vector<double> GS_energy; //!< ground state energies of the clusters + total (component 0)
	size_t n_clus; //!< number of cluster instances in model (derived)
	double omega; //!< value of the Potthoff functional
	double E_pot; //!< potential energy
	double E_kin; //!< kinetic energy
	vector<double> G_host_cumul; //!< cumulative CDMFT host function along the real frequency axis
	vector<vector<matrix<Complex>>> G_host; //!< CDMFT host function
	vector<vector<matrix<Complex>>> G_host_down; //!< CDMFT host function (spin-down component)
	vector<double> CDMFT_freqs; //!< CDMFT frequency grid
	vector<double> CDMFT_weights; //!< weights of the different frequencies in CDMFT

	complex<double> TrSigmaG(Complex w, vector3D<double> &k, bool spin_down);
	complex<double> CDMFT_host_part(Complex w, bool spin_down);
	double Berry_flux(const vector<vector3D<double>> &k, int orb, bool spin_down);
	double Berry_plaquette(Green_function &G, const vector3D<double> &k1, const double deltax, const double deltay, const int opt, int dir, int orb);
	double CDMFT_distance(const vector<double>& p);
	double monopole_part(vector3D<double>& k, double a, int nk, int orb, bool rec, int dir, bool spin_down);
	double monopole(vector3D<double>& k, double a, int nk, int orb, bool rec);
	double potential_energy();
	double Potthoff_functional();
	double spectral_average(const string& name, const complex<double> w);
	Green_function cluster_Green_function(Complex w, bool sig, bool spin_down);
	lattice_model_instance(shared_ptr<lattice_model> _model, int label);
	~lattice_model_instance();
	matrix<complex<double>> cluster_Green_function(size_t i, complex<double> w, bool spin_down, bool blocks);
	matrix<complex<double>> cluster_hopping_matrix(size_t i, bool spin_down);
	matrix<complex<double>> cluster_self_energy(size_t i, complex<double> w, bool spin_down);
	matrix<complex<double>> hybridization_function(size_t i, complex<double> w, bool spin_down);
	matrix<Complex> epsilon(Green_function_k &M);
	matrix<Complex> projected_Green_function(Complex w, bool spin_down);
	matrix<Complex> band_Green_function(Green_function_k &M);
	matrix<Complex> upgrade_cluster_matrix_anomalous(int latt_mix, int clus_mix, matrix<Complex> &g, matrix<Complex> &gm);
	matrix<Complex> upgrade_cluster_matrix(int latt_mix, int clus_mix, matrix<Complex> &g);
	pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> site_and_bond_profile();
	vector<double> Berry_curvature(vector3D<double>& k1, vector3D<double>& k2, int nk, int orb, bool recursive=false, int dir=3);
	vector<double> dispersion(Green_function_k &M);
	vector<double> dos(const complex<double> w);
	vector<double> momentum_profile_per(const lattice_operator& op, const vector<vector3D<double>> &k);
	vector<double> momentum_profile(const lattice_operator& op, const vector<vector3D<double>> &k);
	vector<pair<double,string>> ground_state();
	vector<pair<string,double>> averages(const vector<string> &_ops);
	vector<pair<vector<double>, vector<double>>> Lehmann_Green_function(vector<vector3D<double>> &k, int orb, bool spin_down);
	vector<vector<matrix<Complex>>> get_CDMFT_host(bool spin_down);
	void set_CDMFT_host(const vector<double>& freqs, const int clus, const vector<matrix<Complex>>& H, const bool spin_down);
	void average_integrand_per(Complex w, vector3D<double> &k, const int *nv, double *I);
	void average_integrand(Complex w, vector3D<double> &k, const int *nv, double *I);
	void build_cluster_H();
	void build_H();
	void CDMFT_host(const vector<double>& freqs, const vector<double>& weights);
	void cluster_self_energy(Green_function& G);
	void Green_eigensystem(Green_function &G, const vector3D<double> &k, vector<double> &e, matrix<Complex> &U, int opt);
	void Green_function_solve(); //!< calls the Green_function solver for all clusters
	void inverse_Gcpt(const block_matrix<Complex> &Ginv, Green_function_k &M);
	void periodized_Green_function(Green_function_k &M);
	void potential_energy_integrand(Complex w, vector3D<double> &k, const int *nv, double *I);
	void print_parameters(ostream& out, print_format format);
	void SEF_integrand(Complex w, vector3D<double> &k, const int *nv, double *I);
	void self_energy(Green_function_k& M);
	void set_Gcpt(Green_function_k &M);
	void set_V(Green_function_k &M, bool nohybrid=false);
};

void SEF_integrand(Complex w, vector3D<double> &k, const int *nv, double *I);

#endif /* lattice_model_instance_hpp */
