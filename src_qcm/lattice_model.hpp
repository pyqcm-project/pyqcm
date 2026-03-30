#ifndef lattice_model_hpp
#define lattice_model_hpp

#include <string>
#include <map>
#include <memory>

#include "qcm_ED.hpp"
#include "lattice3D.hpp"
#include "vector3D.hpp"
#include "lattice_operator.hpp"
#include "Green_function.hpp"
#include "parameter_set.hpp"
#include "lattice_hybrid.hpp"

//! site of a repeated unit
struct site{
	size_t cluster; // cluster index of the site (starts at 0)
	size_t index_within_cluster; //
	int orb; // orbital label
	vector3D<int64_t> position; // position of site in the repeated unit
};

//! describes a cluster
struct cluster{
	size_t n_sites; //!< size of cluster
	size_t offset; //!< number of sites before the index for that cluster starts
	vector3D<int64_t> position; //!< base position of cluster
	int ref; //!< label of reference (or equivalent) cluster, from 0 to N_clus - 1
	int mixing; //!< mixing state of the cluster
	bool conj; //!< true if one must take the complex conjugated Green function of the reference cluster, if any
	int sys_start; //!< label of first system associated with the cluster
	int nsys; //!< number of systems associated with the cluster
};

//! describes a system
struct System{
	size_t n_sites; //!< size of cluster
	size_t n_bath; //!< size of cluster's bath
	string name; //!< name of model associated with cluster
	int clus; //!< label of cluster (from 0) to which the system belongs
	int mixing; //!< mixing state of the cluster
	size_t n_sym; //!< number of point group symmetry operations in the system
};


//! description of the lattice model
struct lattice_model{
	basis3D dual; //!< dual of lattice
	basis3D phys; //!< basis of the physical lattice
	basis3D physdual; //!< dual of basis of the physical lattice
	basis3D superdual; //!< dual of superlattice
	bool bath_exists; //!< true if at least one cluster has a bath of uncorrelated orbitals (DMFT)
	bool is_closed; //!< true if operators can no longer be added, as the first instance of the model was created
	int mixing; //!< spin-Nambu mixing of the model (cannot be changed once set)
	int n_mixed; //!< = 2 for mixing=1, 2 and 4 for mixing=3
	int spatial_dimension; //!< dimension of the superlattice
	int nsys; //!< total number of systems in the model
	lattice3D superlattice; //!< superlattice (for the super unit cell)
	lattice3D unit_cell; //!< lattice (useful in case of multiband models)
	map<string, shared_ptr<lattice_operator>> term; //!< pointers to the terms of the Hamiltonian
	map<vector3D<int64_t>,size_t> folded_position_map; //!< maps folded positions to site indices
	map<vector3D<int64_t>,size_t> position_map; //!< maps positions to site indices
	size_t dim_GF; //!< dimension of the Green function
	size_t dim_reduced_GF; //!< dimension of the reduced (periodized) Green function
	size_t Lc; //!< number of lattice sites in the repeated unit = sites.size()/n_band
	size_t n_band; //!< number of bands in the model
	string name; //!< name of the model
	vector<cluster> clusters; //!< list of clusters
	vector<System> systems; //!< list of systems
	vector<int> inequiv; //!< labels of the non-equivalent clusters
	vector<int> is_nambu_index; // 1 if the index is of Nambu type, 0 otherwise
	vector<pair<int, int>> bonds; // list of NN bonds
	vector<shared_ptr<lattice_operator>> one_body_ops; //!< pointers to the one-body or anomalous operators (for computing averages)
	vector<site> sites; //!< list of sites
	vector<size_t> s_; // dimensions of the Green functions for each cluster
	vector<size_t> GF_dims; // dimensions of the Green functions for each cluster
	vector<size_t> GF_offset; // position of the cluster #i in GF index space (depends on mixing)
	vector<size_t> opposite_neighbor; //!< indices of opposite neighbors
	vector<size_t> reduced_Green_index; // lattice Green index (n_band*n_mixed) associated with full Green index (n_mixed*n_sites)
	vector<vector<vector<Complex>>> pauli; //!< Pauli matrices
	vector<vector3D<double>> Green_to_position; // position associated (in superlattice basis) with each Green function index
	vector<vector3D<int64_t>> neighbor; //!< list of neighboring superclusters

	//! pre-computed connectivity for compact_tiling(): one group per distinct spatial displacement
	struct tile_group {
		size_t n_intra;                    //!< number of intra-cluster (n=0) entries at start of entries
		vector<array<size_t,3>> entries;   //!< {s1, s2, n}: spatial site indices and neighbor label
	};
	vector<tile_group> tiling_data;        //!< tiling groups, built once by prepare_tiling_data()
	shared_ptr<parameter_set> param_set; //!< parameter set structure
	vector<string> sector_strings; //!< Hilbert space sectors to explore in the different systems
	string hybrid_file; //!< name of HDF5 file containing the external hybridization
	shared_ptr<lattice_hybrid> hybrid; //!< external hybridization


	static bool model_consolidated;
	
	lattice_model();
	double Potthoff_functional();
	int neighbor_index(vector3D<int64_t>& R);
	matrix<Complex> periodize(const vector3D<double> &k, matrix<Complex> &mat);
	pair<string, int> name_and_label(string &S, bool cut = false);
	size_t to_GF_index(size_t I, size_t spin, bool anomalous, bool spin_down);
	template<typename T> void build_cluster_operators(lattice_operator& op);
	vector<Complex> periodize(const vector3D<double> &k, vector<Complex> &mat);
	void add_anomalous_elements(vector<lattice_matrix_element>& E, int s1, int s2, int ni, int ni_opp, Complex z, latt_op_type SC);
	void add_chemical_potential();
	void anomalous_operator(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb1, int orb2, const string& type);
	void close_model(bool force=false);
	void density_wave(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type);
	void explicit_operator(const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau, int sigma);
	matrix<Complex> compact_tiling(const matrix<Complex>& A, const vector3D<double>& k);
	void find_second_site(int s1, const vector3D<int64_t>& link, int& s2, int& ni, int& ni_opp);
	void neighbor_census();
	void prepare_tiling_data();
	void hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma);
	void current_operator(const string &name, vector3D<int64_t> &link, double amplitude, int b1, int b2, int dir, bool pau=true);
	void interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type);
	void one_body_matrix(lattice_operator& op);
	void post_parameter_consolidate(size_t label);
	void pre_operator_consolidate();
	void print(ostream& fout, bool asy_operators=false, bool asy_labels=false, bool asy_orb=false, bool asy_neighbors=false, bool asy_working_basis=false);
	matrix<Complex> lattice_hybridization(int iw, int ik); //!< extracts a particular hybridization matrix
};



/**
 Builds the cluster operators deriving from the lattice operators
 @param op [in] lattice operator
 */
template<typename T>
void lattice_model::build_cluster_operators(lattice_operator& op)
{
	op.in_cluster.assign(clusters.size(), true);
	// loop over clusters
	set<string> is_built;

	for(int s=0; s<nsys; s++){
		int c = systems[s].clus;
		if(is_built.count(systems[s].name) and op.is_density_wave == false) continue;

		// selecting the matrix elements
		auto data = ED::model_size(systems[s].name);
		size_t n_orb = get<0>(data) + get<1>(data);
		vector<matrix_element<T>> elem;
		for(auto& e : op.elements){
			if(sites[e.site2].cluster == c and sites[e.site1].cluster == c and e.neighbor == 0){
				matrix_element<T> X(sites[e.site1].index_within_cluster + e.spin1*n_orb, sites[e.site2].index_within_cluster + e.spin2*n_orb, native<T>(e.v));
				elem.push_back(X);
			}
			if(global_bool("periodic")){
				if(sites[e.site2].cluster == c and sites[e.site1].cluster == c and e.neighbor != 0){
					matrix_element<T> X(sites[e.site1].index_within_cluster + e.spin1*n_orb, sites[e.site2].index_within_cluster + e.spin2*n_orb, native<T>(e.v));
					elem.push_back(X);
				}
			}
		}
		
		if(elem.size() == 0){
			op.in_cluster[c] = false;
			if(global_bool("verb_warning"))
				cout << "WARNING : operator " << op.name << " has no element in cluster " << c+1 << endl;
			continue;
		}
		else{
			is_built.insert(systems[s].name);
			if(global_bool("verb_Hilbert")) cout << "building operator " << op.name << " on cluster no " << c+1 << " (model " << systems[s].name << ")" << endl;
		}
		string opname = op.name;
		if(op.is_density_wave){
			opname += '@' + to_string(c+1);

		}
		switch(op.type){
			case latt_op_type::one_body:
				ED::new_operator(systems[s].name, opname, string("one-body"), elem);
				break;
			case latt_op_type::singlet :
			case latt_op_type::dz :
			case latt_op_type::dy :
			case latt_op_type::dx :
				ED::new_operator(systems[s].name, opname, string("anomalous"), elem);
				break;
			case latt_op_type::Hubbard:
				ED::new_operator(systems[s].name, opname, string("interaction"), elem);
				break;
			case latt_op_type::Hund:
				ED::new_operator(systems[s].name, opname, string("Hund"), elem);
				break;
			case latt_op_type::Heisenberg:
				ED::new_operator(systems[s].name, opname, string("Heisenberg"), elem);
				break;
			case latt_op_type::X:
				ED::new_operator(systems[s].name, opname, string("X"), elem);
				break;
			case latt_op_type::Y:
				ED::new_operator(systems[s].name, opname, string("Y"), elem);
				break;
			case latt_op_type::Z:
				ED::new_operator(systems[s].name, opname, string("Z"), elem);
				break;
		}
	}
}



#endif /* lattice_model_hpp */
