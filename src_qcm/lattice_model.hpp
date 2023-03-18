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
	size_t n_bath; //!< size of cluster's bath
	size_t offset; //!< number of sites before the index for that cluster starts
	string name; //!< name of model associated with cluster
	vector3D<int64_t> position; //!< base position of cluster
	int ref; //!< label of reference (or equivalent) cluster, from 0 to N_clus - 1
	int mixing; //!< mixing state of the cluster
	size_t n_sym; //!< number of point group symmetry operations in the cluster model
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
	vector<int> inequiv; //!< labels of the non-equivalent clusters
	vector<int> is_nambu_index; // 1 if the index is of Nambu type, 0 otherwise
	vector<pair<int, int>> bonds; // list of NN bonds
	vector<shared_ptr<lattice_operator>> one_body_ops; //!< pointers to the one-body or anomalous operators (for computing averages)
	vector<site> sites; //!< list of sites
	vector<size_t> GF_dims; // dimensions of the Green functions for each cluster
	vector<size_t> GF_offset; // position of the cluster #i in GF index space (depends on mixing)
	vector<size_t> opposite_neighbor; //!< indices of opposite neighbors
	vector<size_t> reduced_Green_index; // lattice Green index (n_band*n_mixed) associated with full Green index (n_mixed*n_sites)
	vector<vector<vector<Complex>>> pauli; //!< Pauli matrices
	vector<vector3D<double>> Green_to_position; // position associated (in superlattice basis) with each Green function index
	vector<vector3D<int64_t>> neighbor; //!< list of neighboring superclusters
	shared_ptr<parameter_set> param_set; //!< parameter set structure
	vector<string> sector_strings; //!< Hilbert space sectors to explore in the different clusters

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
	void find_second_site(int s1, const vector3D<int64_t>& link, int& s2, int& ni, int& ni_opp);
	void hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, int tau, int sigma);
	void interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int orb1, int orb2, const string &type);
	void one_body_matrix(lattice_operator& op);
	void post_parameter_consolidate(size_t label);
	void pre_operator_consolidate();
	void print(ostream& fout, bool asy_operators=false, bool asy_labels=false, bool asy_orb=false, bool asy_neighbors=false, bool asy_working_basis=false);
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

	for(int c=0; c<clusters.size(); c++){

		if(is_built.count(clusters[c].name) and op.is_density_wave == false) continue;

		// selecting the matrix elements
		auto data = ED::model_size(clusters[c].name);
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
		
		if(elem.size() == 0 and global_bool("verb_warning")){
			cout << "WARNING : operator " << op.name << " has no element in cluster " << c+1 << endl;
			continue;
		}
		else{
			is_built.insert(clusters[c].name);
			if(global_bool("verb_Hilbert")) cout << "building operator " << op.name << " on cluster no " << c+1 << " (model " << clusters[c].name << ")" << endl;
		}
		string opname = op.name;
		if(op.is_density_wave){
			opname += '@' + to_string(c+1);

		}
		switch(op.type){
			case latt_op_type::one_body:
				ED::new_operator(clusters[c].name, opname, string("one-body"), elem);
				break;
			case latt_op_type::singlet :
			case latt_op_type::dz :
			case latt_op_type::dy :
			case latt_op_type::dx :
				ED::new_operator(clusters[c].name, opname, string("anomalous"), elem);
				break;
			case latt_op_type::Hubbard:
				ED::new_operator(clusters[c].name, opname, string("interaction"), elem);
				break;
			case latt_op_type::Hund:
				ED::new_operator(clusters[c].name, opname, string("Hund"), elem);
				break;
			case latt_op_type::Heisenberg:
				ED::new_operator(clusters[c].name, opname, string("Heisenberg"), elem);
				break;
			case latt_op_type::X:
				ED::new_operator(clusters[c].name, opname, string("X"), elem);
				break;
			case latt_op_type::Y:
				ED::new_operator(clusters[c].name, opname, string("Y"), elem);
				break;
			case latt_op_type::Z:
				ED::new_operator(clusters[c].name, opname, string("Z"), elem);
				break;
		}
	}

	// second part
	// making sure that equivalent clusters are really equivalent, as far as density-waves are concerned
	if(!op.is_density_wave) return;
	for(int c=0; c<clusters.size(); c++){
		auto data = ED::model_size(clusters[c].name);
		size_t n_orb = get<0>(data) + get<1>(data);
		int cref = clusters[c].ref;
		if(c == cref) continue;
		vector<matrix_element<T>> elem_ref;
		vector<matrix_element<T>> elem;
		for(auto& e : op.elements){
			if(sites[e.site2].cluster == c and sites[e.site1].cluster == c and e.neighbor == 0){
				matrix_element<T> X(sites[e.site1].index_within_cluster + e.spin1*n_orb, sites[e.site2].index_within_cluster + e.spin2*n_orb, native<T>(e.v));
				elem.push_back(X);
			}
		}
		int count=0;
		for(auto& e : op.elements){
			if(sites[e.site2].cluster == cref and sites[e.site1].cluster == cref and e.neighbor == 0){
				if(count >= elem.size()){
					qcm_throw("operator "+op.name+" is incompatible (different number of elements) between clusters "+to_string<int>(cref+1)+" and "+to_string<int>(c));
				}
				if(native<T>(e.v) != elem[count].v){
					qcm_throw("element "+to_string<int>(count)+" of operator "+op.name+" is incompatible between clusters "+to_string<int>(cref+1)+" and "+to_string<int>(c));
				}
				count++;
			}
		}
	}

}



#endif /* lattice_model_hpp */
