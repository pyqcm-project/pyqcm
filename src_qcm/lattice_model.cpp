//
//  lattice_model.cpp
//  qcm3
//
//  Created by David Sénéchal on 16-11-21.
//  Copyright © 2016 David Sénéchal. All rights reserved.
//

#include <iostream>
#include <fstream>

#include "lattice_model.hpp"
#include "qcm_ED.hpp"
#include "QCM.hpp"
#include "parser.hpp"
#include "Green_function.hpp"

#define MAX_GENERATORS 4 //!< maximum number of commuting generators.

size_t Green_function_k::dim_GF;
size_t Green_function_k::dim_reduced_GF;

size_t lattice_index_pair::Nc;
bool lattice_model::model_consolidated = false;

//==============================================================================
/** 
 default constructor
 */
lattice_model::lattice_model() : is_closed(false), bath_exists(false)
{
  dim_reduced_GF=0;
  dim_GF=0;
  Lc = 0;
  n_band = 0;
  spatial_dimension = 0;
  mixing = HS_mixing::normal;
}



//===============================================================================
/**
 Finalizes some initialization of the lattice and superlattice, positions, etc.
 */
void lattice_model::pre_operator_consolidate()
{
  //..............................................................................
  // defining the Pauli matrices
  pauli.assign(4, vector<vector<Complex>>(2,vector<Complex>(2,0.0)));
  pauli[0][0][0] = pauli[0][1][1] = 1.0;
  pauli[1][0][1] = pauli[1][1][0] = 1.0;
  pauli[2][0][1] = Complex(0,-1.0);
  pauli[2][1][0] = Complex(0,1.0);
  pauli[3][0][0] = 1.0;
  pauli[3][1][1] = -1.0;

  //..............................................................................
  // permuting the axes so that the repeated directions are the lowest
  // happens when superlattice.D = 1 and superlattice.e[0] is along the z axis
  
  if(superlattice.D == 1 and superlattice.e[0].z != 0){
    for(int i=0; i<3; ++i) superlattice.e[i].cyclic();
    for(int i=0; i<3; ++i) phys.e[i].cyclic();
    for(int i=0; i<3; ++i) unit_cell.e[i].cyclic();
  }
  
  //..............................................................................
  // adjusting the physical basis and constructing its dual
  
  phys.inverse();
  phys.dual(physdual);
  physdual.init();
  superlattice.dual(superdual);
  superdual.init();
  unit_cell.dual(dual);
  dual.init();
  
  //..............................................................................
  // building the position map (i.e., the reverse of the array 'sites')
  for(size_t i=0; i< sites.size(); i++){
    position_map[sites[i].position] = i;
    vector3D<int64_t> R,S;
    superlattice.fold(sites[i].position, R, S); // r = R + S,  R in superlattice, S in SUC
    folded_position_map[S] = i;
  }
  
  //..............................................................................
  // setting the lattice orbital labels
  vector<vector3D<int64_t>> orb_set;
  for(size_t i=0; i< sites.size(); i++){
    vector3D<int64_t> R,S;
    unit_cell.fold(sites[i].position, R, S); // r = R + S,  R in lattice, S in unit cell
    int b=0;
    for(b=0; b<orb_set.size(); ++b) if(S == orb_set[b]) break;
    if(b==orb_set.size()) orb_set.push_back(S);
    sites[i].orb = b;
  }
  n_band = orb_set.size();
  lattice_index_pair::Nc = sites.size();
  Lc = sites.size()/n_band;
  
  neighbor.push_back(vector3D<int64_t>(0,0,0)); //! adding self to list of neighbors
  neighbor_census(); //! populate all possible inter-cluster neighbor vectors from site-position differences
  if(clusters.size() == 1) prepare_tiling_data(); //! build tiling connectivity for compact_tiling()
  spatial_dimension = (int)superlattice.D;
  
  //..............................................................................
  // computing sys_start and nsys for clusters

  nsys = systems.size();
  for(size_t s=0; s <nsys; s++) clusters[systems[s].clus].nsys++;
  clusters[0].sys_start = 0;
  for(size_t c=1; c <clusters.size(); c++) clusters[c].sys_start = clusters[c-1].sys_start + clusters[c].nsys;


  add_chemical_potential();
}


//===============================================================================
/**
 Finds the index of a neighboring super unit cell, given a position R
 */
int lattice_model::neighbor_index(vector3D<int64_t> &R){
  int ni;
  for(ni=0; ni < neighbor.size(); ++ni) if(R == neighbor[ni]) break;
  if(ni==neighbor.size()) neighbor.push_back(R);
  return ni;
}



//===============================================================================
/**
 finds the second site, given a first site and a link
 @param s1 [in] site
 @param link [in] link
 @param s2 [out] second site
 @param ni [out] neighbor index
 @param ni_opp [out] opposite neighbor index
 */
void lattice_model::find_second_site(int s1, const vector3D<int64_t>& link, int& s2, int& ni, int& ni_opp)
{
  // folding the first site within the SUC
  vector3D<int64_t> R1, S1;
  superlattice.fold(sites[s1].position, R1, S1); // folds r into the SUC
  
  vector3D<int64_t> r = S1;
  r += link; // adding the link to the position
  
  
  // finding the second lattice site s2 and the neighbor link R
  vector3D<int64_t> R2, S2;
  superlattice.fold(r, R2, S2); // folds r into the SUC
  
  auto it_site2 = folded_position_map.find(S2);
  if(it_site2==folded_position_map.end()) {s2 = -1; return;}; // empty site (e.g. graphene). skip.
  s2 = (int)it_site2->second;
  R2 += sites[s1].position - S1 - sites[s2].position + S2; // neighbor link (corrected for shift into conventional unit cell)
  
  ni = neighbor_index(R2);
  R2 *= -1;
  ni_opp = neighbor_index(R2);

}

//===============================================================================
/**
 Adds all possible neighbor vectors to the neighbor list by examining every
 position difference r2-r1 between sites of the super unit cell and applying
 that displacement to every site in the super unit cell.

 Algorithm:
   (1) Loop over sites i1 (position r1) of the super unit cell.
   (2) Loop over sites i2 (position r2) of the super unit cell.
   (3) For each site s of the super unit cell, call find_second_site with
       link = r2-r1: this folds the displaced position back into the SUC,
       computes the neighbor vector, and registers it via neighbor_index if new.
 */
void lattice_model::neighbor_census()
{
  size_t Ns = sites.size();
  for(size_t i1 = 0; i1 < Ns; ++i1){
    for(size_t i2 = 0; i2 < Ns; ++i2){
      vector3D<int64_t> delta = sites[i2].position - sites[i1].position;
      for(size_t s = 0; s < Ns; ++s){
        int s2, ni, ni_opp;
        find_second_site((int)s, delta, s2, ni, ni_opp);
        // neighbor_index (called inside find_second_site) handles deduplication
      }
    }
  }
}


//===============================================================================
/**
 Builds tiling_data: for each distinct spatial displacement d between intra-cluster
 site pairs, stores the list of (s1, s2, n) triplets where s1, s2 are site indices
 and n is the neighbor label (n=0 for intra-cluster, n>0 for inter-cluster wrapping).
 The intra entries (n=0) are placed first; n_intra records their count.
 */
void lattice_model::prepare_tiling_data()
{
  size_t ns = clusters[0].n_sites;

  // Build one group per distinct displacement d, seeding with intra-cluster pairs
  map<vector3D<int64_t>, tile_group> grp_map;
  for(size_t s1 = 0; s1 < ns; ++s1)
    for(size_t s2 = 0; s2 < ns; ++s2)
      grp_map[sites[s2].position - sites[s1].position].entries.push_back({s1, s2, 0});

  // For each displacement, add inter-cluster entries (n>0) from all sites
  for(auto& kv : grp_map){
    const vector3D<int64_t>& d = kv.first;
    auto& grp = kv.second;
    grp.n_intra = grp.entries.size();  // all entries so far are intra (n=0)

    for(size_t s = 0; s < ns; ++s){
      int s2, ni, ni_opp;
      find_second_site((int)s, d, s2, ni, ni_opp);
      if(s2 < 0) continue;   // no site at that position (non-Bravais cluster)
      if(ni == 0) continue;  // intra-cluster, already in the list
      grp.entries.push_back({s, (size_t)s2, (size_t)ni});
    }
  }

  tiling_data.reserve(grp_map.size());
  for(auto& kv : grp_map)
    tiling_data.push_back(std::move(kv.second));
}


//===============================================================================
/**
 Compact-tiling of a dim_GF x dim_GF matrix at wavevector k.

 For each displacement d, all intra-cluster elements A[s,s'] with the same d are
 averaged (step 1), then inter-cluster elements A[s, f(s+d)] are added with the
 Bloch phase exp(i*k*R*2*pi) for the wrapping vector R (step 2). Uses tiling_data
 pre-computed by prepare_tiling_data().

 @param A [in] matrix to compact-tile, of size dim_GF x dim_GF
 @param k [in] wavevector in the superdual basis (same convention as M.k and periodize())
 @returns the compact-tiled matrix Act of size dim_GF x dim_GF
 */
matrix<Complex> lattice_model::compact_tiling(const matrix<Complex>& A, const vector3D<double>& k)
{
  QCM_ASSERT(clusters.size() == 1);

  size_t ns = clusters[0].n_sites;
  size_t nb = dim_GF / ns;           // number of spin/Nambu blocks

  // phase[n] = exp(i * k * neighbor[n] * 2*pi)
  vector<Complex> phase(neighbor.size());
  for(size_t n = 0; n < neighbor.size(); ++n){
    double z = k * neighbor[n] * 2 * M_PI;
    phase[n] = Complex(cos(z), sin(z));
  }

  matrix<Complex> Act(dim_GF);
  for(size_t b1 = 0; b1 < nb; ++b1){
    for(size_t b2 = 0; b2 < nb; ++b2){
      for(auto& grp : tiling_data){
        // Step 1: average A over intra-cluster (n=0) entries for this displacement
        Complex avg = 0.0;
        double iave = 1.0/grp.n_intra;
        if(global_bool("compact_tiling_per_site")) iave = 1.0/ns;

        for(size_t i = 0; i < grp.n_intra; ++i){
          auto& e = grp.entries[i];
          avg += A(e[0] + b1*ns, e[1] + b2*ns);
        }
        avg *= iave;

        // Step 2: add avg*phase[n] for all entries (intra: phase=1, inter: Bloch factor)
        for(auto& e : grp.entries)
          Act(e[0] + b1*ns, e[1] + b2*ns) += avg * phase[e[2]];
      }
    }
  }

  return Act;
}


/**
 Adds the chemical potential to the model (this operator is always present)
 */
void lattice_model::add_chemical_potential()
{
  auto tmp = make_shared<lattice_operator>(*this, "mu", latt_op_type::one_body);
  term[tmp->name] = tmp;
  tmp->is_active = true;
  for(int s=0; s<sites.size(); ++s){
    tmp->elements.push_back({s,0,s,0,0,-1.0});
    tmp->elements.push_back({s,1,s,1,0,-1.0});
  }
}

//===============================================================================
/**
 add anomalous elements associated with a given pair of sites
 @param [in] E : vector of matrix elements
 @param [in] s1 : site no 1
 @param [in] s2 : site no 2
 @param [in] ni : neighbor index
 @param [in] ni_opp : opposite neighbor index
 @param [in] z : amplitude of the element
 @param [in] SC : type of pairing
 */
void lattice_model::add_anomalous_elements(vector<lattice_matrix_element>& E, int s1, int s2, int ni, int ni_opp, Complex z, latt_op_type SC)
{
  switch (SC) {
    case latt_op_type::one_body: // useful for normal charge density waves
      if(ni==0){
        E.push_back({s1, 0, s2, 0, ni, z});
        E.push_back({s1, 1, s2, 1, ni, z});
        E.push_back({s2, 0, s1, 0, ni, conjugate(z)});
        E.push_back({s2, 1, s1, 1, ni, conjugate(z)});
      }
      else{
        E.push_back({s1, 0, s2, 0, ni, z});
        E.push_back({s1, 1, s2, 1, ni, z});
        E.push_back({s2, 0, s1, 0, ni_opp, conjugate(z)});
        E.push_back({s2, 1, s1, 1, ni_opp, conjugate(z)});
      }
      break;

    case latt_op_type::singlet:
      if(ni==0){
        E.push_back({s1, 0, s2, 1, 0, z});
        E.push_back({s1, 1, s2, 0, 0, -z});
        if(s1 != s2){
          E.push_back({s2, 1, s1, 0, 0, -z});
          E.push_back({s2, 0, s1, 1, 0, z});
        }
      }
      else{
        E.push_back({s1, 0, s2, 1, ni_opp, z});
        E.push_back({s1, 1, s2, 0, ni_opp, -z});
        E.push_back({s2, 1, s1, 0, ni, -z});
        E.push_back({s2, 0, s1, 1, ni, z});
      }
      break;

    case latt_op_type::dz:
      E.push_back({s1, 0, s2, 1, ni_opp, -z});
      E.push_back({s1, 1, s2, 0, ni_opp, -z});
      E.push_back({s2, 1, s1, 0, ni, z});
      E.push_back({s2, 0, s1, 1, ni, z});
      break;

    case latt_op_type::dy:
      if(s1 == s2 and ni==0) qcm_throw("triplet SC must be based on different sites");
      E.push_back({s1, 0, s2, 0, ni_opp, z*complex<double>(0,1.0)});
      E.push_back({s1, 1, s2, 1, ni_opp, z*complex<double>(0,1.0)});
      E.push_back({s2, 0, s1, 0, ni, -z*complex<double>(0,1.0)});
      E.push_back({s2, 1, s1, 1, ni, -z*complex<double>(0,1.0)});
      break;

    case latt_op_type::dx:
      if(s1 == s2 and ni==0) qcm_throw("triplet SC must be based on different sites");
      E.push_back({s1, 0, s2, 0, ni_opp, z});
      E.push_back({s1, 1, s2, 1, ni_opp, -z});
      E.push_back({s2, 0, s1, 0, ni, -z});
      E.push_back({s2, 1, s1, 1, ni, z});
      break;
    default:
      qcm_throw("SC case not implemented yet.");
      break;
  }
}




//===============================================================================
/**
 consolidates some model data after reading the operators
 */
void lattice_model::close_model(bool force)
{
  if(is_closed and !force) return;
  is_closed = true;

  // adjusting the norms of all operators defined so that they are mulitplied by 1/L
  for(auto& X : term) X.second->norm = 1.0/sites.size();
  term["mu"]->norm = -1.0/sites.size();
  
  for(auto& X : term) X.second->close();
}


//===============================================================================
/**
 consolidates some model data after reading the first set of parameters
 @param label [in] label of the model instance used
 */
void lattice_model::post_parameter_consolidate(size_t label)
{
  //..............................................................................
  // converting the neighbors to the superlattice basis, for future use
  
  if(model_consolidated){
    qcm_throw("the lattice model can only be consolidated once!");
  }
  model_consolidated = true;

  
  for(auto& i: neighbor){
    vector3D<double> ne = superlattice.to(i);
    i.x = (int)floor(ne.x+0.5);
    i.y = (int)floor(ne.y+0.5);
    i.z = (int)floor(ne.z+0.5);
  }
  // indexing the opposites
  opposite_neighbor.assign(neighbor.size(), 0);
  for(size_t i=1; i<neighbor.size(); i++){
    vector3D<int64_t> opposite = neighbor[i]*(-1);
    for(size_t j=1; j<neighbor.size(); j++){
      if(neighbor[j] == opposite){
        opposite_neighbor[i] = j;
        opposite_neighbor[j] = i;
        break;
      }
    }
  }
  
  //..............................................................................
  // building the list of NN bonds, for eventual use
  
  bonds.reserve(3*sites.size());
  for(size_t i=0; i<sites.size(); i++){
    for(size_t j=0; j<i; j++){
      if(sites[i].cluster != sites[j].cluster) continue;
      vector3D<int64_t> iR = sites[i].position - sites[j].position;
      vector3D<double> R = phys.to(vector3D<double>(iR.x, iR.y, iR.z));
      if(R*R < 1.1) bonds.push_back(pair<int, int>(i,j));
    }
  }

  //..............................................................................
	// erasing the unused operators
	auto it = term.begin();
	while(it != term.end()){
    if(it->first == "mu") it->second->is_active = true;
		if(it->second->is_active == false){
			term.erase(it++);
		}
		else ++it;
	}

  //..............................................................................
	// filling the list of one-body or anomalous operators
	one_body_ops.reserve(term.size());
	for(auto& x : term){
		if(x.second->is_interaction == false) one_body_ops.push_back(x.second);
	}

  //..............................................................................
	// finds the common mixing
	mixing = HS_mixing::normal;
	for(size_t s=0; s<nsys; s++){
    clusters[systems[s].clus].mixing |= ED::mixing(s+label*nsys);
		mixing |= clusters[s].mixing;
	}
	if(mixing > 5) mixing = mixing & HS_mixing::full;
	if(mixing == 5) mixing = HS_mixing::anomalous;

	n_mixed = 1;
	if(mixing & HS_mixing::anomalous) n_mixed *= 2;
	if(mixing & HS_mixing::spin_flip) n_mixed *= 2;
	
  //..............................................................................
  // computes offsets and dimensions

  dim_GF = n_mixed*sites.size();
	dim_reduced_GF = n_mixed*n_band;
	Green_function_k::dim_GF = dim_GF;
	Green_function_k::dim_reduced_GF = dim_reduced_GF;

  clusters[0].offset = 0;
  for(size_t i=1; i<clusters.size(); i++) clusters[i].offset = clusters[i-1].offset + clusters[i-1].n_sites;
	GF_dims.resize(clusters.size());
	for(size_t i=0; i<clusters.size(); i++) GF_dims[i] = n_mixed*clusters[i].n_sites;
	GF_offset.resize(clusters.size());
	GF_offset[0] = 0;
	for(size_t i=1; i<clusters.size(); i++) GF_offset[i] = GF_offset[i-1] + GF_dims[i-1];
	
	// build GF matrix elements for lattice operators
	for(auto& op : term) one_body_matrix(*op.second);
	
	// builds the array reduced_Green_index
	reduced_Green_index.resize(dim_GF);
	Green_to_position.resize(dim_GF);
	for(size_t i = 0; i<sites.size(); i++){
		size_t off = GF_offset[sites[i].cluster];
		size_t ic = sites[i].index_within_cluster;
		size_t ns = clusters[sites[i].cluster].n_sites;
		switch(mixing){
			case HS_mixing::normal:
			case HS_mixing::up_down:
				reduced_Green_index[off+ic] = sites[i].orb;
				Green_to_position[off+ic] = superlattice.to(sites[i].position);
				break;
			case HS_mixing::spin_flip:
			case HS_mixing::anomalous:
				reduced_Green_index[off+ic] = sites[i].orb;
				reduced_Green_index[off+ic+ns] = sites[i].orb + n_band;
				Green_to_position[off+ic] = superlattice.to(sites[i].position);
				Green_to_position[off+ic+ns] = Green_to_position[off+ic];
				break;
			case HS_mixing::full:
				reduced_Green_index[off+ic] = sites[i].orb;
				reduced_Green_index[off+ic+ns] = sites[i].orb + n_band;
				reduced_Green_index[off+ic+2*ns] = sites[i].orb + 2*n_band;
				reduced_Green_index[off+ic+3*ns] = sites[i].orb + 3*n_band;
				Green_to_position[off+ic] = superlattice.to(sites[i].position);
				Green_to_position[off+ic+ns] = Green_to_position[off+ic];
				Green_to_position[off+ic+2*ns] = Green_to_position[off+ic];
				Green_to_position[off+ic+3*ns] = Green_to_position[off+ic];
				break;
		}
	}
	
  //..............................................................................
	// computing is_nambu_index
	is_nambu_index.assign(dim_GF,0);
	for(size_t i = 0; i<sites.size(); i++){
		size_t off = GF_offset[sites[i].cluster];
		size_t ns = clusters[sites[i].cluster].n_sites;
		switch(mixing){
			case HS_mixing::normal:
			case HS_mixing::up_down:
			case HS_mixing::spin_flip:
				break;
			case HS_mixing::anomalous:
				for(size_t j=0; j<ns; j++) is_nambu_index[off+j+ns] = 1;
				break;
			case HS_mixing::full:
				for(size_t j=0; j<2*ns; j++) is_nambu_index[off+j+2*ns] = 1;
				break;
		}
	}

  //..............................................................................
	// reading the external hybridization, if applicable

  if(hybrid_file.empty() == false){
    hybrid = make_shared<lattice_hybrid>(hybrid_file);
    if(n_mixed*hybrid->d != dim_GF) qcm_throw("incorrect dimension of the external hybridization matrix");
  }
}

//===============================================================================
/**
converts an index as used in lattice_operator::elements to an Green function index
 @param I index in lattice_operator
 @param anomalous true if the index pertains to a nambu index
 @param spin_down true if the index pertains to a spin down degree of freedom (in case of HS_mixing::up_down)
 @returns the index
 */
size_t lattice_model::to_GF_index(size_t I, size_t spin, bool anomalous, bool spin_down)
{
	size_t clusterI = sites[I].cluster;
	size_t J = sites[I].index_within_cluster;
	switch(mixing){
		case 0:
			break;
		case HS_mixing::spin_flip:
			if(spin) J += clusters[clusterI].n_sites;
			break;
		case HS_mixing::anomalous:
			if(spin) J += clusters[clusterI].n_sites;
			break;
		case HS_mixing::full:
			if(spin) J += clusters[clusterI].n_sites;
			if(anomalous) J += 2*clusters[clusterI].n_sites;
			break;
		case HS_mixing::up_down:
			break;
	}
	return GF_offset[clusterI]+J;
}


//===============================================================================
/**
 builds the GF matrix elements of the lattice operator op
 @param [in] the lattice operator
 */
void lattice_model::one_body_matrix(lattice_operator& op)
{
  if(!op.is_interaction && !(op.mixing&HS_mixing::anomalous)){ // one-body terms
		switch(mixing){
			case HS_mixing::normal:
				for(auto& x : op.elements){
					if(x.spin1)	continue;
					size_t I = to_GF_index(x.site1, x.spin1, false, false);
					size_t J = to_GF_index(x.site2, x.spin2, false, false);
					if(x.neighbor) op.IGF_elem.push_back({I, J, x.neighbor, x.v});
					else op.GF_elem.push_back({I, J, x.v});
				}
				break;
			case HS_mixing::spin_flip:
        for(auto& x : op.elements){
          size_t I = to_GF_index(x.site1, x.spin1, false, false);
          size_t J = to_GF_index(x.site2, x.spin2, false, false);
          if(x.neighbor) op.IGF_elem.push_back({I, J, x.neighbor, x.v});
          else op.GF_elem.push_back({I, J, x.v});
        }
        break;
			case HS_mixing::anomalous:
        for(auto& x : op.elements){
          if(x.spin1==0){
            size_t I = to_GF_index(x.site1, x.spin1, false, false);
            size_t J = to_GF_index(x.site2, x.spin2, false, false);
            if(x.neighbor) op.IGF_elem.push_back({I, J, x.neighbor, x.v});
            else op.GF_elem.push_back({I, J, x.v});
          }
          else{
            size_t I = to_GF_index(x.site1, x.spin1, true, false);
            size_t J = to_GF_index(x.site2, x.spin2, true, false);
            if(x.neighbor) op.IGF_elem.push_back({J, I, opposite_neighbor[x.neighbor], -x.v});
            else op.GF_elem.push_back({J, I, -x.v});
          }
        }
        break;
			case HS_mixing::full:
				for(auto& x : op.elements){
					size_t I = to_GF_index(x.site1, x.spin1, false, false);
					size_t J = to_GF_index(x.site2, x.spin2, false, false);
					if(x.neighbor) op.IGF_elem.push_back({I, J, x.neighbor, x.v});
					else op.GF_elem.push_back({I, J, x.v});
				}
        for(auto& x : op.elements){
          size_t I = to_GF_index(x.site1, x.spin1, true, false);
          size_t J = to_GF_index(x.site2, x.spin2, true, false);
          if(x.neighbor) op.IGF_elem.push_back({J, I, opposite_neighbor[x.neighbor], -x.v});
          else op.GF_elem.push_back({J, I, -x.v});
        }
				break;
			case HS_mixing::up_down:
				for(auto& x : op.elements){
					if(x.spin1)	continue;
					size_t I = to_GF_index(x.site1, x.spin1, false, false);
					size_t J = to_GF_index(x.site2, x.spin2, false, false);
					if(x.neighbor) op.IGF_elem.push_back({I, J, x.neighbor, x.v});
					else op.GF_elem.push_back({I, J, x.v});
				}
				for(auto& x : op.elements){
					if(!x.spin1)	continue;
					size_t I = to_GF_index(x.site1, x.spin1, false, true);
					size_t J = to_GF_index(x.site2, x.spin2, false, true);
					if(x.neighbor) op.IGF_elem_down.push_back({I, J, x.neighbor, x.v});
					else op.GF_elem_down.push_back({I, J, x.v});
				}
        break;
		}
	}
	else if(op.mixing&HS_mixing::anomalous){
		switch(mixing){
			case HS_mixing::normal:
			case HS_mixing::spin_flip:
			case HS_mixing::up_down:
				break;
			case HS_mixing::anomalous:
        for(auto& x : op.elements){
          if(x.spin1==0 or x.spin2==1) continue;
          size_t I = to_GF_index(x.site1, x.spin1, true, false); // first index is Nambu
          size_t J = to_GF_index(x.site2, x.spin2, false, false); // second index is normal
          if(x.neighbor){
            op.IGF_elem.push_back({I, J, opposite_neighbor[x.neighbor], 2.0*x.v});
            op.IGF_elem.push_back({J, I, x.neighbor, 2.0*conjugate(x.v)}); // hermitian conjugate
          }
          else{
            op.GF_elem.push_back({I, J, 2.0*x.v});
            op.GF_elem.push_back({J, I, 2.0*conjugate(x.v)}); // hermitian conjugate
          }
        }
        break;
			case HS_mixing::full:
				for(auto& x : op.elements){
					size_t I = to_GF_index(x.site1, x.spin1, true, false);
					size_t J = to_GF_index(x.site2, x.spin2, false, false);
          if(x.neighbor){
            op.IGF_elem.push_back({I, J, opposite_neighbor[x.neighbor], x.v});
            op.IGF_elem.push_back({J, I, x.neighbor, conjugate(x.v)}); // hermitian conjugate
          }
          else{
            op.GF_elem.push_back({I, J, x.v});
            op.GF_elem.push_back({J, I, conjugate(x.v)}); // hermitian conjugate
          }
          I = to_GF_index(x.site2, x.spin2, true, false);
          J = to_GF_index(x.site1, x.spin1, false, false);
          if(x.neighbor){
            op.IGF_elem.push_back({I, J, x.neighbor, -x.v});
            op.IGF_elem.push_back({J, I, opposite_neighbor[x.neighbor], conjugate(-x.v)}); // hermitian conjugate
          }
          else{
            op.GF_elem.push_back({I, J, -x.v});
            op.GF_elem.push_back({J, I, conjugate(-x.v)}); // hermitian conjugate
          }
				}
				break;
		}
	}
	else return;
}



//===============================================================================
/**
 Periodizes a matrix (Green function, CPT Green function or self-energy)
 into the spin-orbital matrix gn  (of dimension nband*n_mixed)
 @param k [in] wavevector in the superdual basis
 @param mat [in] input matrix in full orbital space (Green function, CPT Green function or self-energy)
 @returns a complex-valued, periodized matrix
 */
matrix<Complex> lattice_model::periodize(const vector3D<double> &k, matrix<Complex> &mat)
{
	double L1 = 1.0/Lc;
	
	matrix<Complex> gn(dim_reduced_GF);

	for(size_t i1 = 0; i1 < dim_GF; i1++){
		size_t I1 = reduced_Green_index[i1];
		for(size_t i2 = 0; i2 < dim_GF; i2++){
			size_t I2 = reduced_Green_index[i2];
			double phi = k*(Green_to_position[i2] - Green_to_position[i1])*2*M_PI;
			Complex g = Complex(cos(phi), sin(phi));
			gn(I1,I2) += mat(i1,i2)*g*L1;
		}
	}
	return gn;
}




//===============================================================================
/**
 Periodizes a vector
 into the spin-band vector gn  (output, of dimension nband*n_mixed)
 @param k [in] wavevector in the dual supersubtype basis (superdual)
 @param mat [in] input vector in full orbital space
 @returns a complex-valued, periodized vector
 */
vector<Complex> lattice_model::periodize(const vector3D<double> &k, vector<Complex> &mat)
{
	vector<Complex> gn(dim_reduced_GF);
	
	for(size_t i1 = 0; i1 < dim_GF; i1++){
		size_t I1 = reduced_Green_index[i1];
		double phi = k*Green_to_position[i1]*2*M_PI;
		Complex g = Complex(cos(phi), sin(-phi));
		gn[I1] += mat[i1]*g;
	}
	gn *= 1.0/Lc;
	return gn;
}




//===============================================================================
/**
 prints info about the model in a file (for debugging)
 @param fout [in] output channel
 @param asy_operators [in] if true, prints an asymptote file for each operator
 @param bool asy_labels [in] if true, adds labels to the asymptote graphics
 @param bool asy_orb [in] if true, adds lattice orbital labels to the asymptote graphics
 @param bool asy_neighbors [in] if true, adds the neighboring clusters to the asymptote graphics
 @param bool asy_working_basis [in] if true, plots in the working basis instead of the physical basis
*/
void lattice_model::print(ostream &fout, bool asy_operators, bool asy_labels, bool asy_orb, bool asy_neighbors, bool asy_working_basis)
{
  banner('=', "clusters", fout);
  fout << "No\tn_sites\tposition\tref.\tsystems\n";
  for(int c=0; c<clusters.size(); c++){
    cluster& C = clusters[c];
    fout << c+1 << '\t' << C.n_sites << '\t' << C.position << '\t' << C.ref << '\t' << C.conj;
    for(int s=0; s<C.nsys; s++) fout << systems[s+C.sys_start].name << ", ";
    fout << endl;
  }
  banner('=', "sites", fout);
  fout << "No\tcluster\tNo in cluster\torbital\tposition\n";
  for(int i=0; i<sites.size(); i++){
    site& s = sites[i];
    fout << i+1 << '\t' << s.cluster+1 << '\t' << s.index_within_cluster+1 << '\t' << s.orb+1 << '\t' << s.position << endl;
  }
  banner('.', "NN bonds", fout);
  fout << "No\tcluster\tNo in cluster\torbital\tposition\n";
  for(auto& b : bonds){
    fout << b.first+1 << '\t' << sites[b.first].position << '\t' << b.second+1 << '\t' << sites[b.second].position << endl;
  }
	banner('.', "bases", fout);
	fout << "dual:\n" << dual << endl;
	fout << "phys:\n" << phys << endl;
	fout << "physdual:\n" << physdual << endl;
	fout << "superdual:\n" << superdual << endl;
	fout << "superlattice:\n" << superlattice << endl;
	fout << "unit_cell:\n" << unit_cell << endl;

	banner('.', "neighbors", fout);
	for(size_t i=1; i<neighbor.size(); i++) fout << i << " : " << neighbor[i] << endl;

  banner('=', "lattice operators", fout);
  for(auto& x : term){
		lattice_operator& h = *x.second;
		fout << '\n' << h.name << '\t';
		if(h.type == latt_op_type::one_body) fout << "(one-body)";
		else if(h.type == latt_op_type::singlet) fout << "(singlet)";
		else if(h.type == latt_op_type::dx) fout << "(dx)";
		else if(h.type == latt_op_type::dy) fout << "(dy)";
		else if(h.type == latt_op_type::dz) fout << "(dz)";
		else if(h.type == latt_op_type::Hubbard) fout << "(Hubbard or extended Hubbard)";
		else if(h.type == latt_op_type::Hund) fout << "(Hund)";
		else if(h.type == latt_op_type::Heisenberg) fout << "(Heisenberg)";
		else if(h.type == latt_op_type::X) fout << "(spin X)";
		else if(h.type == latt_op_type::Y) fout << "(spin Y)";
		else if(h.type == latt_op_type::Z) fout << "(spin Z)";
    fout << "\tnorm = " << h.norm;
    if(fabs(h.nambu_correction)>SMALL_VALUE) fout << "\tNambu correction = " << h.nambu_correction;
    if(fabs(h.nambu_correction_full)>SMALL_VALUE) fout << "\tNambu correction (full) = " << h.nambu_correction_full;
    fout << endl;
		fout << "elements (primitive form):\n";
		for(auto &x : h.elements) fout << x << endl;
		if(h.is_interaction) continue;
		fout << "elements (in Green function basis):\n";
		for(auto &x : h.GF_elem){
			if(fabs(x.v.imag()) < SMALL_VALUE) fout << '[' << x.r+1 << ',' << x.c+1 << "] : " << x.v.real() << endl;
			else fout << '[' << x.r+1 << ',' << x.c+1 << "] : " << x.v << endl;
		}
		for(auto &x : h.IGF_elem){
			if(fabs(x.v.imag()) < SMALL_VALUE) fout << '[' << x.r+1 << ',' << x.c+1 << ',' << neighbor[x.n] << "] : " << x.v.real() << endl;
			else fout << '[' << x.r+1 << ',' << x.c+1 << ',' << neighbor[x.n] << "] : " << x.v << endl;
		}
		if(h.GF_elem_down.size()) fout << "elements (in Green function basis, spin down):\n";
		for(auto &x : h.GF_elem_down){
			if(fabs(x.v.imag()) < SMALL_VALUE) fout << '[' << x.r+1 << ',' << x.c+1 << "] : " << x.v.real() << endl;
			else fout << '[' << x.r+1 << ',' << x.c+1 << "] : " << x.v << endl;
		}
		for(auto &x : h.IGF_elem_down){
			if(fabs(x.v.imag()) < SMALL_VALUE) fout << '[' << x.r+1 << ',' << x.c+1 << ',' << neighbor[x.n] << "] : " << x.v.real() << endl;
			else fout << '[' << x.r+1 << ',' << x.c+1 << ',' << neighbor[x.n] << "] : " << x.v << endl;
		}
  }
  banner('*', "cluster models", fout);
  ED::print_models(fout);
}







//===============================================================================
/**
 From a string, extract the name of the operator and the system label (the latter separated from the name by '_')
 @param S [in] input string
 @param cut [in] if true, cuts the cluster suffix of the name
 @returns a pair <name, label>
 */
pair<string, int> lattice_model::name_and_label(string &S, bool cut)
{
  int label = 0;
  string name(S);
  size_t pos = name.rfind('_');
  if(pos != string::npos){
    label =  from_string<int>(name.substr(pos+1));
    if(cut) name.erase(pos);
  }
  if(label > nsys) qcm_throw("system label in "+S+" is outside of range");
  if(label == 0 and term.find(name)  == term.end()) qcm_throw("operator "+name+" does not exist in model");
  if(term.find(name) != term.end()){
    if(term.at(name)->is_density_wave and label>0 and cut) name += '@' + to_string(label);
  }
  return {name,label};
}




//===============================================================================
/**
 Defines a lattice interaction operator
 @param name [in] name of the operator
 @param link [in] bond vector
 @param amplitude [in] overall multiplier
 @param b1 [in] lattice orbital label of the first electron
 @param b2 [in] lattice orbital label of the second electron
 @param type [in] type of interaction (Hubbard, i.e., density-density, if nothing is specified)
*/
void lattice_model::interaction_operator(const string &name, vector3D<int64_t> &link, double amplitude, int b1, int b2, const string &type)
{
  shared_ptr<lattice_operator> tmp_op;
  
  // static string previous_name("");
  auto it_op = term.find(name);
  if(it_op == term.end()){
    check_name(name);
    tmp_op = make_shared<lattice_operator>(*this, name, latt_op_type::Hubbard);
    tmp_op->is_interaction = true;
    term[name] = tmp_op;
    if(type == "Hund") tmp_op->type = latt_op_type::Hund;
    else if(type == "Heisenberg") tmp_op->type = latt_op_type::Heisenberg;
    else if(type == "X") tmp_op->type = latt_op_type::X;
    else if(type == "Y") tmp_op->type = latt_op_type::Y;
    else if(type == "Z") tmp_op->type = latt_op_type::Z;

  }
  else{
    tmp_op = it_op->second;
  }
  
  // looping over sites of the super unit cell
  for(int s1=0; s1 < (int)sites.size(); s1++){
    if(sites[s1].orb != b1) continue;
    int s2, ni, ni_opp;
    find_second_site(s1, link, s2, ni, ni_opp);
    if(s2<0) continue;
    if(sites[s2].orb != b2) continue; // wrong orbital. skip.
    
    switch(tmp_op->type){
      case latt_op_type::Hubbard:
        if(s1==s2)
          tmp_op->elements.push_back({s1, 0, s2, 1, ni, amplitude});
        else{
          tmp_op->elements.push_back({s1, 0, s2, 0, ni, amplitude});
          tmp_op->elements.push_back({s1, 1, s2, 0, ni, amplitude});
          tmp_op->elements.push_back({s1, 0, s2, 1, ni, amplitude});
          tmp_op->elements.push_back({s1, 1, s2, 1, ni, amplitude});
        }
        break;
      case latt_op_type::Hund:
      case latt_op_type::Heisenberg:
      case latt_op_type::X:
      case latt_op_type::Y:
      case latt_op_type::Z:
        QCM_ASSERT(s1 != s2);
        tmp_op->elements.push_back({s1, 0, s2, 0, ni, amplitude});
        break;
      default:
        break;
    }
  }
}




//===============================================================================
/**
 Defines a lattice hopping operator
 @param name [in] name of the operator
 @param link [in] bond vector
 @param amplitude [in] overall multiplier
 @param b1 [in] lattice orbital label of the first electron
 @param b2 [in] lattice orbital label of the second electron
 @param tau [in] Pauli matrix for the orbital part
 @param sigma [in] Pauli matrix for the spin part
 */
void lattice_model::hopping_operator(const string &name, vector3D<int64_t> &link, double amplitude, int b1, int b2, int tau, int sigma)
{
  shared_ptr<lattice_operator> tmp_op;
  // static string previous_name("");
  auto it_op = term.find(name);
  if(it_op == term.end()){
    check_name(name);
    tmp_op = make_shared<lattice_operator>(*this, name, latt_op_type::one_body);
    term[name] = tmp_op;
  }
  else{
    tmp_op = it_op->second;
  }

  QCM_ASSERT(tau>=0 and tau<=3 and sigma >=0 and sigma <=3);
  if(tau==0 and !link.is_null()) qcm_throw("Problem with the definition of " + tmp_op->name+ ". Cannot define an operator with nonzero link AND matrix tau = 0");
  if(tau!=0 and link.is_null()) qcm_throw("Problem with the definition of " + tmp_op->name + ". Cannot define an operator with on-site operator AND matrix tau != 0");
  if(tau==3) qcm_throw("Problem with the definition of " + tmp_op->name + ". Cannot define an operator with matrix tau = 3");
  
  // looping over sites of the super unit cell
  for(int s1=0; s1 < (int)sites.size(); s1++){
    if(sites[s1].orb != b1) continue;
    int s2, ni, ni_opp;
    find_second_site(s1, link, s2, ni, ni_opp);
    if(s2<0) continue;
    if(sites[s2].orb != b2) continue; // wrong orbital. skip.
    
    if(tau == 0){
      for(int alpha=0; alpha<2; alpha++){
        for(int beta=0; beta<2; beta++){
          Complex ec = pauli[sigma][alpha][beta]*amplitude;
          if(ec.imag() == 0.0 and ec.real() == 0.0) continue;
          tmp_op->elements.push_back({s1,alpha,s1,beta,ni,ec});
        }
      }
    }
    else{
      for(int i=0; i<2; i++){
        int si = s1+i*(s2-s1);
        for(int j=0; j<2; j++){
          int sj = s1+j*(s2-s1);
          for(int alpha=0; alpha<2; alpha++){
            for(int beta=0; beta<2; beta++){
              Complex ec = pauli[tau][i][j]*pauli[sigma][alpha][beta]*amplitude;
              if(ec.imag() == 0.0 and ec.real() == 0.0) continue;
              if(j<i) tmp_op->elements.push_back({si,alpha,sj,beta,ni_opp,ec});
              else tmp_op->elements.push_back({si,alpha,sj,beta,ni,ec});
            }
          }
        }
      }
    }
  }
}



//===============================================================================
/**
 Defines a lattice current operator
 @param name [in] name of the operator
 @param link [in] bond vector
 @param amplitude [in] overall multiplier
 @param b1 [in] lattice orbital label of the first electron
 @param b2 [in] lattice orbital label of the second electron
 @param dir [in] direction of the current (0,1,2) for (x,y,z)
 @param pau [in] = true for current from real hopping, false for current from imaginary hopping
 */
void lattice_model::current_operator(const string &name, vector3D<int64_t> &link, double amplitude, int b1, int b2, int dir, bool pau)
{
  int pau_index = 2;
  double pau_mult = 1.0;
  if(!pau){
    pau_index = 1;
    pau_mult = -1.0;
  }

  shared_ptr<lattice_operator> tmp_op;
  auto it_op = term.find(name);
  if(it_op == term.end()){
    check_name(name);
    tmp_op = make_shared<lattice_operator>(*this, name, latt_op_type::one_body);
    term[name] = tmp_op;
  }
  else{
    tmp_op = it_op->second;
  }
  // previous_name = name;
  
  // looping over sites of the super unit cell
  for(int s1=0; s1 < (int)sites.size(); s1++){
    if(sites[s1].orb != b1) continue;
    int s2, ni, ni_opp;
    find_second_site(s1, link, s2, ni, ni_opp);
    if(s2<0) continue;
    if(sites[s2].orb != b2) continue; // wrong orbital. skip.
    int V;
    switch(dir){
      case 0: V = link.x; break;
      case 1: V = link.y; break;
      case 2: V = link.z; break;
    //   case 0: V = sites[s1].position.x-sites[s2].position.x; break;
    //   case 1: V = sites[s1].position.y-sites[s2].position.y; break;
    //   case 2: V = sites[s1].position.z-sites[s2].position.z; break;
    }
    for(int i=0; i<2; i++){
      int si = s1+i*(s2-s1);
      for(int j=0; j<2; j++){
        int sj = s1+j*(s2-s1);
        for(int alpha=0; alpha<2; alpha++){
          for(int beta=0; beta<2; beta++){
            Complex ec = pauli[pau_index][i][j]*pauli[0][alpha][beta]*amplitude*pau_mult;
            ec *= V;
            if(ec.imag() == 0.0 and ec.real() == 0.0) continue;
            if(j<i) tmp_op->elements.push_back({si,alpha,sj,beta,ni_opp,ec});
            else tmp_op->elements.push_back({si,alpha,sj,beta,ni,ec});
          }
        }
      }
    }
  }
}



//===============================================================================
/**
 Defines a lattice anomalous operator
 @param name [in] name of the operator
 @param link [in] bond vector
 @param amplitude [in] overall multiplier
 @param b1 [in] lattice orbital label of the first electron
 @param b2 [in] lattice orbital label of the second electron
 @param SC_str [in] type of superconductor (singlet, dz, dx, dy)
 */
void lattice_model::anomalous_operator(const string &name, vector3D<int64_t> &link, complex<double> amplitude, int b1, int b2, const string& SC_str)
{
  latt_op_type SC = latt_op_type::singlet;
  if(SC_str == "singlet") SC = latt_op_type::singlet;
  else if(SC_str == "dz") SC = latt_op_type::dz;
  else if(SC_str == "dy") SC = latt_op_type::dy;
  else if(SC_str == "dx") SC = latt_op_type::dx;
  else qcm_throw("SC type must be either singlet, dz, dy or dz. Error in model file");

  shared_ptr<lattice_operator> tmp_op;
  // static string previous_name("");
  auto it_op = term.find(name);
  if(it_op == term.end()){
    check_name(name);
    tmp_op = make_shared<lattice_operator>(*this, name, SC);
    tmp_op->mixing |= HS_mixing::anomalous;
    term[name] = tmp_op;
  }
  else{
    tmp_op = it_op->second;
  }
  

  
  // looping over sites of the super unit cell
  for(int s1=0; s1 < (int)sites.size(); s1++){
    if(sites[s1].orb != b1) continue;
    int s2, ni, ni_opp;
    find_second_site(s1, link, s2, ni, ni_opp);
    if(s2<0) continue;
    if(sites[s2].orb != b2) continue; // wrong orbital. skip.
    add_anomalous_elements(tmp_op->elements, s1, s2, ni, ni_opp, 0.5*amplitude, SC);
  }
}




//===============================================================================
/**
 Defines a density wave operator
 @param name [in] name of the operator
 @param cdw_link [in] bond for the density wave
 @param amplitude [in] overall multiplier of the operator
 @param orb [in] lattice orbital label affected
 @param Q [in] wavevector
 @param phase [in] phase as it appears in the complex exponential
 @param type [in] type of density wave 
 */
void lattice_model::density_wave(const string &name, vector3D<int64_t> &cdw_link, complex<double> amplitude, int orb, vector3D<double> Q, double phase, const string& type)
{
  shared_ptr<lattice_operator> tmp_op;
  vector3D<int64_t> r, R, S;
  
  // static string previous_name("");
  char dw_type = 'N';
  // checks whether another line has started the definition of the same operator
  auto it_op = term.find(name);
  if(it_op == term.end()){
    check_name(name);
    tmp_op = make_shared<lattice_operator>(*this, name, latt_op_type::one_body);
    tmp_op->is_density_wave = true;
    term[name] = tmp_op;
  }
  else{
    tmp_op = it_op->second;
  }
  
  
  if(type == "spin") dw_type = 'Z';
  else if(type == "N") dw_type = 'N';
  else if(type == "cdw") dw_type = 'N';
  else if(type == "Z") dw_type = 'Z';
  else if(type == "X") dw_type = 'X';
  else if(type == "Y") dw_type = 'Y';
  else if(type == "singlet") tmp_op->type = latt_op_type::singlet;
  else if(type == "dz") tmp_op->type = latt_op_type::dz;
  else if(type == "dy") tmp_op->type = latt_op_type::dy;
  else if(type == "dx") tmp_op->type = latt_op_type::dx;
  else qcm_throw(type+" : unknown type of density wave");
  if(cdw_link != vector3D<int64_t>(0,0,0)) dw_type = 'L';

  if(tmp_op->type == latt_op_type::singlet or tmp_op->type == latt_op_type::dz or tmp_op->type == latt_op_type::dy or tmp_op->type == latt_op_type::dx) tmp_op->mixing |= HS_mixing::anomalous;

  Q = physdual.from(0.5*Q);
  phase *= M_PI;

  // checking that the wavevector Q is compatible with the superlattice
  vector3D<double> Qs =  superdual.to(Q); // Qs is Q in the superdual basis. Its components must be integers (or numerically close to).
  if(abs(Qs.x - floor(Qs.x+1e-6)) > 1e-5 or abs(Qs.y - floor(Qs.y+1e-6)) > 1e-5 or abs(Qs.z - floor(Qs.z+1e-6)) > 1e-5){
    qcm_throw("density-wave vector for " + name + " is incommensurate with the superlattice");
  }
  
  tmp_op->norm = 1.0/sites.size();
  auto& E = tmp_op->elements;
  
  // loop over sites
  for(int i=0; i<sites.size(); i++){
    if(orb >= 0 && sites[i].orb != orb) continue; // wrong lattice orbital
    Complex z = amplitude*cos(Q*sites[i].position+phase);
    if(abs(z) > 1e-8){
      if(dw_type=='Z'){
        E.push_back({i,0,i,0,0,z});
        E.push_back({i,1,i,1,0,-z});
      }
      else if(dw_type=='X'){
        E.push_back({i,0,i,1,0,z});
        E.push_back({i,1,i,0,0,z});
      }
      else if(dw_type=='Y'){
        E.push_back({i,0,i,1,0,z*Complex(0,-1.0)});
        E.push_back({i,1,i,0,0,z*Complex(0,1.0)});
      }
      else if(dw_type=='N'){
        E.push_back({i,0,i,0,0,z});
        E.push_back({i,1,i,1,0,z});
      }
    }
  }
  
  // loop over sites (bond CDWs)
  if(dw_type=='L' or tmp_op->mixing&HS_mixing::anomalous){
    for(int s1=0; s1 < (int)sites.size(); s1++){
      if(orb >= 0 && sites[s1].orb != orb) continue; // wrong lattice orbital
      vector3D<int64_t>& r = sites[s1].position;
      int s2, ni, ni_opp;
      find_second_site(s1, cdw_link, s2, ni, ni_opp);
      if(s2<0) continue;
      
      Complex z;
      z = amplitude*Complex(cos(Q*r+phase),-sin(Q*r+phase)); // sign change 2016-04-15
      if(abs(z) < SMALL_VALUE) continue;
      add_anomalous_elements(tmp_op->elements, s1, s2, ni, ni_opp, 0.5*z, tmp_op->type);
    }
  }
}





//===============================================================================
/**
 Defines a one-body operator explicitly
 @param name [in] name of the operator
 @param type [in] type or operator (see enumerate latt_op_type)
 @param elem [in] list of elements specified by position label, link and amplitude.
 @param tau [in] same meaning as for hopping terms.
 @param sigma [in] same meaning as for hopping terms.
 */
void lattice_model::explicit_operator(const string &name, const string &type, const vector<tuple<vector3D<int64_t>, vector3D<int64_t>, complex<double>>> &elem, int tau, int sigma)
{
  check_name(name);
  auto tmp_op = make_shared<lattice_operator>(*this, name);
  term[name] = tmp_op;
  tmp_op->type = lattice_operator::op_type_map.at(type);
  tmp_op->is_density_wave = true;

  if(tmp_op->type == latt_op_type::Hubbard 
    or tmp_op->type == latt_op_type::Hund 
    or tmp_op->type == latt_op_type::Heisenberg
    or tmp_op->type == latt_op_type::X
    or tmp_op->type == latt_op_type::Y
    or tmp_op->type == latt_op_type::Z
  ) tmp_op->is_interaction = true;
  if(tmp_op->type == latt_op_type::singlet or tmp_op->type == latt_op_type::dz or tmp_op->type == latt_op_type::dy or tmp_op->type == latt_op_type::dx) tmp_op->mixing |= HS_mixing::anomalous;

  string opt("10");
  opt[0]  = char('0')+tau;
  opt[1]  = char('0')+sigma;
  for(auto& x : elem){
    tmp_op->add_matrix_element(get<0>(x), get<1>(x), get<2>(x), opt);
  }
  tmp_op->close();

}



//===============================================================================
/**
 extracts a particular lattice hybridization matrix
*/
matrix<Complex> lattice_model::lattice_hybridization(int iw, int ik){
  
  if(hybrid == nullptr) qcm_throw("Lattice hybridization has not been defined");
  lattice_hybrid& H = *hybrid;
  matrix<Complex> gamma(H.d);
  int r = H.d*(ik+H.nk*iw);
  for(int i=0; i<H.d; i++){
    for(int j=0; j<H.d; j++){
      gamma(j,i) = complex<double>(H.R[j+H.d*(i+r)], H.I[j+H.d*(i+r)]); // note the reverse order (row order)
    }
  }

  // upgrade depending on mixing state
  if(mixing == 0) return gamma;
  else if(HS_mixing::anomalous){
    matrix<Complex> h(2*H.d);
    gamma.move_sub_matrix(H.d, H.d, 0, 0, 0, 0, h);
    gamma.move_sub_matrix(H.d, H.d, 0, 0, H.d, H.d, h, -1.0);
    return h;
  }
  else if(HS_mixing::spin_flip){
    matrix<Complex> h(2*H.d);
    gamma.move_sub_matrix(H.d, H.d, 0, 0, 0, 0, h);
    gamma.move_sub_matrix(H.d, H.d, 0, 0, H.d, H.d, h);
    return h;
  }
  else qcm_throw("This mixing case is not currently covered in the treatment of lattice hybridizations");
}

  //! extracts a rectangular submatrix of size (R,C), starting at (i,j) of this, and copies it to position (I, J) of A
  // move_sub_matrix_HC(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0)
  // template <typename S>
  // void move_sub_matrix(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0)