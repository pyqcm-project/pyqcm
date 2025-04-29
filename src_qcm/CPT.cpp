#include "lattice_model_instance.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <fstream>

#define ETADELTA 0.25
//==============================================================================
/** 
 * Builds the matrix tk, the momentum-dependent hopping matrix
 then Calculates V, defined as \f$ t(k) - t' - \Gamma(\omega)\f$
 if nohybrid is true, then only \f$ t(k) - t' \f$
 @param M [in, out] Green_function_k object
 @param nohybrid [in] if true, does not add the hybridization function (used in producing the Lehmann form of the Green function)
 */
void lattice_model_instance::set_V(Green_function_k &M, bool nohybrid){
	
	size_t nv = model->neighbor.size();
	M.phase.assign(nv, 1.0);
	// computing the phases
	for(int i=0; i<nv; ++i){
		double z = M.k*model->neighbor[i]*2*M_PI;
		M.phase[i] = Complex(cos(z), sin(z));
	}
	
	// computing tk
	if(M.G.spin_down){
		M.t = H_down;
		for(auto& op : model->term){
			for(auto& e : op.second->IGF_elem_down){
				M.t(e.r,e.c) += e.v*params[op.second->name]*M.phase[e.n];
			}
		}
	}
	else{
		M.t = H; // adds the momentum-independent part of V
		for(auto& op : model->term){
			for(auto& e : op.second->IGF_elem){
				M.t(e.r,e.c) += e.v*params[op.second->name]*M.phase[e.n];
			}
		}
	}

	// computing V = tk - t' - Gamma
	M.V = M.t;
	if(M.G.spin_down) Hc_down.add(M.V,-1.0);
	else Hc.add(M.V,-1.0);
	if(model->bath_exists and nohybrid==false) M.G.gamma.add(M.V,-1.0);
}





//==============================================================================
/** 
 Computes the CPT Green function (in partial real-space representation)
 \f[ G_{\rm cpt}^{-1} = G^{-1} - V\f]
 @param M [in, out]  Green_function_k object
 */
void lattice_model_instance::set_Gcpt(Green_function_k &M)
{
	matrix<Complex> VG(Green_function_k::dim_GF);
	
	set_V(M);
	M.G.G.mult_right(M.V, VG); // VG = V*G
	VG.v *= -1.0;
	VG.add(1.0);
	VG.inverse();
	M.G.G.mult_left(VG, M.Gcpt); // Gcpt = G/(1-V*G)
}




//==============================================================================
/** 
 Computes the inverse CPT Green function (in partial real-space representation)
 Ginv is the inverse cluster Green function
 At the end M.V contains the inverse CPT Green function, instead of the CPT
 Green function as in the usual call.
 @param Ginv [in]  inverse cluster Green function
 @param M [in, out]  Green_function_k object
 */
void lattice_model_instance::inverse_Gcpt(const block_matrix<Complex> &Ginv, Green_function_k &M)
{
  set_V(M);
  Ginv.add(M.V, -1.0);
  M.V.v *= -1.0;
}




//==============================================================================
/** 
 calculates the periodized kinetic energy
 (of dimension nband, or 2*nband in the anomalous case)
 @param M [in, out]  Green_function_k object
 @returns a complex-valued matrix
 */
matrix<Complex> lattice_model_instance::epsilon(Green_function_k &M)
{
	char periodization = global_char("periodization");
	set_V(M);

	if(periodization == 'N') return M.t;
	else return model->periodize(M.k,M.t);
}



//==============================================================================
/** 
 calculates the dispersion relation
 (of dimension nband, or 2*nband in the anomalous case)
 @param M [in, out]  Green_function_k object
 @returns an array of eigenvalues
 */
vector<double> lattice_model_instance::dispersion(Green_function_k &M)
{
  matrix<Complex> eps = epsilon(M);
  vector<double> d(eps.r);
  eps.eigenvalues(d);
  return d;
}



//==============================================================================
/** 
 Produces the periodized Green function, according to possible periodization schemes
 @param M [in, out]  Green_function_k object
 */
matrix<Complex> lattice_model_instance::band_Green_function(Green_function_k &M)
{
	// first compute the lattice Green function
	periodized_Green_function(M);
	// then compute the dispersion relation
	auto eps = epsilon(M);
	int nband = model->n_band;
	matrix<Complex> U(nband);
	vector<double> d(nband);
	eps.eigensystem(d, U);
	matrix<Complex> bandG(M.g);
	bandG.simil(U, M.g);
	return bandG;
}


//==============================================================================
/** 
 Produces the periodized Green function, according to possible periodization schemes
 @param M [in, out]  Green_function_k object
 */
void lattice_model_instance::periodized_Green_function(Green_function_k &M)
{
  char periodization = global_char("periodization");
	set_Gcpt(M);
	if(periodization == 'G'){
		M.g = model->periodize(M.k, M.Gcpt);
	}
	else if(periodization == 'M'){ // cumulant
    	double mu = 0.0;
    	if(params.find("mu") != params.end()) mu = params.at("mu");
		matrix<Complex> gtmp(M.Gcpt);
		matrix<Complex> ep(M.g.r);
		gtmp.inverse();
		for(int i=0; i<M.dim_GF; i++) M.t(i,i) += mu*(1-2*model->is_nambu_index[i]); // mu removed from t
		gtmp.v += M.t.v; // the new t removed from gtmp, which is Gcpt^{-1}
		gtmp.inverse();
		M.g = model->periodize(M.k, gtmp); // gtmp is periodized and put in M.g
		ep = model->periodize(M.k, M.t); // M.t (without mu) is periodized and put in ep
		M.g.inverse();
		M.g.v -= ep.v;
		M.g.inverse();
	}
	else if(periodization == 'S'){ // sigma
		to_zero(M.Gcpt.v);
		cluster_self_energy(M.G);
		M.Gcpt = M.G.sigma.build_matrix();
		M.g = model->periodize(M.k, M.Gcpt);
		matrix<Complex> eps(model->dim_reduced_GF);
		eps = epsilon(M);
		M.g.v += eps.v;
		M.g.v -= M.G.w;
		M.g.v *= -1.0;
		M.g.inverse();
	}
	else if(periodization == 'C'){ // cluster
		assert(n_clus==1);
		matrix<Complex> Gc = M.G.G.block[0];
		M.g = model->periodize(M.k, Gc);
	}
	else if(periodization == 'N'){ // None
		M.g = M.Gcpt;
	}
	else{
		qcm_throw("undefined periodization scheme");
	}
}





//==============================================================================
/** 
 calculates the lattice self-energy matrix
 (of dimension nband, or 2*nband in the anomalous case)
 @param M [in, out]  Green_function_k object
 */
void lattice_model_instance::self_energy(Green_function_k &M)
{
	periodized_Green_function(M);
	matrix<Complex> eps(model->dim_reduced_GF);
	M.sigma = M.g;
	M.sigma.inverse();
	M.sigma.add(-M.G.w);
	eps = epsilon(M);
	M.sigma.v += eps.v;
	M.sigma.v *= -1.0;
}






//==============================================================================
/** 
 Calculates the integral of Gcpt over wavevectors, using the grid kgrid
 @param w [in] complex frequency
 @param spin_down [in] if true, computes the spin-down component (mixing = 4)
 */
matrix<Complex> lattice_model_instance::projected_Green_function(Complex w, bool spin_down)
{

	if(global_bool("verb_ED")) cout << "Computing projected GF at w = " << w << endl;
	matrix<Complex> PGF(model->dim_GF);
	Green_function G = cluster_Green_function(w, false, spin_down);
	size_t kgrid_side = global_int("kgrid_side");
  
	double step = 1.0/kgrid_side;
  
	
	size_t nk = 0;
	switch((int)model->superlattice.D){
		case 0:
			{
				Green_function_k M(G,{0.0,0.0,0.0});
				set_Gcpt(M);
				PGF += M.Gcpt;
        		nk++;
			}
			break;
		case 1:
			// #pragma omp parallel for
    		for(int i=0; i<kgrid_side; i++){
				Green_function_k M(G,{(i+0.5)*step,0.0,0.0});
				set_Gcpt(M);
				// #pragma omp critical
				PGF += M.Gcpt;
        		nk++;
			}
			break;
		case 2:
			for(int i=0; i<kgrid_side; i++){
				// #pragma omp parallel for
				for(int j=0; j<kgrid_side; j++){
					Green_function_k M(G,{(i+0.5)*step, (j+0.5)*step,0.0});
					set_Gcpt(M);
					// #pragma omp critical
					PGF += M.Gcpt;
          			nk++;
				}
			}
			break;
		case 3:
			for(int i=0; i<kgrid_side; i++){
				for(int j=0; j<kgrid_side; j++){
					// #pragma omp parallel for
					for(int k=0; k<kgrid_side; k++){
						Green_function_k M(G,{(i+0.5)*step, (j+0.5)*step, (k+0.5)*step});
						set_Gcpt(M);
						// #pragma omp critical
						PGF += M.Gcpt;
            			nk++;
					}
				}
			}
			break;
		default:
			qcm_throw("Forbidden dimension in construction of wavevector grid!");
	}
	PGF.v *= 1.0/nk;
	return PGF;
}




//==============================================================================
/** 
 Computes the CDMFT host
 @param freqs [in] frequency array (along the imaginary axis)
 @param weights [in] weights of the different frequencies in the distance function
 */
void lattice_model_instance::CDMFT_host(const vector<double>& freqs, const vector<double>& weights)
{
	CDMFT_weights = weights;
	CDMFT_freqs = freqs;

	if(G_host.size()==0){
		G_host.assign(freqs.size(), vector<matrix<Complex>>(n_clus));
		for(int i=0; i<freqs.size(); i++){
			for(int c=0; c<n_clus; c++){
				int d = model->GF_dims[c];
				G_host[i][c].set_size(d,d);
			}
		}
	}

	// #pragma omp parallel for
	if(global_bool("verb_ED")) cout << "Building host function" << endl;
	for(int i=0; i<freqs.size(); i++){
		Complex w(0.0, freqs[i]);
		auto Gproj= projected_Green_function(w, false);
		for(int c=0; c<n_clus; c++){
			int d = model->GF_dims[c];
			int o = model->GF_offset[c];
			Gproj.move_sub_matrix(d, d, o, o, 0, 0, G_host[i][c]);
			G_host[i][c].inverse();
			auto X = cluster_self_energy(c, w, false);
			auto Y = cluster_hopping_matrix(c, false);
			G_host[i][c].v += X.v;
			G_host[i][c].v += Y.v;
			G_host[i][c].add(-w);
		}
	}
	if(model->mixing & HS_mixing::up_down){
		if (global_bool("verb_ED")) cout << "Building host function for spin down" << endl;
		if(G_host_down.size()==0){
			G_host_down.assign(freqs.size(), vector<matrix<Complex>>(n_clus));
			for(int i=0; i<freqs.size(); i++){
				for(int c=0; c<n_clus; c++){
					int d = model->GF_dims[c];
					G_host_down[i][c].set_size(d,d);
				}
			}
		}

		// #pragma omp parallel for
		for(int i=0; i<freqs.size(); i++){
			Complex w(0.0, freqs[i]);
			auto Gproj= projected_Green_function(w, true);
			for(int c=0; c<n_clus; c++){
				int d = model->GF_dims[c];
				int o = model->GF_offset[c];
				Gproj.move_sub_matrix(d, d, o, o, 0, 0, G_host_down[i][c]);
				G_host_down[i][c].inverse();
				auto X = cluster_self_energy(c, w, true);
				auto Y = cluster_hopping_matrix(c, true);
				G_host_down[i][c].v += X.v;
				G_host_down[i][c].v += Y.v;
				G_host_down[i][c].add(-w);
			}
		}
	}
}

vector<vector<matrix<Complex>>> lattice_model_instance::get_CDMFT_host(bool spin_down){
	if(spin_down){
		if(G_host_down.size() == 0) qcm_throw("G_host_down has not been computed for this instance");
		return G_host_down;
	}
	else {
		if(G_host.size() == 0) qcm_throw("G_host has not been computed for this instance");
		return G_host;
	}
}

//==============================================================================
/** 
 Computes the integrated CDMFT host along the real frequency axis
 @param freqs [in] frequency array (along the real axis)
 @param spin_down [in] boolean : true if spin-down sector (mixing = 4) 
 */

Complex lattice_model_instance::CDMFT_host_part(Complex w, bool spin_down)
{
	Complex gamma(0.,0.);
	auto Gproj= projected_Green_function(w, spin_down);
	for(int c=0; c<n_clus; c++){
		int d = model->GF_dims[c];
		int o = model->GF_offset[c];
		matrix<Complex> G_host_tmp(d);
		G_host_tmp.zero();
		Gproj.move_sub_matrix(d, d, o, o, 0, 0, G_host_tmp);
		G_host_tmp.inverse();
		auto X = cluster_self_energy(c, w, false);
		auto Y = cluster_hopping_matrix(c, false);
		G_host_tmp.v += X.v;
		G_host_tmp.v += Y.v;
		G_host_tmp.add(-w);
		gamma += G_host_tmp.trace();
	}
	return gamma;
}



//==============================================================================
/** 
 sets the CDMFT host from input data
 @param freqs [in] frequency array (along the imaginary axis)
 @param clus [in] label of the cluster
 @param H [in] vector of matrices (host)
 @param spin_down [in] boolean : true if spin-down sector (mixing = 4) 
 */
void lattice_model_instance::set_CDMFT_host(const vector<double>& freqs, const int clus, const vector<matrix<Complex>>& H, const bool spin_down)
{
	CDMFT_weights.assign(freqs.size(), 1.0/freqs.size());
	CDMFT_freqs = freqs;

	if(G_host.size()==0){
		G_host.assign(freqs.size(), vector<matrix<Complex>>(n_clus));
		for(int i=0; i<freqs.size(); i++){
			for(int c=0; c<n_clus; c++){
				int d = model->GF_dims[c];
				G_host[i][c].set_size(d,d);
			}
		}
	}
	
	if(spin_down){
		for(int i=0; i<freqs.size(); i++) G_host_down[i][clus] = H[i];
	}
	else{
		for(int i=0; i<freqs.size(); i++) G_host[i][clus] = H[i];
	}
}
//==============================================================================
/** 
 Computes the CDMFT distance function
 @param p values of the variational parameters
 @returns the distance function
 */
double lattice_model_instance::CDMFT_distance(const vector<double>& p)
{
	double dist = 0.0;
	for(int i=0; i<model->param_set->CDMFT_variational.size(); i++){
		model->param_set->set_value(model->param_set->CDMFT_variational[i], p[i]);
	}
	auto I = lattice_model_instance(model, label+999);
	
	// #pragma omp parallel for reduction (+:dist)
	for(int i=0; i<CDMFT_freqs.size(); i++){
		double distw = 0.0;
		Complex w(0.0, CDMFT_freqs[i]);
		for(int c=0; c<n_clus; c++){
			auto gamma = I.hybridization_function(c, w, false);
			gamma.v += G_host[i][c].v;
			distw += norm2(gamma.v);
		}
		dist += distw*CDMFT_weights[i];
	}

	if(model->mixing & HS_mixing::up_down){
		// #pragma omp parallel for reduction (+:dist)
		for(int i=0; i<CDMFT_freqs.size(); i++){
			double distw = 0.0;
			Complex w(0.0, CDMFT_freqs[i]);
			for(int c=0; c<n_clus; c++){
				auto gamma = I.hybridization_function(c, w, true);
				gamma.v += G_host_down[i][c].v;
				distw += norm(gamma.v);
			}
			dist += distw*CDMFT_weights[i];
		}
	}

	if(model->mixing == 0) dist *= 2;
	else if(model->mixing == 3) dist *= 0.5;

	return dist;
}


