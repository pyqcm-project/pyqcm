/**
 * @file CPT.cpp
 * @brief Implementation of cluster perturbation theory (CPT) Green-function methods.
 */
#include "lattice_model_instance.hpp"
#include "matrix_continued_fraction.hpp"

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

	if(M.ik >=0 && model->hybrid == nullptr) qcm_throw("Cannot execute this task with a preset lattice hybridization!");
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
			auto it = params.find(op.second->name);
			if(it == params.end()) continue;
			double pv = it->second;
			for(auto& e : op.second->IGF_elem_down){
				M.t(e.r,e.c) += e.v*pv*M.phase[e.n];
			}
		}
	}
	else{
		M.t = H; // adds the momentum-independent part of V
		for(auto& op : model->term){
			auto it = params.find(op.second->name);
			if(it == params.end()) continue;
			double pv = it->second;
			for(auto& e : op.second->IGF_elem){
				M.t(e.r,e.c) += e.v*pv*M.phase[e.n];
			}
		}
	}

	// computing V = tk - t' - Gamma_c + Gamma_k
	M.V = M.t;
	if(M.G.spin_down) Hc_down.add(M.V,-1.0);
	else Hc.add(M.V,-1.0);
	if(model->bath_exists and nohybrid==false) M.G.gamma.add(M.V,-1.0);
	if(model->hybrid != nullptr) M.V.v += model->lattice_hybridization(M.G.iw, M.ik).v;

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
		QCM_ASSERT(model->clusters.size()==1);
		matrix<Complex> Gc = M.G.G.block[0];
		M.g = model->periodize(M.k, Gc);
	}
	else if(periodization == 'N'){ // None
		M.g = M.Gcpt;
	}
	else if(periodization == 'L'){ // Lanczos MCF periodization
		QCM_ASSERT(model->clusters.size() == 1);

		// Retrieve combined MCF in the full cluster site-orbital basis.
		// Works with GF_method='M' or GF_method='L' + combine_mcf=True.
		size_t sys_label = model->nsys * label + model->clusters[0].sys_start;
		auto mcf_data = ED::get_combined_mcf(M.G.spin_down, sys_label);

		// Incorporate the inter-cluster hopping into the first diagonal block.
		// A[0] lives in the cluster orbital basis (dim_GF × dim_GF), as does M.V.
		mcf_data.A[0].v += M.V.v;

		// Apply compact_tiling to A[j>=1] and B[j>=1] before periodizing.
		// A[0] is excluded because it already has M.V incorporated.
		int nA = (int)mcf_data.A.size();
		int nB = (int)mcf_data.B.size();
		for(int j = 1; j < nA; ++j)
			mcf_data.A[j] = model->compact_tiling(mcf_data.A[j], M.k);
		for(int j = 1; j < nB; ++j)
			mcf_data.B[j] = model->compact_tiling(mcf_data.B[j], M.k);

		// Periodize all A[j], B[j] and W from the cluster basis (dim_GF × dim_GF)
		// to the band basis (dim_reduced_GF × dim_reduced_GF).
		vector<matrix<Complex>> A_per(nA), B_per(nB);
		for(int j = 0; j < nA; ++j)
			A_per[j] = model->periodize(M.k, mcf_data.A[j]);
		for(int j = 0; j < nB; ++j)
			B_per[j] = model->periodize(M.k, mcf_data.B[j]);
		matrix<Complex> W_per = model->periodize(M.k, mcf_data.W);

		// Build the periodized MCF and evaluate at the current frequency.
		matrix_continued_fraction<Complex> mcf_per(A_per, B_per, W_per);
		M.g = mcf_per.evaluate(M.G.w);
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
	size_t n_clus = model->clusters.size();

	if(G_host.size()==0){
		G_host.assign(freqs.size(), vector<matrix<Complex>>(n_clus));
		for(int i=0; i<freqs.size(); i++){
			for(int c=0; c<n_clus; c++){
				int d = model->GF_dims[c];
				G_host[i][c].set_size(d,d);
			}
		}
	}
	else return;
	if(model->hybrid != nullptr){
		QCM_ASSERT(model->hybrid->nw == freqs.size());
		CDMFT_host();
		return;
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




//==============================================================================
/**
 Computes the CDMFT host from an external hybridization
 */
void lattice_model_instance::CDMFT_host()
{
	auto &H = *model->hybrid;

	cout << "Building host function with an external hybridization" << endl;
	for(int iw=0; iw<H.nw; iw++){
		Complex w(0.0, H.w[iw]);
		Green_function G = cluster_Green_function(w, false, false);
		G.iw = iw;
		matrix<Complex> Gproj(model->dim_GF);
		for(int ik=0; ik<H.nk; ik++){
			Green_function_k M(G,H.k[ik],ik);
			set_Gcpt(M);
			Gproj += M.Gcpt;
		}
		Gproj.v *= 1.0/H.nk;

		for(int c=0; c<model->clusters.size(); c++){
			int d = model->GF_dims[c];
			int o = model->GF_offset[c];
			Gproj.move_sub_matrix(d, d, o, o, 0, 0, G_host[iw][c]);
			G_host[iw][c].inverse();
			auto X = cluster_self_energy(c, w, false);
			auto Y = cluster_hopping_matrix(c, false);
			G_host[iw][c].v += X.v;
			G_host[iw][c].v += Y.v;
			G_host[iw][c].add(-w);
		}
	}
	if(model->mixing & HS_mixing::up_down){
		qcm_throw("mixing up_down not yet possible with external hybridization");
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
	int n_clus = model->clusters.size();
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
double lattice_model_instance::CDMFT_distance(const vector<double>& p, int clus)
{
	double dist = 0.0;
	for(int i=0; i<model->param_set->CDMFT_variational[clus].size(); i++){
		model->param_set->set_value(model->param_set->CDMFT_variational[clus][i], p[i]);
	}
	auto I = lattice_model_instance(model, label+999);

	double dw = CDMFT_freqs[1]-CDMFT_freqs[0];
	int dim = G_host[0][0].r;

	#pragma omp parallel for reduction (+:dist)
	for(int i=0; i<CDMFT_freqs.size(); i++){
		Complex w(0.0, CDMFT_freqs[i]);
		auto gamma = I.hybridization_function(clus, w, false);
		gamma.v += G_host[i][clus].v;
		dist += norm2(gamma.v)*CDMFT_weights[i];
	}

	if(model->mixing & HS_mixing::up_down){
		#pragma omp parallel for reduction (+:dist)
		for(int i=0; i<CDMFT_freqs.size(); i++){
			Complex w(0.0, CDMFT_freqs[i]);
			auto gamma = I.hybridization_function(clus, w, true);
			gamma.v += G_host_down[i][clus].v;
			dist += norm2(gamma.v)*CDMFT_weights[i];
		}
	}

	if(model->mixing == 0) dist *= 2;
	else if(model->mixing == 3) dist *= 0.5;

	dist *= dw; // normalizes by the frequency step, to give something that does not depend on the step when step --> 0
	dist /= (dim*dim); // divides by the number of elements in the matrices, to obtain an average distance per matrix element

	return dist;
}

//==============================================================================
/**
 Computes the CDMFT residual vector (real-valued) used by least_squares optimizers.
 The residuals are r_n = sqrt(w_n) * (Gamma(iw_n) + G_host(iw_n)), packed as
 [Re(.).ravel(), Im(.).ravel()] for each frequency, then stacked over frequencies.
 If the model has up_down mixing, residuals for both spin components are concatenated.

 NOTE: unlike CDMFT_distance, this does NOT multiply by dw nor divide by (dim*dim),
 because least_squares uses the residuals directly (sum of squares forms the cost).
 A constant rescaling does not change the optimum.

 @param p values of the variational parameters
 @param clus cluster index
 @returns the real residual vector
 */
vector<double> lattice_model_instance::CDMFT_residuals(const vector<double>& p, int clus)
{
	for(int i=0; i<model->param_set->CDMFT_variational[clus].size(); i++){
		model->param_set->set_value(model->param_set->CDMFT_variational[clus][i], p[i]);
	}
	auto I = lattice_model_instance(model, label+999);

	int dim = G_host[0][0].r;
	int dim2 = dim*dim;
	int Nfreq = (int)CDMFT_freqs.size();
	bool has_down = (model->mixing & HS_mixing::up_down) != 0;
	int n_spins = has_down ? 2 : 1;

	vector<double> r(2 * n_spins * Nfreq * dim2, 0.0);

	#pragma omp parallel for if(!omp_in_parallel())
	for(int i=0; i<Nfreq; i++){
		Complex w(0.0, CDMFT_freqs[i]);
		double sw = sqrt(CDMFT_weights[i]);
		auto gamma = I.hybridization_function(clus, w, false);
		for(int k=0; k<dim2; k++){
			Complex z = gamma.v[k] + G_host[i][clus].v[k];
			r[2*dim2*i + k]        = sw * z.real();
			r[2*dim2*i + dim2 + k] = sw * z.imag();
		}
	}

	if(has_down){
		int off = 2 * Nfreq * dim2;
		#pragma omp parallel for if(!omp_in_parallel())
		for(int i=0; i<Nfreq; i++){
			Complex w(0.0, CDMFT_freqs[i]);
			double sw = sqrt(CDMFT_weights[i]);
			auto gamma = I.hybridization_function(clus, w, true);
			for(int k=0; k<dim2; k++){
				Complex z = gamma.v[k] + G_host_down[i][clus].v[k];
				r[off + 2*dim2*i + k]        = sw * z.real();
				r[off + 2*dim2*i + dim2 + k] = sw * z.imag();
			}
		}
	}

	return r;
}

//==============================================================================
/**
 Computes the finite-difference Jacobian d r / d p of the CDMFT residual vector
 using central differences: J[:,j] = (r(p+δeⱼ) − r(p−δeⱼ)) / (2δ).
 Stored row-major with shape (Nrows, Nparams), where Nrows = 2 * n_spins * Nfreq * dim^2.

 Cost: 2*Nparams calls to CDMFT_residuals (no baseline evaluation needed).
 Accuracy: O(δ²) ≈ 1e-10 versus O(δ) ≈ 1e-7 for forward differences.

 Implementation note: the bath parameters -> Gamma(iw) map is evaluated by
 finite difference because individual operator derivatives are not exposed by the
 lattice_model_instance interface. This is still substantially cheaper than
 scipy's default numerical Jacobian (all computation stays in C++).

 @param p values of the variational parameters
 @param clus cluster index
 @returns the Jacobian as a flat row-major vector (shape: Nrows × Nparams, row-major)
 */
vector<double> lattice_model_instance::CDMFT_gradient(const vector<double>& p, int clus)
{
	int Nparams = (int)p.size();
	int dim = G_host[0][0].r;
	int dim2 = dim*dim;
	int Nfreq = (int)CDMFT_freqs.size();
	bool has_down = (model->mixing & HS_mixing::up_down) != 0;
	int n_spins = has_down ? 2 : 1;
	int Nrows = 2 * n_spins * Nfreq * dim2;

	vector<double> J(Nrows * Nparams, 0.0);

	const double delta = global_double("cdmft_jacobian_delta");
	for(int j=0; j<Nparams; j++){
		double scale = std::max(1.0, std::abs(p[j]));
		double dj = delta * scale;

		vector<double> pp = p;
		pp[j] = p[j] + dj;
		vector<double> rp = CDMFT_residuals(pp, clus);

		vector<double> pm = p;
		pm[j] = p[j] - dj;
		vector<double> rm = CDMFT_residuals(pm, clus);

		double inv = 1.0 / (2.0 * dj);
		for(int row=0; row<Nrows; row++){
			J[row * Nparams + j] = (rp[row] - rm[row]) * inv;
		}
	}

	// Restore original parameter values so subsequent calls see consistent state
	for(int i=0; i<model->param_set->CDMFT_variational[clus].size(); i++){
		model->param_set->set_value(model->param_set->CDMFT_variational[clus][i], p[i]);
	}

	return J;
}
