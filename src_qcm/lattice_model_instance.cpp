/**
 * @file lattice_model_instance.cpp
 * @brief Implementation of lattice_model_instance (parameter-bound model instance).
 */
#include <iomanip>
#include <fstream>

#include "lattice_model_instance.hpp"
#include "integrate.hpp"
#include "parser.hpp"
#include "QCM.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif

extern vector<double> grid_freqs;   // optional imaginary frequency grid for discrete integrals
extern vector<double> grid_weights; // weights associated with grid_freqs

string strip_at(const string& s)
{
  string name = s;
  size_t pos1 = name.rfind('@');
  size_t pos2 = name.rfind('_');
  if(pos1 != string::npos){
    int label =  from_string<int>(name.substr(pos2+1));
    name.erase(pos1);
    name += '_' + to_string(label);
  }
  return name;
}



//==============================================================================
/** 
 default constructor
 @param _model [in] : lattice model
 @param _label [in] : label given to the instance 
 */

lattice_model_instance::lattice_model_instance(shared_ptr<lattice_model> _model, int _label)
: label(_label), model(_model), complex_HS(false)
{
  vector<map<string,double>> sys_values(model->nsys);
  gs_solved = false;
  gf_solved = false;
  average_solved = false;
  SEF_solved = false;
  PE_solved = false;
  E_pot = 0.0;
  E_kin = 0.0;
  if(model->sector_strings.size() == 0) qcm_throw("target sectors were not specified!");
  params = model->param_set->value_map();
  
  model->close_model();
  
  for(auto& x : params){
    string name = x.first;
    auto P = model->name_and_label(name, true);
    if(P.second){
      if(x.second != 0.0 or P.first == "mu"){
        sys_values[P.second-1][P.first] = x.second;
      }
    }
    else model->term.at(name)->is_active = true;
  }

  for(size_t s=0; s<model->nsys; s++){
    if(sys_values[s].size() == 0) qcm_throw("system "+to_string(s)+" has no nonzero operators");
    ED::new_model_instance(model->systems[s].name, sys_values[s], model->sector_strings[s], label*model->nsys + s);
    model->clusters[model->systems[s].clus].mixing = ED::mixing(label*model->nsys + s);
  }
	if(model->GF_offset.size() == 0) model->post_parameter_consolidate(label);
  #ifdef QCM_DEBUG
  cout << "lattice model instance " << label << " created" << endl;
  #endif
}


lattice_model_instance::~lattice_model_instance()
{
  #ifdef QCM_DEBUG
    cout << "lattice model instance #" << label << " deleted." << endl;
  #endif
}
//==============================================================================
/** 
 finds the ground states of all clusters
 @returns a vector of (double, string) giving the ground state energy and sector for each cluster
 */
vector<pair<double,string>> lattice_model_instance::ground_state()
{
  if(gs_solved) return gs;
  static bool first_time = true;

  clus_ave.resize(model->nsys);
  gs.resize(model->nsys);
	GS_energy.resize(model->clusters.size()+1);
	GS_energy[0] = 0.0;


  first_time = false;
  int sc=0;
  // When nsys > 1, solve sub-systems in parallel. Inner OMP regions (Hamiltonian
  // build, Q-matrix irrep loop) will serialize under nested parallelism, but the
  // outer loop exposes more coarse-grained work than the inner loops typically do.
  #pragma omp parallel for schedule(dynamic,1) if(model->nsys > 1)
  for(size_t s = 0; s < model->systems.size(); s++){
    gs[s] = ED::ground_state_solve(model->nsys*label + s);
    clus_ave[s] = ED::cluster_averages(model->nsys*label + s);
    #pragma omp atomic
    GS_energy[model->systems[s].clus+1] +=  gs[s].first;
  }

  for(size_t c = 0; c<model->clusters.size(); c++) GS_energy[c+1] /= model->clusters[c].nsys;

  for(size_t s = 0; s<model->nsys; s++){
    if(ED::complex_HS(model->nsys*label + s)) complex_HS = true;
  }

  gs_solved = true;
  
  return gs;
}



//==============================================================================
/** 
 building the k-independent part of the one-body matrices
 */
void lattice_model_instance::build_H()
{
  H.set_size(model->dim_GF);
  if(model->mixing == HS_mixing::up_down) H_down.set_size(model->dim_GF);
  for(auto& x : model->term){
    lattice_operator& op = *x.second;
    auto it = params.find(op.name);
    if(it == params.end()) continue;
    double pv = it->second;
    for(auto& e : op.GF_elem) H(e.r, e.c) += e.v*pv;
    if(model->mixing == HS_mixing::up_down) for(auto& e : op.GF_elem_down) H_down(e.r, e.c) += e.v*pv;
  }
  
  // building the cluster one-body matrices
  build_cluster_H();
  
  gf_solved = true;
}

//==============================================================================
/** 
 finds the Green function representation of all systems
 */
void lattice_model_instance::Green_function_solve()
{
  static bool first_time = true;
  if(gf_solved) return;
  if(!gs_solved) ground_state();
  #pragma omp parallel for schedule(dynamic,1) if(model->nsys > 1)
  for(size_t s = 0; s<model->nsys; s++) ED::Green_function_solve(model->nsys*label+s);
  build_H();
}



//==============================================================================
/** 
 building the cluster one-body matrices
 */
void lattice_model_instance::build_cluster_H()
{
  size_t n_clus = model->clusters.size();
  Hc.block.assign(n_clus, matrix<Complex>());
  for(size_t c = 0; c<n_clus; c++){
    Hc.block[c].assign(cluster_hopping_matrix(c, false));
  }
  Hc.set_size();
  if(model->mixing == HS_mixing::up_down){
    Hc_down.block.assign(n_clus, matrix<Complex>());
    for(size_t c = 0; c<n_clus; c++){
      Hc_down.block[c].assign(cluster_hopping_matrix(c, true));
    }
    Hc_down.set_size();
  }
}


//==============================================================================
/**
 returns the remixed cluster Green function for cluster # i at frequency w
 if the cluster only contains one system, then this is directly that system's Green function.
 Otherwise they are averaged inverse-wise.
 @param c [in] index of the cluster (starts at 0)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster Green function
 */
matrix<complex<double>> lattice_model_instance::cluster_Green_function_remix(size_t c, complex<double> w, bool spin_down, bool blocks)
{
  if(model->clusters[c].nsys == 1) return ED::Green_function(w, spin_down, model->nsys*label + model->clusters[c].sys_start, blocks);
  else{
    // g is sized from the first system's GF (cluster-mixing dim, which may be
    // smaller than GF_dims[c] when the lattice has anomalous/spin-flip mixing
    // beyond what the cluster itself carries — the upgrade to lattice mixing
    // happens later in cluster_Green_function()).
    matrix<Complex> g;
    for(int s=0; s<model->clusters[c].nsys; s++){
      // ATTENTION : ICI ON DEVRAIT MOYENNER AVEC DES POIDS QUI DÉPENDENT de K
      auto gs = ED::Green_function(w, spin_down, model->nsys*label + s + model->clusters[c].sys_start, blocks);
      gs.inverse();
      if(s == 0) g.set_size(gs.r, gs.c);
      g.v += gs.v;
    }
    g.v *= 1.0/model->clusters[c].nsys;
    g.inverse();
    return g;
  }
}

//==============================================================================
/**
 returns the cluster Green function for cluster # i at frequency w
 @param c [in] index of the cluster (0-based)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster Green function
 */
matrix<complex<double>> lattice_model_instance::cluster_Green_function(size_t c, complex<double> w, bool spin_down, bool blocks)
{
  if(c >= model->clusters.size()) qcm_throw("cluster label out of range");

  int cr = model->clusters[c].ref;
  if(cr != c){
    if(model->clusters[c].conj) w = conjugate(w);
    matrix<Complex> G = cluster_Green_function(cr, w, spin_down, blocks);
    if(model->clusters[c].conj) G.cconjugate();
    return G;
  }

  if(!gf_solved) Green_function_solve();
  matrix<Complex> g = cluster_Green_function_remix(c, w, spin_down, blocks);
  matrix<Complex> G;
  int mix = model->clusters[c].mixing;
  if(model->mixing == mix) return g;

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    G = upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = cluster_Green_function_remix(c, -w, spin_down, blocks);
    if(model->clusters[c].conj) gm.cconjugate();
    G = upgrade_cluster_matrix_anomalous(model->mixing, mix, g, gm);
  }
  else{
    qcm_throw("undefined mixing combinations in cluster_Green_function()");
  }
  return G;
}

//==============================================================================
/**
 returns the remixed cluster Green function for cluster # i at frequency w
 if the cluster only contains one system, then this is directly that system's Green function.
 Otherwise they are averaged inverse-wise.
 @param c [in] index of the cluster (starts at 0)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster Green function
 */
matrix<complex<double>> lattice_model_instance::cluster_self_energy_remix(size_t c, complex<double> w, bool spin_down)
{
  if(model->clusters[c].nsys == 1) return ED::self_energy(w, spin_down, model->nsys*label + model->clusters[c].sys_start);
  else{
    // g is sized from the first system's self-energy (cluster-mixing dim).
    matrix<Complex> g;
    for(int s=0; s<model->clusters[c].nsys; s++){
      // ATTENTION : ICI ON DEVRAIT MOYENNER AVEC DES POIDS QUI DÉPENDENT de K
      auto gs = ED::self_energy(w, spin_down, model->nsys*label + s + model->clusters[c].sys_start);
      if(s == 0) g.set_size(gs.r, gs.c);
      g += gs;
    }
    g.v *= 1.0/model->clusters[c].nsys;
    return g;
  }
}

//==============================================================================
/** 
 returns the cluster Green function for cluster # i at frequency w
 @param c [in] index of the cluster (0 to the number of clusters)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster self-energy
 */
matrix<complex<double>> lattice_model_instance::cluster_self_energy(size_t c, complex<double> w, bool spin_down)
{

  int cr = model->clusters[c].ref;
  if(cr != c){
    if(model->clusters[c].conj) w = conjugate(w);
    matrix<Complex> G = cluster_self_energy(cr, w, spin_down);
    if(model->clusters[c].conj) G.cconjugate();
    return G;
  }

  if(c >= model->clusters.size()) qcm_throw("cluster label out of range");
  if(!gf_solved) Green_function_solve();
  matrix<Complex> g = cluster_self_energy_remix(cr, w, spin_down);
  int mix = model->clusters[c].mixing;
  if(model->mixing == mix) return g;

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    return upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = cluster_self_energy_remix(c, -w, spin_down);
    if(model->clusters[c].conj) gm.cconjugate();
    return upgrade_cluster_matrix_anomalous(model->mixing, mix, g, gm);
  }
  else{
    qcm_throw("undefined mixing combinations in cluster_self_energy()");
    matrix<complex<double>> empty;
    return empty;
  }
}

//==============================================================================
/**
 returns the remixed cluster Green function for cluster # i at frequency w
 if the cluster only contains one system, then this is directly that system's Green function.
 Otherwise they are averaged inverse-wise.
 @param c [in] index of the cluster (starts at 0)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster Green function
 */
matrix<complex<double>> lattice_model_instance::hybridization_function_remix(size_t c, complex<double> w, bool spin_down)
{
  if(model->clusters[c].nsys==1) return ED::hybridization_function(w, spin_down, model->nsys*label + model->clusters[c].sys_start);
  else{
    // g is sized from the first system's hybridization (cluster-mixing dim).
    matrix<Complex> g;
    for(int s=0; s<model->clusters[c].nsys; s++){
      // ATTENTION : ICI ON DEVRAIT MOYENNER AVEC DES POIDS QUI DÉPENDENT de K
      auto gs = ED::hybridization_function(w, spin_down, model->nsys*label + s + model->clusters[c].sys_start);
      if(s == 0) g.set_size(gs.r, gs.c);
      g += gs;
    }
    g.v *= 1.0/model->clusters[c].nsys;
    return g;
  }
}


//==============================================================================
/** 
 returns the hybridization function for cluster # i at frequency w
 @param c [in] index of the cluster (1 to the number of clusters)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster hybridization function
 */
matrix<complex<double>> lattice_model_instance::hybridization_function(size_t c, complex<double> w, bool spin_down)
{
  int cr = model->clusters[c].ref;
  if(cr != c){
    if(model->clusters[c].conj) w = conjugate(w);
    matrix<Complex> G = hybridization_function(cr, w, spin_down);
    if(model->clusters[c].conj) G.cconjugate();
    return G;
  }

  if(c >= model->clusters.size()) qcm_throw("cluster label out of range");
  matrix<Complex> g = hybridization_function_remix(c, w, spin_down);
  int mix = model->clusters[c].mixing;
  if(model->mixing == mix) return g;

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    return upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = hybridization_function_remix(c, -w, spin_down);
    if(model->clusters[c].conj) gm.cconjugate();
    return upgrade_cluster_matrix_anomalous(model->mixing, mix, g, gm);
  }
  else{
    qcm_throw("undefined mixing combinations in hybridization_function()");
    matrix<complex<double>> empty;
    return empty;
  }
}


//==============================================================================
/** 
 returns the cluster hopping matrix for cluster # i
 @param c [in] index of the cluster (starts at 0)
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster non interacting matrix (hopping + pairing)
 */
matrix<complex<double>> lattice_model_instance::cluster_hopping_matrix(size_t c, bool spin_down)
{
  int cr = model->clusters[c].ref;
  if(cr != c){
    matrix<Complex> G = cluster_hopping_matrix(cr, spin_down);
    if(model->clusters[c].conj) G.cconjugate();
    return G;
  }

  if(c >= model->clusters.size()) qcm_throw("cluster label out of range");
  matrix<Complex> g = ED::hopping_matrix(spin_down, model->nsys*label+model->clusters[c].sys_start);

  int mix = model->clusters[c].mixing;
  if(model->mixing == mix) return g;

  // combinations 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    return upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinations 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    return upgrade_cluster_matrix_anomalous(model->mixing, mix, g, g);
  }
  else{
    qcm_throw("undefined mixing combinations in cluster_hopping_matrix()");
    matrix<complex<double>> empty;
    return empty;
  }
}


//==============================================================================
/** 
 computes the self-energy for the Green function object G
 @param G [in] Green function object
 */
void lattice_model_instance::cluster_self_energy(Green_function& G)
{
  size_t n_clus = model->clusters.size();

  if(!gf_solved) Green_function_solve();
	G.sigma.block.assign(n_clus, matrix<Complex>());
  for(size_t c = 0; c<n_clus; c++){
    auto S = cluster_self_energy(c, G.w, G.spin_down);
    G.sigma.block[c].assign(S);
  }
  G.sigma.set_size();
}


//==============================================================================
/** 
 computes cluster Green function and puts it in the structure G
 @param w [in] complex frequency
 @param sig [in] if true, computes also the self-energy
 @param spin_down [in] if true, computes the spin-down part
 @returns a Green_function object
 */
Green_function lattice_model_instance::cluster_Green_function(Complex w, bool sig, bool spin_down)
{
  size_t n_clus = model->clusters.size();
  if(!gf_solved) Green_function_solve();
	Green_function G;
	G.w = w;
	G.spin_down = spin_down;
	G.G.block.resize(n_clus);

  for(size_t c = 0; c<n_clus; c++){
    G.G.block[c].assign(cluster_Green_function(c, w, spin_down, false));
  }
  G.G.set_size();
  if(sig){
    G.sigma.block.resize(n_clus);
    for(size_t c = 0; c<n_clus; c++){
      G.sigma.block[c].assign(cluster_self_energy(c, w, spin_down));
    }
    G.sigma.set_size();
  }
  
  if(model->bath_exists){
    G.gamma.block.resize(n_clus);
    for(size_t c = 0; c<n_clus; c++){
      G.gamma.block[c].assign(hybridization_function(c, w, spin_down));
    }
    G.gamma.set_size();
  }
	return G;
}


//==============================================================================
/** 
 Calculates the self-energy functional (SEF) = Potthoff functional
 @returns the Potthoff functional
 */
double lattice_model_instance::Potthoff_functional()
{
  if(SEF_solved) return omega;
  if(!gf_solved) Green_function_solve();
	
	double omega_clus=0.0;
	for(size_t c=0; c<model->clusters.size(); c++){
    int C = model->clusters[c].ref;
    double omega_c = 0.0;
    for(int s=0; s<model->clusters[c].nsys; s++)
      omega_c += ED::Potthoff_functional(model->nsys*label + s + model->clusters[c].sys_start);
    omega_clus += omega_c/model->clusters[c].nsys;
  }
	
	vector<double> Iv(1);
	Iv[0] = 0.0;

  if(grid_freqs.size() > 0){
    // ---- Discrete (omega, k) grid path ----
    // Mirrors the averages() pattern: build the cluster Green function once per
    // frequency, then parallelize the k-sum via k_integral_grid. The cluster
    // GF is captured by value into the lambda and read concurrently inside the
    // OpenMP parallel region (no shared mutable state).
    int nkx, nky, nkz;
    QCM::get_wavevector_grid(nkx, nky, nkz);
    if(global_bool("verb_integrals"))
      cout << "computing SEF integral from a grid of " << grid_freqs.size() << " frequencies and "
           << nkx << "x" << nky << "x" << nkz << " k-points (dim=" << model->spatial_dimension << ")" << endl;
    bool has_dn = (model->mixing == HS_mixing::up_down);
    vector<double> Iw(1);
    for(int iw = 0; iw < (int)grid_freqs.size(); iw++){
      Complex wc(0, grid_freqs[iw]);
      Green_function G_up = cluster_Green_function(wc, false, false);
      Green_function G_dn;
      if(has_dn) G_dn = cluster_Green_function(wc, false, true);

      auto F_k = [this, G_up, G_dn, has_dn] (vector3D<double> &k, const int *nv, double *I) mutable {
        SEF_integrand_k(G_up, has_dn ? &G_dn : nullptr, k, nv, I);
      };
      Iw[0] = 0.0;
      QCM::k_integral_grid(model->spatial_dimension, nkx, nky, nkz, F_k, Iw);
      Iv[0] += Iw[0] * grid_weights[iw];
    }
  }
  else{
    // ---- Adaptive cubature in (omega, k) ----
    auto F = [this] (Complex w, vector3D<double> &k, const int *nv, double *I) {SEF_integrand(w, k, nv, I);};
    QCM::wk_integral(model->spatial_dimension, F, Iv, global_double("accur_SEF"), global_bool("verb_integrals"));
  }
	double omega_trace = -Iv[0];
	
	
  //-------------------------------------------------------------------------------
	// contribution of the diagonal terms
	double omega_diag=0.0;
	for(size_t i=0; i<model->dim_GF; i++){
		if(model->is_nambu_index[i]) omega_diag += realpart(H(i,i)-Hc(i,i));
		else omega_diag -= realpart(H(i,i)-Hc(i,i));
	}
	
	if(model->mixing == HS_mixing::up_down){
		for(size_t i=0; i<model->dim_GF; i++) omega_diag -= realpart(H_down(i,i)-Hc_down(i,i));
	}
	if(model->mixing == HS_mixing::normal) omega_diag *= 2.0;
	
	omega_diag *= 0.5;
	
  //-------------------------------------------------------------------------------
  // Putting it all together
  if(model->n_mixed == 4){ // If Nambu doubling, then your really have to divide by 2.
    omega_diag *= 0.5;
    omega_trace *= 0.5;
  }
  omega = omega_trace - omega_diag + omega_clus;

  //-------------------------------------------------------------------------------
	// We want the energy density (or per atom), so we divide by the number of cluster orbitals
	omega /= model->sites.size();
  SEF_solved = true;

	return(omega);
}




//==============================================================================
/** 
 Integrand of the SEF integral
 @param w [in] complex frequency
 @param k [in] wavevector
 @param nv number of values in the integrand
 @param I values of these components
 */
void lattice_model_instance::SEF_integrand(Complex w, vector3D<double> &k, const int *nv, double *I){

  check_signals();
	Green_function G = cluster_Green_function(w, false, false);
	if(model->mixing == HS_mixing::up_down){
    Green_function G_down = cluster_Green_function(w, false, true);
    SEF_integrand_k(G, &G_down, k, nv, I);
  }
  else SEF_integrand_k(G, nullptr, k, nv, I);
}


//==============================================================================
/**
 Per-k SEF integrand, given a precomputed cluster Green function. Used by the
 discrete (omega, k) grid path to hoist the frequency-dependent (and
 k-independent) cluster Green function build out of the parallel k-loop in
 k_integral_grid: with G_up (and optionally G_down) supplied as read-only
 inputs, the integrand contains no shared mutable state and is safe to call
 concurrently.
 @param G_up cluster Green function at the current frequency
 @param G_down cluster Green function for the spin-down sector when mixing is
        up_down, otherwise nullptr
 @param k wavevector
 @param nv number of components (1)
 @param I output: log|det(V*G - 1)| at this (omega, k)
 */
void lattice_model_instance::SEF_integrand_k(Green_function &G_up, Green_function *G_down, vector3D<double> &k, const int *nv, double *I){

	matrix<Complex> VG(model->dim_GF);

	Green_function_k K(G_up, k);
  set_V(K);
	G_up.G.mult_left(K.V, VG);
	VG.add(Complex(-1.0,0));
	I[0] = log(abs(VG.determinant()));

	if(G_down){
		VG.zero();
		Green_function_k K_down(*G_down, k);
		set_V(K_down);
		G_down->G.mult_left(K_down.V, VG);
		VG.add(Complex(-1.0,0));
		I[0] += log(abs(VG.determinant()));
	}
	else if(model->mixing == HS_mixing::normal) I[0] *= 2.0;
}


//==============================================================================
/**
 Prints model_instance parameters to a stream
  @param out [in] output stream
  @param format [in] print format, as described in the enumeration print_format
 */
void lattice_model_instance::print_parameters(ostream& out, print_format format)
{
  size_t n_clus = model->clusters.size();

  bool print_all = global_bool("print_all");
  bool print_variances = global_bool("print_variances");

  out << setprecision((int)global_int("print_precision"));
  if(model->param_set == nullptr) return;
  switch(format){
    case print_format::names :
      for(auto& x : params) if(model->param_set->is_dependent(x.first) ==  false or print_all==true) out << x.first << '\t';
      for(auto& x : ave) out << "ave_" << x.first << '\t';
      for(size_t i = 0; i<n_clus; i++){
        if(model->clusters[i].ref != i) continue;
        for(auto& x : clus_ave[i]) out  << "ave_" << strip_at(get<0>(x)) << '\t';
      }
      for(size_t i = 0; i<n_clus; i++){
        if(model->clusters[i].ref != i) continue;
        if(print_variances) for(auto& x : clus_ave[i]) out  << "var_" << strip_at(get<0>(x)) << '\t';
      }
      break;
    case print_format::values :
      for(auto& x : params) if(model->param_set->is_dependent(x.first) ==  false or print_all==true) out << chop(x.second, 1e-10) << '\t';
      for(auto& x : ave) out << chop(x.second, 1e-10) << '\t';
      for(size_t i = 0; i<n_clus; i++){
        if(model->clusters[i].ref != i) continue;
        for(auto& x : clus_ave[i]) out << chop(get<1>(x), 1e-10) << '\t';
      }
      if(print_variances) for(size_t i = 0; i<n_clus; i++){
        if(model->clusters[i].ref != i) continue;
        for(auto& x : clus_ave[i]) out << chop(get<2>(x), 1e-10) << '\t';
      }
      break;
    default:
      break;
  }
}



//==============================================================================
/** 
 Produces the Lehmann form of an array of Green functions at fixed frequency 
 @param k [in] array of wavevectors
 @param orb [in] lattice orbital label
 @param spin_down [in] true if the spin-down part is to be produced (mixing = 4)
 @returns for each wavevector, a pair of arrays giving the location of the pole and the residue
 */
vector<pair<vector<double>, vector<double>>> lattice_model_instance::Lehmann_Green_function(vector<vector3D<double>> &k, int orb, bool spin_down)
{
  if(model->nsys != model->clusters.size()) qcm_throw("For the time being, Lehmann_Green_function() only works without subsystems");

  if(orb >= model->n_mixed*model->n_band) qcm_throw("the lattice orbital label is out of range in Lehmann_Green_function");
  vector<pair<vector<double>, vector<double>>> res(k.size());
  auto G = cluster_Green_function(Complex(0., 1.0), false, spin_down);
  vector<double> Lambda;
  block_matrix<Complex> QB;
  size_t nclus = model->clusters.size();
  for(size_t c=0; c<nclus; c++){
    auto q = ED::qmatrix(spin_down, nclus*label+c);
    Lambda.insert(Lambda.end(), q.first.begin(), q.first.end());
    matrix<Complex> Q(model->GF_dims[c], q.first.size());
    Q.v = q.second;
    QB.block.push_back(Q);
  }
  QB.set_size();
  size_t m = Lambda.size();
  const double threshold = 1e-6;
	
  for(int i=0; i<k.size(); i++){
    res[i].first.resize(m);
    res[i].second.resize(m);
    Green_function_k K(G, k[i]);
	  matrix<Complex> Qk(QB.r,QB.c);
	  matrix<Complex> Lk(m);
	  matrix<Complex> Uk(m);
    set_V(K, true);
	  QB.simil(Lk,K.V);
    Lk.add_to_diagonal(Lambda);
	  Lk.eigensystem(res[i].first,Uk);
	  QB.mult_left(Uk, Qk);
	
    vector<Complex> qk(QB.r);
    vector<Complex> psi(model->n_band);
    for(size_t a=0; a<m; ++a){
      Qk.extract_column(a,qk);
      auto psi = model->periodize(K.k, qk);
      res[i].second[a] = abs(psi[orb]*psi[orb])*model->Lc;
    }
    // cleaning up
    vector<double>& e = res[i].first;
    for(size_t a=1; a<m; ++a){
		  if(abs(e[a]-e[a-1]) < 1e-6){
			  e[a] = (e[a]+e[a-1])/2;
			  res[i].second[a] += res[i].second[a-1];
			  res[i].second[a-1] = 0.0;
		  }
    }
    vector<double> ep, rp;
    ep.reserve(m);
    rp.reserve(m);
    for(size_t a=1; a<m; ++a){
      if(res[i].second[a] > threshold){
        ep.push_back(res[i].first[a]);
        rp.push_back(res[i].second[a]);
      }
    }
    res[i].first = ep;
    res[i].second = rp;
	}
  return res;
}



//==============================================================================
/** 
 upgrades a cluster Green function g from its native mixing clus_mix to the 
 lattice model mixing latt_mix. No additional anomalous component required. 
 @param latt_mix [in] mixing state of the lattice model
 @param clus_mix [in] mixing state of the cluster
 @param g [in] the cluster Green function 
 @returns the upgraded Green function
 */
matrix<Complex> lattice_model_instance::upgrade_cluster_matrix(int latt_mix, int clus_mix, matrix<Complex> &g)
{
  int d = g.r;
  if(clus_mix == HS_mixing::normal and latt_mix == HS_mixing::spin_flip){
    matrix<Complex> G(d*2);
    g.move_sub_matrix(d, d, 0, 0, 0, 0, G);
    g.move_sub_matrix(d, d, 0, 0, d, d, G, 1.0);
    return G;
  }
  else if(clus_mix == HS_mixing::normal and latt_mix == HS_mixing::up_down){
    return g;
  }
  else if(clus_mix == HS_mixing::anomalous and latt_mix == HS_mixing::full){
    matrix<Complex> G(d*2);
    int h = d/2;
    g.move_sub_matrix(h, h, 0, 0, 0, 0, G);
    g.move_sub_matrix(h, h, 0, 0, h, h, G);
    g.move_sub_matrix(h, h, h, h, d, d, G);
    g.move_sub_matrix(h, h, h, h, d+h, d+h, G);
    g.move_sub_matrix(h, h, h, 0, d+h, 0, G);
    g.move_sub_matrix(h, h, h, 0, d, h, G, -1.0);
    g.move_sub_matrix(h, h, h, 0, h, d, G, -1.0);
    g.move_sub_matrix(h, h, h, 0, 0, d+h, G);
    return G;
  }
  else{
    qcm_throw("impossible combination of cluster and lattice mixings");
    matrix<Complex> empty;
    return empty;
  }
}



//==============================================================================
/** 
 upgrades a cluster Green function g and its Nambu version gm from its native mixing clus_mix to the 
 lattice model's mixing latt_mix.
 @param latt_mix [in] mixing state of the lattice model
 @param clus_mix [in] mixing state of the cluster
 @param g [in] the cluster Green function 
 @param gm [in] Nambu version of the cluster Green function 
 @returns the upgraded Green function
 */
matrix<Complex> lattice_model_instance::upgrade_cluster_matrix_anomalous(int latt_mix, int clus_mix, matrix<Complex> &g, matrix<Complex> &gm)
{
  int d = g.r;
  if(clus_mix == HS_mixing::normal and latt_mix == HS_mixing::anomalous){
    matrix<Complex> G(d*2);
    g.move_sub_matrix(d, d, 0, 0, 0, 0, G);
    gm.move_sub_matrix_transpose(d, d, 0, 0, d, d, G, -1.0);
    return G;
  }
  else if(clus_mix == HS_mixing::normal and latt_mix == HS_mixing::full){
    matrix<Complex> G(d*4);
    g.move_sub_matrix(d, d, 0, 0, 0, 0, G);
    g.move_sub_matrix(d, d, 0, 0, d, d, G, 1.0);
    gm.move_sub_matrix_transpose(d, d, 0, 0, 2*d, 2*d, G, -1.0);
    gm.move_sub_matrix_transpose(d, d, 0, 0, 3*d, 3*d, G, -1.0);
    return G;
  }
  else if(clus_mix == HS_mixing::spin_flip and latt_mix == HS_mixing::full){
    matrix<Complex> G(d*2);
    g.move_sub_matrix(d, d, 0, 0, 0, 0, G);
    gm.move_sub_matrix_transpose(d, d, 0, 0, d, d, G, -1.0);
    return G;
  }
  else{
    qcm_throw("impossible combination of cluster and lattice mixings");
    matrix<complex<double>> empty;
    return empty;
  }
}
