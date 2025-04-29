#include <iomanip>
#include <fstream>

#include "lattice_model_instance.hpp"
#include "integrate.hpp"
#include "parser.hpp"
#include "QCM.hpp"
#ifdef _OPENMP
  #include <omp.h>
#endif

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
  n_clus = model->clusters.size();
  vector<map<string,double>> cluster_values(n_clus);
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
        cluster_values[P.second-1][P.first] = x.second;
      }
    }
    else model->term.at(name)->is_active = true;
  }

  for(size_t i=0; i<n_clus; i++){
    if(cluster_values[i].size() == 0) qcm_throw("cluster "+to_string(i)+" has no nonzero operators");
    ED::new_model_instance(model->clusters[i].name, cluster_values[i], model->sector_strings[i], label*n_clus+i);
    model->clusters[i].mixing = ED::mixing(label*n_clus+i);
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

  clus_ave.resize(n_clus);
  gs.resize(n_clus);
	GS_energy.resize(n_clus+1);
	GS_energy[0] = 0.0;


  if(first_time){
    first_time = false;
    for(size_t i = 0; i<model->inequiv.size(); i++){
      auto I = model->inequiv[i];
      gs[I] = ED::ground_state_solve(n_clus*label+I);
      clus_ave[I] = ED::cluster_averages(n_clus*label+I);
    }
  }
  else{
    #pragma omp parallel for 
    for(size_t i = 0; i<model->inequiv.size(); i++){
      auto I = model->inequiv[i];
      gs[I] = ED::ground_state_solve(n_clus*label+I);
      clus_ave[I] = ED::cluster_averages(n_clus*label+I);
    }
  }
  for(size_t i = 0; i<n_clus; i++){
    if(model->clusters[i].ref != i){
      gs[i] = gs[model->clusters[i].ref];
      clus_ave[i] = clus_ave[model->clusters[i].ref];
    }
  }
  for(size_t i = 0; i<n_clus; i++){
    GS_energy[i+1] = gs[i].first;
    GS_energy[0] += gs[i].first;
    for(auto& x : clus_ave[i]) get<0>(x) += '_' + to_string(i+1);
  }

  for(size_t i = 0; i<n_clus; i++){
    if(i != model->clusters[i].ref) continue;
    if(ED::complex_HS(n_clus*label+i)) complex_HS = true;
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
    for(auto& e : op.GF_elem) H(e.r, e.c) += e.v*params[op.name];
    if(model->mixing == HS_mixing::up_down) for(auto& e : op.GF_elem_down) H_down(e.r, e.c) += e.v*params[op.name];
  }
  
  // building the cluster one-body matrices
  build_cluster_H();
  
  gf_solved = true;
}

//==============================================================================
/** 
 finds the Green function representation of all clusters
 */
void lattice_model_instance::Green_function_solve()
{
  static bool first_time = true;
  if(gf_solved) return;
  if(!gs_solved) ground_state();
  if(first_time){
    first_time = false;
    for(size_t i = 0; i<model->inequiv.size(); i++){
      auto I = model->inequiv[i];
      ED::Green_function_solve(n_clus*label+I);
    }
  }
  else{
    #pragma omp parallel for
    for(size_t i = 0; i<model->inequiv.size(); i++){
      auto I = model->inequiv[i];
      ED::Green_function_solve(n_clus*label+I);
    }
  } 
  build_H();
}



//==============================================================================
/** 
 building the cluster one-body matrices
 */
void lattice_model_instance::build_cluster_H()
{
  Hc.block.assign(n_clus, matrix<Complex>());
  for(size_t i = 0; i<n_clus; i++){
    Hc.block[i].assign(cluster_hopping_matrix(i, false));
  }
  Hc.set_size();
  if(model->mixing == HS_mixing::up_down){
    Hc_down.block.assign(n_clus, matrix<Complex>());
    for(size_t i = 0; i<n_clus; i++){
      Hc_down.block[i].assign(cluster_hopping_matrix(i, true));
    }
    Hc_down.set_size();
  }
}


//==============================================================================
/** 
 returns the cluster Green function for cluster # i at frequency w
 @param i [in] index of the cluster (1 to the number of clusters)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster Green function
 */
matrix<complex<double>> lattice_model_instance::cluster_Green_function(size_t i, complex<double> w, bool spin_down, bool blocks)
{
  if(i >= model->clusters.size()) qcm_throw("cluster label out of range");
  if(!gf_solved) Green_function_solve();
  int I = n_clus*label+model->clusters[i].ref;
  matrix<Complex> g = ED::Green_function(w, spin_down, I, blocks);
  matrix<Complex> G;
  int mix = model->clusters[i].mixing;
  if(model->mixing == mix){
    return g;
  }

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    G = upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = ED::Green_function(-w, false, I, false);
    G = upgrade_cluster_matrix_anomalous(model->mixing, mix, g, gm);
  }
  else{
    qcm_throw("undefined mixing combinations in cluster_Green_function()");
  }
  return G;
}

//==============================================================================
/** 
 returns the cluster Green function for cluster # i at frequency w
 @param i [in] index of the cluster (1 to the number of clusters)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster self-energy
 */
matrix<complex<double>> lattice_model_instance::cluster_self_energy(size_t i, complex<double> w, bool spin_down)
{
  if(i >= model->clusters.size()) qcm_throw("cluster label out of range");
  if(!gf_solved) Green_function_solve();
  int I = n_clus*label+model->clusters[i].ref;
  matrix<Complex> g = ED::self_energy(w, spin_down, I);
  int mix = model->clusters[i].mixing;
  if(model->mixing == mix) return g;

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    return upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = ED::self_energy(-w, false, I);
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
 returns the hybridization function for cluster # i at frequency w
 @param i [in] index of the cluster (1 to the number of clusters)
 @param w [in] complex frequency
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster hybridization function
 */
matrix<complex<double>> lattice_model_instance::hybridization_function(size_t i, complex<double> w, bool spin_down)
{
  if(i >= model->clusters.size()) qcm_throw("cluster label out of range");
  int I = n_clus*label+model->clusters[i].ref;
  matrix<Complex> g = ED::hybridization_function(w, spin_down, I);
  int mix = model->clusters[i].mixing;
  if(model->mixing == mix) return g;

  // combinaisons 0:2, 0:4, 1:3, 1:5
  else if(((model->mixing&1) == 0 and (mix&1) == 0) or ((model->mixing&1)==1 and (mix&1)==1)) {
    return upgrade_cluster_matrix(model->mixing, mix, g);
  }
  // combinaisons 0:1, 0:3, 0:5, 2:3
  else if((model->mixing&1) == 1 and (mix&1) == 0){
    auto gm = ED::hybridization_function(-w, false, I);
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
 @param i [in] index of the cluster (1 to the number of clusters)
 @param spin_down [in] true if we are asking for the spin down part (mixing = 4)
 @returns a complex-valued matrix containing the cluster non interacting matrix (hopping + pairing)
 */
matrix<complex<double>> lattice_model_instance::cluster_hopping_matrix(size_t i, bool spin_down)
{
  if(i >= model->clusters.size()) qcm_throw("cluster label out of range");
  int I = n_clus*label+model->clusters[i].ref;
  matrix<Complex> g = ED::hopping_matrix(spin_down, I);
  int mix = model->clusters[i].mixing;

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
  if(!gf_solved) Green_function_solve();
	G.sigma.block.assign(n_clus, matrix<Complex>());
  for(size_t i = 0; i<n_clus; i++){
    auto I = model->clusters[i].ref;
    auto S =  cluster_self_energy(I, G.w, G.spin_down);
    G.sigma.block[i].assign(S);
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
  if(!gf_solved) Green_function_solve();
	Green_function G;
	G.w = w;
	G.spin_down = spin_down;
	G.G.block.resize(n_clus);

  for(size_t i = 0; i<n_clus; i++){
    auto I = model->clusters[i].ref;
    G.G.block[i].assign(cluster_Green_function(I, w, spin_down, false));
  }
  G.G.set_size();
  if(sig){
    G.sigma.block.resize(n_clus);
    for(size_t i = 0; i<n_clus; i++){
      auto I = model->clusters[i].ref;
      G.sigma.block[i].assign(cluster_self_energy(I, w, spin_down));
    }
    G.sigma.set_size();
  }
  
  if(model->bath_exists){
    G.gamma.block.resize(n_clus);
    for(size_t i = 0; i<n_clus; i++){
      auto I = model->clusters[i].ref;
      G.gamma.block[i].assign(hybridization_function(I, w, spin_down));
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
	for(size_t i=0; i<n_clus; i++){
    int I = model->clusters[i].ref;
    omega_clus += ED::Potthoff_functional(label*n_clus+I);
  }
	
	vector<double> Iv(1);
	Iv[0] = 0.0;
  
  // lambda function
  auto F = [this] (Complex w, vector3D<double> &k, const int *nv, double *I) {SEF_integrand(w, k, nv, I);};
  QCM::wk_integral(model->spatial_dimension, F, Iv, global_double("accur_SEF"), global_bool("verb_integrals"));
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
  // cout << "omega_trace = " << omega_trace << endl;
  // cout << "omega_diag = " << omega_diag << endl;
  // cout << "omega_clus = " << omega_clus << endl;

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
	
	matrix<Complex> VG(model->dim_GF);
	
  check_signals();
	Green_function G = cluster_Green_function(w, false, false);
	Green_function_k K(G,k);
  set_V(K);
	G.G.mult_left(K.V, VG);
	VG.add(Complex(-1.0,0));
	I[0] = log(abs(VG.determinant()));
	
	if(model->mixing == HS_mixing::up_down){
		VG.zero();
    Green_function G_down = cluster_Green_function(w, false, true);
		Green_function_k K_down(G_down,k);
		set_V(K_down);
		G_down.G.mult_left(K_down.V, VG);
		VG.add(Complex(-1.0,0));
		double I_down = log(abs(VG.determinant()));
    I[0] += I_down;
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
