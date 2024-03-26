#include <fstream>
#include "lattice_model_instance.hpp"
#include "integrate.hpp"
#include "parser.hpp"
#include "QCM.hpp"

double tr_sigma_inf(0.0);

vector<shared_ptr<lattice_operator>> ops = {};

//==============================================================================
/**
 Calculates the lattice expectation value of all operators in the model
 @param ops list of names of operators to compute the averages of
 @returns an array of (string, double) giving the average for each lattice operator
 */
vector<pair<string,double>> lattice_model_instance::averages(const vector<string> &_ops)
{
  if(average_solved) return ave;
  if(_ops.size() == 0) ops = model->one_body_ops;
  else{
    ops.clear();
    ops.reserve(_ops.size());
    for(int i=0; i<_ops.size(); i++){
      if(model->term[_ops[i]]->is_interaction == false) ops.push_back(model->term[_ops[i]]);
    }
  }
  if(!gf_solved) Green_function_solve();
  double accur_OP = global_double("accur_OP");
  bool periodized_averages = global_bool("periodized_averages");

	// frequency integral
	vector<double> Iv(ops.size());
  // lambda function
  if(periodized_averages){
    auto F = [this] (Complex w, vector3D<double> &k, const int *nv, double *I) mutable {average_integrand_per(w, k, nv, I);};
    QCM::wk_integral(model->spatial_dimension, F, Iv, accur_OP, global_bool("verb_integrals"));
  }
  else{
    auto F = [this] (Complex w, vector3D<double> &k, const int *nv, double *I) mutable {average_integrand(w, k, nv, I);};
    QCM::wk_integral(model->spatial_dimension, F, Iv, accur_OP, global_bool("verb_integrals"));
  }
	  
  size_t i = 0;
  ave.resize(ops.size());
	for(auto& op : ops){
    if(periodized_averages) Iv[i] *= model->Lc;
    if(model->mixing == HS_mixing::full) {
      Iv[i] += op->nambu_correction_full;
      Iv[i] *= 0.5;
    }
    else if((model->mixing&HS_mixing::anomalous)  == HS_mixing::anomalous) Iv[i] += op->nambu_correction;
    ave[i] = {op->name, Iv[i]*op->norm};
		i++;
	}
  average_solved = true;
  
  // computing the kinetic energy
  E_kin = 0.0;
  i = 0;
  for(auto& op : ops){
    if(op->name != "mu") E_kin += Iv[i]*params.at(op->name);
    i++;
  }
  E_kin /= model->Lc;
  
	return ave;
}




//==============================================================================
/**
 integrands in the calculation of lattice averages
 @param w frequency
 @param k wavevector
 @param nv number of components
 @param I values of the components of the integrand
 */
void lattice_model_instance::average_integrand(Complex w, vector3D<double> &k, const int *nv, double *I)
{
	
  check_signals();

  const double w_offset = 2.0;
  Complex G_pole = 1.0/(w - w_offset);

  Green_function G = cluster_Green_function(w, false, false);
	Green_function_k K(G,k);
	set_Gcpt(K);
	K.Gcpt.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
	
	size_t i = 0;
	for(auto& op : ops){
		Complex z(0.0);
		for(auto &x : op->GF_elem) z += K.Gcpt(x.c,x.r)*x.v;
		for(auto &x : op->IGF_elem) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n];

		I[i++] = real<double>(z);
	}
	i = 0;
	if(model->mixing == HS_mixing::up_down){
    Green_function G = cluster_Green_function(w, false, true);
    Green_function_k K(G,k);
    set_Gcpt(K);
    K.Gcpt.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
		for(auto& op : ops){
			Complex z(0.0);
			for(auto &x : op->GF_elem_down) z += K.Gcpt(x.c,x.r)*x.v;
			for(auto &x : op->IGF_elem_down) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n];
			I[i++] += real<double>(z);
		}
	}
  if(model->mixing == HS_mixing::normal) for(size_t i=0; i<ops.size(); i++) I[i] *= 2;
}



//==============================================================================
/**
 integrands in the calculation of lattice averages
 @param w frequency
 @param k wavevector
 @param nv number of components
 @param I values of the components of the integrand
 */
void lattice_model_instance::average_integrand_per(Complex w, vector3D<double> &k, const int *nv, double *I)
{
  
  const double w_offset = 2.0;
  Complex G_pole = 1.0/(w - w_offset);
  
  Green_function G = cluster_Green_function(w, false, false);
  vector3D<double> k_red = model->superdual.to(model->physdual.from(k));

  Green_function_k K(G,k_red);
  periodized_Green_function(K);
  K.g.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
  
  size_t i = 0;
  for(auto& op : ops){
    matrix<Complex> S(G.G.r);
    // build the operator matrix
    for(auto &x : op->GF_elem) S(x.r, x.c) += x.v;
    for(auto &x : op->IGF_elem) S(x.r, x.c) += x.v*K.phase[x.n];
    matrix<Complex> S_per = model->periodize(k_red, S);
    I[i++] = realpart(K.g.trace_product(S_per));
  }
  i = 0;
  if(model->mixing == HS_mixing::up_down){
    Green_function G = cluster_Green_function(w,false,true);
    Green_function_k K(G,k_red);
    periodized_Green_function(K);
    K.g.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
    for(auto& op : ops){
      matrix<Complex> S(G.G.r);
      // build the operator matrix
      for(auto &x : op->GF_elem_down) S(x.r, x.c) += x.v;
      for(auto &x : op->IGF_elem_down) S(x.r, x.c) += x.v*K.phase[x.n];
      matrix<Complex> S_per = model->periodize(k_red, S);
      I[i++] += realpart(K.g.trace_product(S_per));
    }
  }
  if(model->mixing == HS_mixing::normal) for(size_t i=0; i<ops.size(); i++) I[i] *= 2;
}





//==============================================================================
/**
 calculates the density of states
 In the anomalous case, this function will compute the DoS at the opposite frequency (-w)
 for the spin down part.
 
 @param w frequency
 returns an array of dimension 2*n_band 
 */
vector<double> lattice_model_instance::dos(const complex<double> w)
{
  double accur_OP = global_double("accur_OP");
  if(!gf_solved) Green_function_solve();

  size_t D_dim = model->n_band;
  size_t d = D_dim;
  if(model->mixing&3) d *= 2;
  vector<double> D(2*D_dim);
  
  Green_function G = cluster_Green_function(w, false, false);

  auto F = [this, G] (vector3D<double> &k, const int *nv, double *I) mutable {
    for(size_t i = 0 ; i<*nv; i++) I[i] = 0.0;
    Green_function_k K(G,k);
    set_Gcpt(K);
    for(size_t i=0; i<model->dim_GF; i++) I[model->reduced_Green_index[i]] += K.Gcpt(i,i).imag();
  };

  vector<double> Iv(model->dim_reduced_GF,0.0);
  QCM::k_integral(model->spatial_dimension, F, Iv, accur_OP, global_bool("verb_integrals"));
  for(size_t i=0; i<d; i++) D[i] = -M_1_PI*Iv[i]/model->Lc;

  if(model->mixing == HS_mixing::up_down){
    Green_function G_down = cluster_Green_function(w, false, true);

    auto F_down = [this, G_down] (vector3D<double> &k, const int *nv, double *I) mutable {
      for(size_t i = 0 ; i<*nv; i++) I[i] = 0.0;
      Green_function_k K(G_down,k);
      set_Gcpt(K);
      for(size_t i=0; i<model->dim_GF; i++) I[model->reduced_Green_index[i]] += K.Gcpt(i,i).imag();
    };

    to_zero(Iv);
    QCM::k_integral(model->spatial_dimension, F_down, Iv, accur_OP, global_bool("verb_integrals"));
    for(size_t i=0; i<D_dim; i++) D[i+D_dim] = -M_1_PI*Iv[i]/model->Lc;
  }
  else if(model->mixing == HS_mixing::normal){
    for(size_t i=0; i<D_dim; i++) D[i+D_dim] = D[i];
  }
  return D;
}


//==============================================================================
/**
 calculates the contribution of a frequency to the average of the operator 'name'
 @param name [in] name of the operator
 @param w [in] complex frequency
 */
double lattice_model_instance::spectral_average(const string& name, const complex<double> w)
{
  double accur_OP = global_double("accur_OP");
  if(model->term.find(name) == model->term.end()) qcm_throw("the operator "+name+" does not exist");
  
  lattice_operator op = *model->term.at(name);
  
  if(!gf_solved) Green_function_solve();
  
  Green_function G = cluster_Green_function(w, false, false);


  auto F = [this, op, G] (vector3D<double> &k, const int *nv, double *I) mutable {
    for(size_t i = 0 ; i<*nv; i++) I[i] = 0.0;
    Green_function_k K(G,k);
    set_Gcpt(K);
    Complex z(0.0);
    for(auto &x : op.GF_elem) z += K.Gcpt(x.c,x.r)*x.v;
    for(auto &x : op.IGF_elem) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n] + K.Gcpt(x.r,x.c)*conjugate(x.v*K.phase[x.n]);
    I[0] = imag(z);
  };
  vector<double> Iv(1,0.0);
  QCM::k_integral(model->spatial_dimension, F, Iv, accur_OP, global_bool("verb_integrals"));

  double result = -M_1_PI*Iv[0]/model->Lc;
  
  if(model->mixing == HS_mixing::up_down){
    Green_function G_down = cluster_Green_function(w, false, true);
    G_down.w = w;
    G_down.spin_down = true;
    G_down.G.block.assign(n_clus, matrix<Complex>());
    for(size_t i = 0; i<n_clus; i++){
      G_down.G.block[i] = matrix<Complex>(model->GF_dims[i], ED::Green_function(w, true, n_clus*label+model->clusters[i].ref, false));
    }
    G_down.G.set_size();
    
    auto F_down = [this, G_down, op] (vector3D<double> &k, const int *nv, double *I) mutable {
      for(size_t i = 0 ; i<*nv; i++) I[i] = 0.0;
      Green_function_k K(G_down,k);
      set_Gcpt(K);
      Complex z(0.0);
      for(auto &x : op.GF_elem) z += K.Gcpt(x.c,x.r)*x.v;
      for(auto &x : op.IGF_elem) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n] + K.Gcpt(x.r,x.c)*conjugate(x.v*K.phase[x.n]);
      I[0] = imag(z);
    };
    
    to_zero(Iv);
    QCM::k_integral(model->spatial_dimension, F_down, Iv, accur_OP, global_bool("verb_integrals"));
    result += -M_1_PI*Iv[0]/model->Lc;
  }
  if(model->mixing == HS_mixing::normal) result *= 2;
  
  return result;
}



//==============================================================================
/** calculates the momentum-dependent average of an operator
 @param op [in] lattice operator
 @param k_set [in] array of wavevectors
 @returns an array of average values, one for each wavevector 
 */
vector<double> lattice_model_instance::momentum_profile(const lattice_operator& op, const vector<vector3D<double>> &k_set)
{
  double accur_OP = global_double("accur_OP");
  auto Fmp = [this, k_set, op] (Complex w, vector3D<double> &ki, const int *nv, double *I) mutable {
    const double w_offset = 2.0;
    Complex G_pole = 1.0/(w - w_offset);
    
    Green_function G = cluster_Green_function(w, false, false);
    size_t i = 0;
    for(auto& k : k_set){
      Green_function_k K(G,k);
      set_Gcpt(K);
      K.Gcpt.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
      Complex z(0.0);
      for(auto &x : op.GF_elem) z += K.Gcpt(x.c,x.r)*x.v;
      for(auto &x : op.IGF_elem) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n];
      I[i++] = real<double>(z);
    }
    if(model->mixing == HS_mixing::up_down){
      Green_function G_dw = cluster_Green_function(w,false,true);
      i = 0;
      for(auto& k : k_set){
        Green_function_k K(G_dw,k);
        set_Gcpt(K);
        K.Gcpt.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
        Complex z(0.0);
        for(auto &x : op.GF_elem) z += K.Gcpt(x.c,x.r)*x.v;
        for(auto &x : op.IGF_elem) z += K.Gcpt(x.c,x.r)*x.v*K.phase[x.n];
        I[i++] += real<double>(z);
      }
    }
  };
  vector<double> A(k_set.size(), 0.0);
  QCM::wk_integral(0, Fmp, A, accur_OP, global_bool("verb_integrals"));
  if(model->mixing == HS_mixing::normal) A *= 2;
  else if((model->mixing&HS_mixing::anomalous)  == HS_mixing::anomalous) A += op.nambu_correction;
  else if(model->mixing == HS_mixing::full) {A += op.nambu_correction_full; A *= 0.5;}
  A *= op.norm;
  return A;
}

//==============================================================================
/** calculates the momentum-dependent average of an operator, from the periodized operator
 @param op [in] lattice operator
 @param k_set [in] array of wavevectors
 @returns an array of average values, one for each wavevector 
 */
vector<double> lattice_model_instance::momentum_profile_per(const lattice_operator& op, const vector<vector3D<double>> &k_set)
{
  double accur_OP = global_double("accur_OP");
  auto Fmp = [this, k_set, op] (Complex w, vector3D<double> &ki, const int *nv, double *I) mutable {
    const double w_offset = 2.0;
    Complex G_pole = 1.0/(w - w_offset);
    
    Green_function G = cluster_Green_function(w, false, false);
    size_t i = 0;
    for(auto& k : k_set){
      vector3D<double> k_red = model->superdual.to(model->physdual.from(k));
      Green_function_k K(G,k_red);
      periodized_Green_function(K);      
      K.g.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
      matrix<Complex> S(G.G.r);
      // build the operator matrix
      for(auto &x : op.GF_elem) S(x.r, x.c) += x.v;
      for(auto &x : op.IGF_elem) S(x.r, x.c) += x.v*K.phase[x.n];
      matrix<Complex> S_per = model->periodize(k_red, S);
      I[i++] = realpart(K.g.trace_product(S_per));
    }
    if(model->mixing == HS_mixing::up_down){
      Green_function G_dw = cluster_Green_function(w,false,true);
      i = 0;
      for(auto& k : k_set){
        vector3D<double> k_red = model->superdual.to(model->physdual.from(k));
        Green_function_k K(G_dw,k_red);
        periodized_Green_function(K);      
        K.g.add(-G_pole); // regulates the Green function at high frequency (subtracts G_pole times the identity matrix)
        matrix<Complex> S(G.G.r);
        // build the operator matrix
        for(auto &x : op.GF_elem) S(x.r, x.c) += x.v;
        for(auto &x : op.IGF_elem) S(x.r, x.c) += x.v*K.phase[x.n];
        matrix<Complex> S_per = model->periodize(k_red, S);
        I[i++] = realpart(K.g.trace_product(S_per));
      }
    }
  };
  vector<double> A(k_set.size(), 0.0);
  QCM::wk_integral(0, Fmp, A, accur_OP, global_bool("verb_integrals"));
  if(model->mixing == HS_mixing::normal) A *= 2;
  else if((model->mixing&HS_mixing::anomalous)  == HS_mixing::anomalous) A += op.nambu_correction;
  else if(model->mixing == HS_mixing::full) {A += op.nambu_correction_full; A *= 0.5;}
  A *= op.norm;
  return A;
}


#define WINF 1e5
//==============================================================================
/** Calculates the potential energy using $V = Tr(\Sigma G)$
 The self-energy cannot be accurately computed when the frequency is too high, because
 of round off errors when subtracting the frequency from the inverse Green function.
 Best to not go beyond half the accuracy of doubles, i.e., 1e-7 (or freq ~ 1e7).
 @returns the potential energy
 */
double lattice_model_instance::potential_energy()
{
  if(PE_solved) return E_pot;
  PE_solved = true;

  double accur_OP = global_double("accur_OP");
  if(!gf_solved) Green_function_solve();
 
  // computing the infinite frequency limit
  double prev_cutoff = global_double("cutoff_scale");
  set_global_double("cutoff_scale", WINF); // important for convergence
  
  
  tr_sigma_inf = 0.0;
  for(int i=0; i<model->clusters.size(); i++){
    tr_sigma_inf += ED::tr_sigma_inf(i + label*model->clusters.size());
  }

  // frequency integral
  vector<double> I(1);
  // lambda function
  auto F = [this] (Complex w, vector3D<double> &k, const int *nv, double *I) mutable {potential_energy_integrand(w, k, nv, I);};
  QCM::wk_integral(model->spatial_dimension, F, I, 0.1*accur_OP, global_bool("verb_integrals"));

  set_global_double("cutoff_scale", prev_cutoff); // restores to previous value
  if(model->mixing == HS_mixing::full) I[0] *= 0.5; // full Nambu doubling overestimates by a factor of 2
  E_pot = 0.5*I[0]/model->Lc;
  return E_pot;
}




//==============================================================================
/** computes Tr(Sigma G) at a given frequency and wavevector, for the computation of the potential energy
 @param w frequency
 @param k wavevector
 @param spin_down true if in spin down sector (mixing = 4)
 @returns Tr(Sigma G)
 */
complex<double> lattice_model_instance::TrSigmaG(Complex w, vector3D<double> &k, bool spin_down)
{
  complex<double> z(0.0);
  Green_function G = cluster_Green_function(w, false, spin_down);
  Green_function_k K(G,k);
  set_Gcpt(K);
  cluster_self_energy(G);
  z = G.sigma.build_matrix().trace_product(K.Gcpt);
  return z;
}

//==============================================================================
/** integrand in the computation of the potential energy
 @param w frequency
 @param k wavevector
 @param nv number of components
 @param I values of the components of the integrand
 */
void lattice_model_instance::potential_energy_integrand(Complex w, vector3D<double> &k, const int *nv, double *I)
{
  check_signals();
  
  const double w_offset = 2.0;
  
  Complex e = TrSigmaG(w, k, false);
  if(model->mixing == HS_mixing::up_down) e += TrSigmaG(w, k, true);
  else if(model->mixing == HS_mixing::normal) e *= 2;
  
  e -= tr_sigma_inf/(w-w_offset);
  I[0] = real<double>(e);
}

