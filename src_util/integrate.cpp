 /*! \file
 \brief Integration routines, based on the CUBA library, or home-made (gauss_kronrod).
 They provide resources to integrate over frequency and wavevector, or over wavevector alone.
 */

#include <chrono>
#include "global_parameter.hpp"
#include "integrate.hpp"
#include "vector3D.hpp"
#include "QCM.hpp"
#include "qcm_ED.hpp"
#include "cuba.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#define CUBA_FLAG 0
#define GLS_INTERVALS 128

//How does Cuhre parallel integration work ?
//Cuhre integration rule require multiple evaluation of the integrand
//depending of the dimension and the rule chosen per sample.
//In one example, there is 127 evaluations per Cuhre sample
//The parallelization is achieved by passing the min(127,MAX_EVAL_PER_CORE*OMP_NUM_THREADS)
//parameters to the integrand function which use OpenMP to evaluate them in parallel.
//However! Convergence is achieved for a high number of samples, such as
//30000, which induce a large OpenMP overhead for thread fork/join.
//Try using another integration lib / algo, such as: 
// https://github.com/marcpaterno/gpuintegration/tree/master
#define MAX_EVAL_PER_CORE 100000


double small_scale;
double large_scale;
bool integrand_verb = false;
int integrand_count = 0;


//------------------------------------------------------------------------------
// declarations local to this file

bool sliced = false;
int cuba_threads=1;

typedef int (*integrand_t_vec)(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int k_cuba_integrand(const int *dim, const double x[], const int *nv, double I[], void *userdata, const int *nvec, const int *core);
int low_freq_cuba_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int mid_freq_cuba_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int high_freq_cuba_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int low_freq_w_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int mid_freq_w_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
int high_freq_w_integrand(const int *ndim, const double x[], const int *ncomp, double f[], void *userdata, const int *nvec, const int *core);
void gauss_kronrod(const int ncomp, integrand_t_vec integrand, double a, double b, double accur, double integral[], bool vec, int& neval);

//------------------------------------------------------------------------------
// variables local to this file

double w_domain; //! domain of the frequency integral
double iw_cutoff;
Complex frequency; //! frequency of a particular k-integral
double the_accuracy;

function<void (Complex w, const int *nv, double I[])> w_integrand;
function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> wk_integrand;
function<void (vector3D<double> &k, const int *nv, double I[])> k_integrand;

double GK_x[8] = { // Gauss-Kronrod rule abscissas
	0.000000000000000,
	0.949107912342759,
	0.741531185599394,
	0.405845151377397,
	0.991455371120813,
	0.864864423359769,
	0.586087235467691,
	0.207784955007898
};

double GK_w[8] = { // Gauss-Kronrod rule weights for the 15-point rule
	0.209482141084728,
	0.063092092629979,
	0.140653259715525,
	0.190350578064785,
	0.022935322010529,
	0.104790010322250,
	0.169004726639267,
	0.204432940075298
};

double GK_gw[4] = { // Gauss-Kronrod rule weights for the 7-point rule
	0.417959183673469,
	0.129484966168870,
	0.279705391489277,
	0.381830050505119
};







/**
 Performs an integral over frequencies and wavevectors
 Uses the CUBA library
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param dim spatial dimension (1 to 3)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properlyl initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::wk_integral(int dim, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
	int ndim = dim+1;
	int cuba_mineval, cuba_maxpoints;
  cuba_threads = omp_get_max_threads();
	
	if(verb) cout << "CUBA integration (" << cuba_threads << " threads)" << endl;

	int ncomp = (int)Iv.size();
	vector<double> value(ncomp,0);
	vector<double> err(ncomp,0);
	vector<double> prob(ncomp,0);
	
	wk_integrand = f; // sets the file-wide pointer to the integrand

	if(ndim==2){
		cuba_mineval=(int)global_int("cuba2D_mineval");
		cuba_maxpoints=100000;
	}
	else if(ndim==3){
		cuba_mineval=(int)global_int("cuba3D_mineval");
		cuba_maxpoints=10000000;
	}
	else{
		cuba_mineval=8192;
		cuba_maxpoints=20000000;
	}
	
	int neval, nregions, fail;
  
  small_scale = global_double("small_scale");
  large_scale = global_double("large_scale");
  double cutoff_scale = global_double("cutoff_scale");
  iw_cutoff = 1.0/cutoff_scale;
  integrand_verb = verb;
	integrand_count = 0;
	//------------------------------------------------------------------------------
	// first region : frequencies below small_scale
	neval=0;
	w_domain = small_scale;
	double accur = accuracy*M_PI/w_domain;
	double fac = w_domain*M_1_PI;
	auto t1 = std::chrono::high_resolution_clock::now();
	if(ndim==1){
		gauss_kronrod(ncomp, low_freq_cuba_integrand, 0, 1, accur, value.data(), false, neval);
	}
	else{
		Cuhre(ndim, ncomp, (integrand_t)low_freq_cuba_integrand, nullptr, MAX_EVAL_PER_CORE*cuba_threads, 1e-10, accur, CUBA_FLAG, cuba_mineval, cuba_maxpoints, 0, "", nullptr, &nregions, &neval, &fail, value.data(), err.data(), prob.data());
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	double seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "region 1 : " << neval << " evaluations in " << seconds << " seconds" << endl;

	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	
	//------------------------------------------------------------------------------
	// second region : frequencies between small_scale and large_scale
	
	neval=0;
	w_domain = large_scale-small_scale;
	accur = accuracy*M_PI/w_domain;
	fac = w_domain*M_1_PI;
  to_zero(value);

	t1 = std::chrono::high_resolution_clock::now();
	if(ndim==1){
		gauss_kronrod(ncomp, mid_freq_cuba_integrand, 0, 1, accur, value.data(), false, neval);
	}
	else{
		Cuhre(ndim, ncomp, (integrand_t)mid_freq_cuba_integrand, nullptr, MAX_EVAL_PER_CORE*cuba_threads, 1e-10, accur, CUBA_FLAG, cuba_mineval, cuba_maxpoints, 0, "", nullptr, &nregions, &neval, &fail, value.data(), err.data(), prob.data());
	}
	t2 = std::chrono::high_resolution_clock::now();
	seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "region 2 : " << neval << " evaluations in " << seconds << " seconds" << endl;

	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	
	//------------------------------------------------------------------------------
	// third region : inverse frequencies below 1/large_scale
  
	neval=0;
  w_domain = 1.0/large_scale;
  accur = accuracy*M_PI/w_domain;
  fac = w_domain*M_1_PI;
  to_zero(value);
  
	t1 = std::chrono::high_resolution_clock::now();
  if(ndim==1){
    gauss_kronrod(ncomp, high_freq_cuba_integrand, 0, 1, accur, value.data(), false, neval);
  }
  else{
    Cuhre(ndim, ncomp, (integrand_t)high_freq_cuba_integrand, nullptr, MAX_EVAL_PER_CORE*cuba_threads, 1e-10, accur, CUBA_FLAG, cuba_mineval, cuba_maxpoints, 0, "", nullptr, &nregions, &neval, &fail, value.data(), err.data(), prob.data());
  }
	t2 = std::chrono::high_resolution_clock::now();
	seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
  if(verb) cout << "region 3 : " << neval << " evaluations in " << seconds << " seconds" << endl;

  for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
}









/**
 Performs an integral over wavevectors
 Uses the CUBA library
 @param f		function to integrate (may be multi-component)
 @param Iv		value of the integral (adds to previous value: must be properlyl initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::k_integral(int dim, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
  int cuba_mineval, cuba_maxpoints;
  cuba_threads = omp_get_max_threads();
  
  int ncomp = (int)Iv.size();
  vector<double> value(ncomp,0);
  vector<double> err(ncomp,0);
  double accur = accuracy;
  int neval, nregions, fail;
  neval=0;
  
  k_integrand = f;
  
	auto t1 = std::chrono::high_resolution_clock::now();
  if(dim==1) gauss_kronrod(ncomp, k_cuba_integrand, 0.0, 1.0, accur, value.data(), true, neval);
  else{
    vector<double> prob(ncomp,0);
    if(dim==2){
      cuba_mineval=(int)global_int("cuba2D_mineval");
      cuba_maxpoints=300000;
    }
    else {
      cuba_mineval=(int)global_int("cuba3D_mineval");
      cuba_maxpoints=4000000;
    }
    Cuhre(dim, ncomp, (integrand_t)k_cuba_integrand, nullptr, MAX_EVAL_PER_CORE*cuba_threads, 1e-10, accur, CUBA_FLAG, cuba_mineval, cuba_maxpoints, 0, "", nullptr, &nregions, &neval, &fail, value.data(), err.data(), prob.data());
	auto t2 = std::chrono::high_resolution_clock::now();
	double seconds = (double) std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count() / 1000;
	if(verb) cout << "Cuhre  : " << neval << " points in " << nregions << "regions in " << seconds << " seconds" << endl;
  }
  for(int i=0; i<ncomp; ++i) Iv[i] += value[i];
}


/**
 Performs an integral over frequencies and wavevectors
 Uses the CUBA library
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param dim spatial dimension (1 to 3)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properlyl initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void ED::w_integral(function<void (Complex w, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
	int ncomp = (int)Iv.size();
	vector<double> value(ncomp,0);
	vector<double> err(ncomp,0);
	vector<double> prob(ncomp,0);
	
	w_integrand = f; // sets the file-wide pointer to the integrand

	int nregions, fail;
  
  small_scale = global_double("small_scale");
  large_scale = global_double("large_scale");
  double cutoff_scale = global_double("cutoff_scale");
  iw_cutoff = 1.0/cutoff_scale;

	//------------------------------------------------------------------------------
	// first region : frequencies below small_scale
	
	w_domain = small_scale;
	double accur = accuracy*M_PI/w_domain;
	double fac = w_domain*M_1_PI;
  int neval=0;
	gauss_kronrod(ncomp, low_freq_w_integrand, 0, 1, accur, value.data(), false, neval);
	if(verb) cout << "region 1 : " << neval << " evaluations" << endl;
	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	
	//------------------------------------------------------------------------------
	// second region : frequencies between small_scale and large_scale
	
	w_domain = large_scale-small_scale;
	accur = accuracy*M_PI/w_domain;
	fac = w_domain*M_1_PI;
  to_zero(value);
  neval=0;

	gauss_kronrod(ncomp, mid_freq_w_integrand, 0, 1, accur, value.data(), false, neval);
	if(verb) cout << "region 2 : " << neval << " evaluations" << endl;
	for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	
	//------------------------------------------------------------------------------
	// third region : inverse frequencies below 1/large_scale
  
  neval=0;
  w_domain = 1.0/large_scale;
  accur = accuracy*M_PI/w_domain;
  fac = w_domain*M_1_PI;
  to_zero(value);
  
  gauss_kronrod(ncomp, high_freq_w_integrand, 0, 1, accur, value.data(), false, neval);
  if(verb) cout << "region 3 : " << neval << " evaluations" << endl;
  for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
}

//******************************************************************************
// THE ROUTINES BELOW THIS LINE ARE FOR INTERNAL USE TO THIS FILE ONLY


/**
 Low-frequency integrand in frequency-wavevector integrals
 The type is the one used by the CUBA library (integrand_t)
 @param dim		dimension of the integral
 @param x		point of evaluation
 @param nv		number of components of the integrand
 @param I		values of the integrand (to be calculated)
 @param userdata		parameters (not used)
 */

int low_freq_cuba_integrand(const int *dim, const double x[], const int *nv, double I[], void* userdata, const int *nvec, const int *core)
{

  //function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> wk_integrand;
  //wk_integrand = (function<void (Complex w, vector3D<double> &k, const int *nv, double I[])>) userdata;
  
  if(*dim==1){
    #pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      Complex w(0,x[i]*w_domain);
      vector3D<double> k(0.0,0.0,0.0);
      wk_integrand(w,k,nv,&I[(*nv)*i]);
    }
  }

  else if(*dim==2){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[2*i]*w_domain);
			vector3D<double> k(x[2*i+1],0.0,0.0);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	
	else if(*dim==3){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[3*i]*w_domain);
			vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	
	else if(*dim==4){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[4*i]*w_domain);
			vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	integrand_count += *nvec;
	if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;

	return 0;
}


/**
 mid-frequency integrand in frequency-wavevector integrals.
 The integral is performed over inverse frequency between 0 and 1/w_domain
 The type is the one used by the CUBA library (integrand_t)
 */
int mid_freq_cuba_integrand(const int *dim, const double x[], const int *nv, double I[],void *userdata, const int *nvec, const int *core)
{
  if(*dim==1){
  #pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      Complex w(0,x[i]*w_domain+small_scale);
      vector3D<double> k(0.0,0.0,0.0);
      wk_integrand(w,k,nv,&I[(*nv)*i]);
    }
  }

  else if(*dim==2){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[2*i]*w_domain+small_scale);
			vector3D<double> k(x[2*i+1],0.0,0.0);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	
	else if(*dim==3){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[3*i]*w_domain+small_scale);
			vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	
	else if(*dim==4){
    #pragma omp parallel for schedule(static)
		for(int i=0; i< *nvec; i++){
			Complex w(0,x[4*i]*w_domain+small_scale);
			vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
			wk_integrand(w,k,nv,&I[(*nv)*i]);
		}
	}
	integrand_count += *nvec;
	if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;
	return 0;
}



/**
 high-frequency integrand in frequency-wavevector integrals.
 The relation between iw and w is  iw = w_domain*(1/w - 1/w_cutoff)
 The integral is performed over inverse frequency between 0 and 1/w_domain
 The type is the one used by the CUBA library (integrand_t)
 */
int high_freq_cuba_integrand(const int *dim, const double x[], const int *nv, double I[], void *userdata, const int *nvec, const int *core){
  
  if(*dim==1){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      double iw = x[i]*w_domain;
      if(iw < iw_cutoff){
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(0.0, 0.0, 0.0);
        wk_integrand(w,k,nv,&I[(*nv)*i]);
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] /= iw*iw;
      }
    }
  }
  
  else if(*dim==2){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      double iw = x[2*i]*w_domain;
      if(iw < iw_cutoff){
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[2*i+1],0.0,0.0);
        wk_integrand(w,k,nv,&I[(*nv)*i]);
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] /= iw*iw;
      }
    }
  }
  
  else if(*dim==3){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      double iw = x[3*i]*w_domain;
      if(iw < iw_cutoff){
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[3*i+1],x[3*i+2],0.0);
        wk_integrand(w,k,nv,&I[(*nv)*i]);
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] /= iw*iw;
      }
    }
  }
  
  else if(*dim==4){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      double iw = x[4*i]*w_domain;
      if(iw < iw_cutoff){
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] = 0.0;
      }
      else{
        Complex w(0,1.0/iw);
        vector3D<double> k(x[4*i+1],x[4*i+2],x[4*i+3]);
        wk_integrand(w,k,nv,&I[(*nv)*i]);
        for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] /= iw*iw;
      }
    }
  }
	integrand_count += *nvec;
	if(integrand_count%1000 == 0 and integrand_verb) cout << integrand_count << " points" << endl;
  return 0;
}




int k_cuba_integrand(const int *dim, const double x[], const int *nv, double I[], void *userdata, const int *nvec, const int *core){
  
  if(*dim==1){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      vector3D<double> k(x[i],0.0,0.0);
      k_integrand(k,nv,&I[(*nv)*i]);
    }
  }
  else if(*dim==2){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      vector3D<double> k(x[2*i],x[2*i+1],0.0);
      k_integrand(k,nv,&I[(*nv)*i]);
    }
  }
  else if(*dim==3){
#pragma omp parallel for schedule(static)
    for(int i=0; i< *nvec; i++){
      vector3D<double> k(x[3*i],x[3*i+1],x[3*i+2]);
      k_integrand(k,nv,&I[(*nv)*i]);
    }
  }
  
  return 0;
}


/**
 Low-frequency integrand in frequency-wavevector integrals
 The type is the one used by the CUBA library (integrand_t)
 @param dim		dimension of the integral
 @param x		point of evaluation
 @param nv		number of components of the integrand
 @param I		values of the integrand (to be calculated)
 @param userdata		parameters (not used)
 */

int low_freq_w_integrand(const int *dim, const double x[], const int *nv, double I[],void *userdata, const int *nvec, const int *core)
{
	#pragma omp parallel for schedule(static)
	for(int i=0; i< *nvec; i++){
		Complex w(0,x[i]*w_domain);
		w_integrand(w,nv,&I[(*nv)*i]);
	}
	return 0;
}


/**
 mid-frequency integrand in frequency-wavevector integrals.
 The integral is performed over inverse frequency between 0 and 1/w_domain
 The type is the one used by the CUBA library (integrand_t)
 */
int mid_freq_w_integrand(const int *dim, const double x[], const int *nv, double I[],void *userdata, const int *nvec, const int *core)
{
  #pragma omp parallel for schedule(static)
	for(int i=0; i< *nvec; i++){
		Complex w(0,x[i]*w_domain+small_scale);
		w_integrand(w,nv,&I[(*nv)*i]);
	}
	return 0;
}



/**
 high-frequency integrand in frequency-wavevector integrals.
 The relation between iw and w is  iw = w_domain*(1/w - 1/w_cutoff)
 The integral is performed over inverse frequency between 0 and 1/w_domain
 The type is the one used by the CUBA library (integrand_t)
 */
int high_freq_w_integrand(const int *dim, const double x[], const int *nv, double I[], void *userdata, const int *nvec, const int *core){
  
#pragma omp parallel for schedule(static)
	for(int i=0; i< *nvec; i++){
		double iw = x[i]*w_domain;
		if(iw < iw_cutoff){
			for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] = 0.0;
		}
		else{
			Complex w(0,1.0/iw);
			w_integrand(w,nv,&I[(*nv)*i]);
			for(size_t j=0; j < *nv; ++j) I[(*nv)*i+j] /= iw*iw;
		}
	}
  
  return 0;
}




//! region of the domain of Gauss-Kronrod integration
struct GK_region{
	
	static bool vec; // true if parallelized
	static int ncomp; // number of components
	double a; // start
	double b; // end
	vector<double> value; // value of the integrals in the regions
	double err; // value of the largest quadratic error among the components
	
	GK_region(double _a, double _b, integrand_t_vec integrand) : a(_a), b(_b), err(0.0)
	{
		value.resize(ncomp);
		
		double alpha = 0.5*(b-a);
		double beta = 0.5*(b+a);
		
		int ndim = 1;
		
		// filling the array of abcissas
		vector<double> x(15);
		x[0] = alpha*GK_x[0] + beta;
		for(int j=1; j<4; ++j){
			x[j] =  alpha*GK_x[j] + beta;
			x[j+3] = -alpha*GK_x[j] + beta;
		}
		for(int j=4; j<8; ++j){
			x[j+3] =  alpha*GK_x[j] + beta;
			x[j+7] = -alpha*GK_x[j] + beta;
		}
		
		
		// allocating space for the values
		vector<double> v(15*ncomp);
		v.resize(15*ncomp);
		
		int nvec;
		int cores = 0;
		if(vec){
			nvec=15;
			integrand(&ndim,&x[0],&ncomp,&v[0],nullptr,&nvec,&cores);
		}
		else{
			nvec=1;
			for(int j=0; j<15; j++){
				integrand(&ndim,&x[j],&ncomp,&v[j*ncomp],nullptr,&nvec,&cores);
			}
		}
		
		vector<double> integral_G(ncomp);
		vector<double>& integral_K = value;
		integral_G.resize(ncomp);
		
		for(int i=0; i<ncomp; ++i) integral_G[i] += GK_gw[0]*v[i];
		for(int i=0; i<ncomp; ++i) integral_K[i] +=  GK_w[0]*v[i];
		int k;
		for(int j=1; j<4; ++j){
			k = j;
			for(int i=0; i<ncomp; ++i) integral_G[i] += GK_gw[j]*v[i+k*ncomp];
			for(int i=0; i<ncomp; ++i) integral_K[i] +=  GK_w[j]*v[i+k*ncomp];
			k = j+3;
			for(int i=0; i<ncomp; ++i) integral_G[i] += GK_gw[j]*v[i+k*ncomp];
			for(int i=0; i<ncomp; ++i) integral_K[i] +=  GK_w[j]*v[i+k*ncomp];
		}
		for(int j=4; j<8; ++j){
			k = j+3;
			for(int i=0; i<ncomp; ++i) integral_K[i] +=  GK_w[j]*v[i+k*ncomp];
			k = j+7;
			for(int i=0; i<ncomp; ++i) integral_K[i] +=  GK_w[j]*v[i+k*ncomp];
		}
		
		for(int i=0; i<ncomp; ++i){
			integral_G[i] *= (b-a)*0.5;
			integral_K[i] *= (b-a)*0.5;
			double diff2 = abs(integral_G[i]-integral_K[i]);
			if(diff2 > err) err = diff2;
		}
		err = err*sqrt(err)*200.0;
	}
};



bool GK_region::vec; // true if parallelized
int GK_region::ncomp; // number of components



namespace std
{
	template<>
	struct less<GK_region>{
		bool operator()(const GK_region &x, const GK_region &y) const{
			if(x.err > y.err) return true; // we want the regions in the order of decreasing error
			else if(x.err < y.err) return false;
			else if(x.a < y.a) return true;
			else return false;
		}
	};
}







/**
 Performs a one-dimensional integral of a multi-component integrand, using the same integrand format as the CUBA library
 @param ncomp		number of components
 @param integrand		integrand (same type as the CUBA library)
 @param a		lower bound
 @param b		upper bound
 @param accur		required absolute accuracy of the integral
 @param integral		values of the integrals
 @param neval		number of function evaluation (out)
 */
void gauss_kronrod(const int ncomp, integrand_t_vec integrand, double a, double b, double accur, double integral[], bool vec, int& neval){
	
	
	const int min_regions = 8;
	const int max_regions = 1024;
	GK_region::ncomp = ncomp;
	GK_region::vec = vec;
	set<GK_region> regions;
	
	// start by defining a minimum number of integration regions
	double L = (b-a)/min_regions;
	for(size_t i=0; i<min_regions; i++){
		regions.insert(GK_region(a+i*L,a+(i+1)*L, integrand));
	}
	
	// then iterate until the total error fits into the bounds
	
	while(true){
		double total_error = 0.0;
		for(auto& x : regions) total_error += x.err*x.err;
		total_error = sqrt(total_error);
		
		if(total_error < accur or regions.size() > max_regions) break;
		
		// subdivide the regions with the largest error
		set<GK_region>:: iterator it = regions.begin();
		double ap = it->a;
		double bp = it->b;
		regions.erase(it);
		regions.insert(GK_region(ap, 0.5*(ap+bp), integrand)); neval++;
		regions.insert(GK_region(0.5*(ap+bp), bp, integrand)); neval++;
	}
	
	for(auto& x : regions){
		for(int i=0; i<ncomp; i++) integral[i] += x.value[i];
	}
}









