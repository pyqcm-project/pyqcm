 /*! \file
 \brief Integration routines, based on the Cubature library (pcubature_v or hcubature_v).
 They provide resources to integrate over frequency and wavevector, or over wavevector alone.
 */

#include <chrono>
#include "global_parameter.hpp"
#include "integrate.hpp"
#include "vector3D.hpp"
#include "QCM.hpp"
#include "qcm_ED.hpp"
#include "cubature.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//------------------------------------------------------------------------------
// Context structs and cubature callbacks (local to this file)

namespace {

struct WkContext {
	function<void(Complex, vector3D<double>&, const int*, double[])>& f;
	double w_domain;
	double w_offset;  //!< 0 for low-freq region, small_scale for mid-freq region
	double iw_cutoff; //!< used only in the high-freq (inverse) region
	bool high;        //!< true for the high-freq (inverse) region
	bool verb;
	int& count;
};

int wk_cb(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
	auto& ctx = *static_cast<WkContext*>(fdata);
	const int nv = (int)fdim;

	if(!ctx.high) {
		for(size_t i = 0; i < npts; i++) {
			const double *xi = x + i*ndim;
			Complex w(0, xi[0]*ctx.w_domain + ctx.w_offset);
			vector3D<double> k(ndim>1 ? xi[1] : 0.0, ndim>2 ? xi[2] : 0.0, ndim>3 ? xi[3] : 0.0);
			ctx.f(w, k, &nv, &fval[fdim*i]);
		}
	} else {
		for(size_t i = 0; i < npts; i++) {
			const double *xi = x + i*ndim;
			double iw = xi[0]*ctx.w_domain;
			double *fv = &fval[fdim*i];
			if(iw < ctx.iw_cutoff) {
				for(unsigned j = 0; j < fdim; ++j) fv[j] = 0.0;
			} else {
				Complex w(0, 1.0/iw);
				vector3D<double> k(ndim>1 ? xi[1] : 0.0, ndim>2 ? xi[2] : 0.0, ndim>3 ? xi[3] : 0.0);
				ctx.f(w, k, &nv, fv);
				double iw2 = iw*iw;
				for(unsigned j = 0; j < fdim; ++j) fv[j] /= iw2;
			}
		}
	}
	ctx.count += (int)npts;
	if(ctx.count % 1000 == 0 && ctx.verb) cout << ctx.count << " points" << endl;
	return 0;
}


struct KContext {
	function<void(vector3D<double>&, const int*, double[])>& f;
};

int k_cb(unsigned ndim, size_t npts, const double *x, void *fdata, unsigned fdim, double *fval)
{
	auto& ctx = *static_cast<KContext*>(fdata);
	const int nv = (int)fdim;

	for(size_t i = 0; i < npts; i++) {
		const double *xi = x + i*ndim;
		vector3D<double> k(xi[0], ndim>1 ? xi[1] : 0.0, ndim>2 ? xi[2] : 0.0);
		ctx.f(k, &nv, &fval[fdim*i]);
	}
	return 0;
}

} // anonymous namespace




/**
 Performs an integral over frequencies and wavevectors
 Uses the Cubature library
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param dim spatial dimension (1 to 3)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properly initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::wk_integral(int dim, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
	int ndim = dim+1;
	int cuba_maxpoints;

	auto cubature = global_bool("use_pcubature") ? pcubature_v : hcubature_v;
	if(verb) cout << "Cubature integration (" << (global_bool("use_pcubature") ? "p" : "h") << "-adaptive)" << endl;

	int ncomp = (int)Iv.size();

	if(ndim==2)      cuba_maxpoints = 500000;
	else if(ndim==3) cuba_maxpoints = 10000000;
	else if(ndim==4) cuba_maxpoints = 20000000;
	else             cuba_maxpoints = 10000;

	int fail;
	const double xmin[] = {0,0,0,0};
	const double xmax[] = {1,1,1,1};

	const double small_scale = global_double("small_scale");
	const double large_scale = global_double("large_scale");
	const double iw_cutoff = 1.0/global_double("cutoff_scale");
	int count = 0;

	//------------------------------------------------------------------------------
	// first region : frequencies below small_scale
	{
		double w_domain = small_scale;
		vector<double> value(ncomp, 0.0), err(ncomp, 0.0);
		WkContext ctx{f, w_domain, 0.0, iw_cutoff, false, verb, count};
		auto t1 = std::chrono::high_resolution_clock::now();
		fail = cubature(ncomp, wk_cb, &ctx, ndim, xmin, xmax, (size_t)cuba_maxpoints, accuracy*M_PI/w_domain, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
		if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
		if(verb) cout << "region 1 : " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count()/1000 << " seconds" << endl;
		double fac = w_domain*M_1_PI;
		for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	}

	//------------------------------------------------------------------------------
	// second region : frequencies between small_scale and large_scale
	{
		double w_domain = large_scale - small_scale;
		vector<double> value(ncomp, 0.0), err(ncomp, 0.0);
		WkContext ctx{f, w_domain, small_scale, iw_cutoff, false, verb, count};
		auto t1 = std::chrono::high_resolution_clock::now();
		fail = cubature(ncomp, wk_cb, &ctx, ndim, xmin, xmax, (size_t)cuba_maxpoints, accuracy*M_PI/w_domain, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
		if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
		if(verb) cout << "region 2 : " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count()/1000 << " seconds" << endl;
		double fac = w_domain*M_1_PI;
		for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	}

	//------------------------------------------------------------------------------
	// third region : inverse frequencies below 1/large_scale
	{
		double w_domain = 1.0/large_scale;
		vector<double> value(ncomp, 0.0), err(ncomp, 0.0);
		WkContext ctx{f, w_domain, 0.0, iw_cutoff, true, verb, count};
		auto t1 = std::chrono::high_resolution_clock::now();
		fail = cubature(ncomp, wk_cb, &ctx, ndim, xmin, xmax, (size_t)cuba_maxpoints, accuracy*M_PI/w_domain, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
		if(fail) qcm_throw("error in Cubature integral : fail = "+to_string<int>(fail));
		if(verb) cout << "region 3 : " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count()/1000 << " seconds" << endl;
		double fac = w_domain*M_1_PI;
		for(int i=0; i<ncomp; ++i) Iv[i] += value[i]*fac;
	}
}




/**
 Performs an integral over frequencies and wavevectors on a predetermined grid
 Actually computes $\int {dw\over\pi}_0^\infty  \int {d^3k\over (2\pi)^3} f(iw)
 @param w array of frequencies along the positive imaginary axis (double)
 @param weight array of weights associated with the frequencies
 @param dim spatial dimension (1 to 3)
 @param nk_side number of wavevector on each side of the momentum integration domain (Brillouin zone)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properly initialized)
 */
void QCM::wk_integral_grid(const vector<double> &w, const vector<double> &weight, int dim, int nk_side, function<void (Complex w, vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv)
{
	cout << "computing integrals from a grid of " << w.size() << " frequencies and " << nk_side << "**" << dim << " k-points" << endl;
	int nv = (int)Iv.size();
	vector<double> I(nv);
	vector<double> Ik(nv);
	for(int iw=0; iw<w.size(); iw++){
		Complex wc(0, w[iw]);
		to_zero(I);
		double ikside = 1.0/nk_side;
		if(dim==1){
			for(int ikx = 0; ikx < nk_side; ikx++){
				vector3D<double> k(ikx*ikside, 0.0, 0.0);
				f(wc, k, &nv, Ik.data());
				I += Ik;
			}
			I *= ikside;
		}
		else if (dim==2){
			for(int ikx = 0; ikx < nk_side; ikx++){
				for(int iky = 0; iky < nk_side; iky++){
					vector3D<double> k(ikx*ikside, iky*ikside, 0.0);
					f(wc, k, &nv, Ik.data());
					I += Ik;
				}
			}
			I *= ikside*ikside;
		}
		else if (dim==3){
			for(int ikx = 0; ikx < nk_side; ikx++){
				for(int iky = 0; iky < nk_side; iky++){
					for(int ikz = 0; ikz < nk_side; ikz++){
						vector3D<double> k(ikx*ikside, iky*ikside, ikz*ikside);
						f(wc, k, &nv, Ik.data());
						I += Ik;
					}
				}
			}
			I *= ikside*ikside*ikside;
		}
		Iv += I*weight[iw];
	}
}




/**
 Performs an integral over wavevectors on a predetermined regular grid.
 @param dim spatial dimension (1 to 3)
 @param nk_side number of wavevectors on each side of the momentum integration domain (Brillouin zone)
 @param f		function to integrate (may be multi-component)
 @param Iv	value of the integral (adds to previous value: must be properly initialized)
 */
void QCM::k_integral_grid(int dim, int nk_side, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv)
{
	// cout << "computing integral on a grid of " << nk_side << "**" << dim << " k-points" << endl;
	int nv = (int)Iv.size();
	vector<double> I(nv, 0.0);
	vector<double> Ik(nv);
	double ikside = 1.0/nk_side;
	if(dim==1){
		for(int ikx = 0; ikx < nk_side; ikx++){
			vector3D<double> k(ikx*ikside, 0.0, 0.0);
			f(k, &nv, Ik.data());
			I += Ik;
		}
		I *= ikside;
	}
	else if (dim==2){
		for(int ikx = 0; ikx < nk_side; ikx++){
			for(int iky = 0; iky < nk_side; iky++){
				vector3D<double> k(ikx*ikside, iky*ikside, 0.0);
				f(k, &nv, Ik.data());
				I += Ik;
			}
		}
		I *= ikside*ikside;
	}
	else if (dim==3){
		for(int ikx = 0; ikx < nk_side; ikx++){
			for(int iky = 0; iky < nk_side; iky++){
				for(int ikz = 0; ikz < nk_side; ikz++){
					vector3D<double> k(ikx*ikside, iky*ikside, ikz*ikside);
					f(k, &nv, Ik.data());
					I += Ik;
				}
			}
		}
		I *= ikside*ikside*ikside;
	}
	Iv += I;
}




/**
 Performs an integral over wavevectors
 Uses the Cubature library
 @param f		function to integrate (may be multi-component)
 @param Iv		value of the integral (adds to previous value: must be properly initialized)
 @param accuracy		required absolute accuracy of the integral
 */
void QCM::k_integral(int dim, function<void (vector3D<double> &k, const int *nv, double I[])> f, vector<double> &Iv, const double accuracy, bool verb)
{
	int cuba_maxpoints;
	if(dim==1)       cuba_maxpoints = 100000;
	else if(dim==2)  cuba_maxpoints = 300000;
	else             cuba_maxpoints = 4000000;

	int ncomp = (int)Iv.size();
	vector<double> value(ncomp, 0.0), err(ncomp, 0.0);
	int fail;

	auto cubature = global_bool("use_pcubature") ? pcubature_v : hcubature_v;
	const double xmin[] = {0,0,0,0};
	const double xmax[] = {1,1,1,1};

	KContext ctx{f};
	auto t1 = std::chrono::high_resolution_clock::now();
	fail = cubature(ncomp, k_cb, &ctx, dim, xmin, xmax, (size_t)cuba_maxpoints, accuracy, 1e-10, ERROR_INDIVIDUAL, value.data(), err.data());
	if(verb) cout << "Cubature : done in " << (double)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-t1).count()/1000 << " seconds" << endl;
	for(int i=0; i<ncomp; ++i) Iv[i] += value[i];
}
