#ifndef lanczos_h
#define lanczos_h

// #include "matrix.hpp"
#include "continued_fraction.hpp"
#include "parser.hpp"

void EigensystemTridiagonal(bool evector_flag, vector<double> &alpha, vector<double> &beta, vector<double> &energy, vector<double> &evector);
extern std::normal_distribution<double> normal_dis;

//! Implementation of the Lanczos method
template<typename T, typename HilbertField>
void Lanczos(T &hamil, size_t dim, double &val, vector<HilbertField> &x, bool verb=false)
{
	vector<double> energy;
	vector<double> alpha;
	vector<double> beta;
	size_t niter = 0;

	LanczosEigenvalue(hamil, x, alpha, beta, energy, niter, verb);
	val = energy[0];
	LanczosEigenvector(hamil, x, alpha, beta, verb);
}


//! Implementation of the Lanczos method for the lowest eigenvalue
/**
 Applies the Lanczos algorithm for the lowest eigenvalue.
 Doesn't calculate the correponding eigenvector.
 Assumes the multiply-add matrix routine is usually provided by a struct method
 @param H			Hamiltonian matrix
 @param gs			initial vector
 @param alpha			Diagonal of the tridiagonal matrix (allocated on input, provided on output).
 @param beta			Second Diagonal of the tridiagonal matrix (allocated on input, provided on output).
 @param energy		eigenvalues
 @param niter			Number of iterations (maximum number provided on input, actual number on output).
 
 The structure H is assumed to have the following elements:
 - A method hamil.multadd(vector<HilbertField> &x, vector<HilbertField> &y) that does y -->  y + H.x
 */
template<typename T, typename HilbertField>
void LanczosEigenvalue(T &H, vector<HilbertField> &gs, vector<double> &alpha, vector<double> &beta, vector<double> &energy, size_t &niter, bool verb=false)
{
	size_t dim = gs.size();
	size_t max_iter_lanczos = global_int("max_iter_lanczos");

	if(niter>dim) niter = dim;
	else niter = max_iter_lanczos;
  
	if(verb) cout << "\nLanczos method for the ground state; dim = " << dim << endl;

	energy.clear();
	alpha.clear();
	beta.clear();
	alpha.reserve(niter);
	beta.reserve(niter);
	energy.reserve(niter);
	vector<double> evector;
	
	vector<HilbertField> r(gs); // residue
	vector<HilbertField> q(dim); // Lanczos basis vector
	
	double Ritz=2.0*global_double("accur_lanczos");
	while(Ritz > global_double("accur_lanczos"))
	{
		check_signals();

		// building q
		if(beta.size()){ // does nothing anyway if j=0
			r *= 1.0/beta.back();
			q *= -beta.back();
		}
		
		// computing q = q + H*r
		H.mult_add(r,q);
		
		// swap q and r
		swap(r,q);
		
		HilbertField z = q*r;
		alpha.push_back(realpart(z)); // alpha=<q|r>;
		mult_add(-alpha.back(), q, r); // r=r-q*alpha
		
		// computing beta_j
		beta.push_back(norm(r));
		EigensystemTridiagonal(true,alpha,beta,energy,evector);
		Ritz = abs(evector.back()*beta.back());
		if(verb && alpha.size()%10 == 0){
			cout.precision(10);
			cout << "--> iteration " << beta.size() << ", evalue = " << energy[0] << ", residual = " << Ritz << endl;
		}
    	if(alpha.size() > max_iter_lanczos) qcm_ED_throw("Lanczos procedure exceeded " + to_string(max_iter_lanczos) + " iterations");
	}
	niter = beta.size();
}




//! Implementation of the Lanczos method for the lowest eigenvector
/**
 Calculates the eigenvector associated with the lowest eigenvalue previously calculated using LanczosEigenvalue.
 Assumes the multiply-add matrix routine is usually provided by a struct method
 @param H	pointer (cast to void) to the struct object to which the multiply-add routine belongs
 @param gs	Initial state (provided on input. Modified by the routine).
 @param alpha  Diagonal of the tridiagonal matrix (computed by LanczosEigenvalue).
 @param beta  Second Diagonal of the tridiagonal matrix (computed by LanczosEigenvalue).
 */
template<typename T, typename HilbertField>
void LanczosEigenvector(T &H, vector<HilbertField> &gs, vector<double> &alpha, vector<double> &beta, bool verb=false)
{
	if(verb) cout << "Lanczos: calculation of the eigenvector..." << endl;
	vector<HilbertField> r(gs); // residue. beware of the default copy constructor !!!
	vector<HilbertField> q(gs.size()); // Lanczos basis vector
	
	vector<double> energy;
	vector<double> evector;
	
	EigensystemTridiagonal(true,alpha,beta,energy,evector);
	
	gs *= evector[0];
	for(size_t j=1; j<evector.size(); ++j){
    	check_signals();
		H.mult_add(r,q);
		mult_add(-alpha[j-1],r,q);
		for(size_t i=0; i<r.size(); i++){  // r = q/beta   &    q = -beta*r
			HilbertField tmp(r[i]);
			r[i] = q[i]/beta[j-1];
			q[i] = -beta[j-1]*tmp;
		}
		mult_add(evector[j],r,gs);
		if(verb && j%20 == 0) cout << "--> iteration " << j << endl;
	}

	// elective check at the end
	if(global_bool("check_lanczos_residual")){
		to_zero(q);
		H.mult_add(gs,q);
		mult_add(-energy[0],gs,q);
		cout << "norm of the ground state : " << norm(gs) << endl;
		cout << "norm of the Ritz residual : " << norm(q) << endl;
	}
}





//! Implementation of the Lanczos method for the Green function
/**
 Performs a Lanczos procedure on the vector psi and thus calculates the coefficients of the tridiagonal matrix: alpha and beta.
 Used in calculating Green function in the continued fraction representation
 @param H	pointer (cast to void) to the struct object to which the multiply-add routine belongs
 @param psi	[in] starting vector (provided on input).
 */
template<typename T, typename HilbertField>
pair<vector<double>, vector<double>> LanczosGreen(T &H, vector<HilbertField> &psi, bool verb=false)
{
	vector<HilbertField> q(psi.size());
	
	vector<double> a;
	vector<double> b;
	b.push_back(1.0);
  
  double accur_continued_fraction = global_double("accur_continued_fraction");
  size_t max_iter_CF = global_int("max_iter_CF");
  
  double prod = 1.0;
	
//  while(fabs(b.back()) > accur_continued_fraction and a.size() < max_iter_CF){
  while(prod > accur_continued_fraction and fabs(b.back()) > accur_continued_fraction and a.size() < max_iter_CF){
    check_signals();
	if(a.size()) swap(q,psi,b.back()); // psi <-- q/beta   and   q <-- -beta*psi
	H.mult_add(psi,q); //calculate q = q + H*psi
	HilbertField z = psi*q;
	a.push_back(real(z));
	mult_add(-a.back(),psi,q); // q -= a*psi
    double zz = norm(q);
	b.push_back(zz);
    prod *= fabs(zz);
	}
  return {a,b};
}






//! Implementation of the band Lanczos method for the Green function
/**
 band Lanczos method.
 See the chapter by Freund in the SIAM book
 www.cs.utk.edu/~dongarra/etemplates
 
 - steps indicated in comments refer to Freund's algorithm in the SIAM book
 - indices start from 0 here, contrary to Freund's text where they start from 1.
 Adjustements are therefore brought to some formula of his algorithm
 - Also, Freund's text does not treat memory management.
 
 This is used to calculate the Lehmann representation of the Green function of a Hamiltonian
 
 WARNING : this routine allocates memory (for \a evalues, \a evec_red, \a P0) without liberating it.
 This memory must be freed by the routines calling this one.
 This is necessary as the size of these arrays depends on the proceeding of this routine.
 
 @param H		pointer (cast to void) to the struct object to which the multiply-add routine belongs
 @param phi		Initial vectors (provided on input. Not modified by the routine except if eigenvectors_flag=true).
 @param evalues	Eigenvalues of the reduced Hamiltonian. Allocated inside the routine.
 @param evec_red	Eigenvectors of the reduced Hamiltonian. Allocated inside the routine.
 @param P0		matrix of inner products \<b[i]|v[j]\> (returned on output). Allocated inside the routine.
 @param M0		Number of iterations
 */
template<typename TYPE, typename HilbertField>
bool bandLanczos(
				 TYPE &H,
				 vector<vector<HilbertField>> &phi,
				 vector<double> &evalues,
				 matrix<HilbertField> &evec_red,
				 matrix<HilbertField> &P0,
				 int &M0,
				 bool verb=false
				 )
{
	int i,j,k;
	int nd=0;			// number of deflated vectors
	int nvec;			// number of vectors currently allocated
	int nvec_max;		// maximum number of vectors currently allocated
	
	double accur_deflation = global_double("accur_deflation");
	double accur_band_lanczos = global_double("accur_band_lanczos");
	if(accur_band_lanczos<0.0) accur_band_lanczos = 0.0;
	size_t max_iter_BL = global_int("max_iter_BL");

	double band_lanczos_minimum_gap = global_double("band_lanczos_minimum_gap");
	bool no_degenerate_BL = global_bool("no_degenerate_BL");
  
	size_t dim = phi[0].size();
	
	if(verb) cout << "\nband Lanczos procedure with " << phi.size() << " starting vectors. Dimension " << dim << endl;
	
	// make sure the targeted number of iterations is a multiple of the number of starting vectors
	int pc = (int)phi.size();
	if(M0 > max_iter_BL) M0 = max_iter_BL;
	int M0p = M0/(int)phi.size();
	M0p++;
	M0p *= phi.size();
	M0 = M0p-pc;
	
	vector<vector<HilbertField>> v;  // trial and orthogonalized vectors
	v.assign(M0p,vector<HilbertField>());
	
	size_t I[M0p];
	matrix<HilbertField> t(M0);
	matrix<HilbertField> P((int)phi.size(),M0); // matrix of inner products <b[i]|v[j]>
	vector<int> def(M0);
	
	// step (1)
	for(i=0; i<phi.size(); ++i){
		v[i] = phi[i];
	}
	nvec = 2*pc;
	nvec_max = nvec;
	
	// main loop over j
	bool converged = false;
	double ev_old=1e12;
	for(j=0; j<M0; ++j){
    check_signals();
		double z = norm(v[j]); // step (3)
    if(z < accur_deflation){	// need to deflate vector (step (4))
			if(verb) cout << "deflating vector no " << j << endl;
			if(j >= pc){		// step (4a) [adjusted]
				I[nd] = j-pc;	// note : at any time, j > indices in I[]
				def[j-pc] = 1;
				nd++;
			}
			pc--;			// step (4b)
			if(pc==0) break;
			erase(v[j]);	// freeing memory
			nvec--;
			for(k=j; k<j+pc; ++k) {	// step (4c)
				v[k] = v[k+1];
			}
			erase(v[j+pc]);
			j--; continue;		// step (4d)
		}
		v[j] *= 1.0/z;
		if(j>=pc){
			t(j,j-pc) = z;  // step (5)
		}
		
		// dot products with the starting vectors
		for(k=0; k<phi.size(); ++k) P(k,j) = phi[k]*v[j];
		
		size_t jm = j+pc;
		for(k=j+1; k<jm; ++k) {	// step (6)
			HilbertField zz = v[j]*v[k];
			mult_add(-zz, v[j], v[k]); // v_k = v_k - (v_j*.v_k) v_j
			if(k>=pc){
				t(j,k-pc) = zz;
			}
		}
		
		// step (7)
		v[j+pc].resize(dim); nvec++;
		if(nvec > nvec_max) nvec_max = nvec;
		H.mult_add(v[j],v[j+pc]);
		
		int k0 = 0;	// step (8)
		if(j>pc){
			k0 = j-pc;
			if(def[k0-1]==0){ // freeing memory on the way
				erase(v[k0-1]);
				nvec--;
			}
		}
		for(k=k0; k<j; ++k){
			t(k,j) = conjugate(t(j,k));
			mult_add(-t(k,j), v[k], v[j+pc]);	// v[j+pc] = v[j+pc] - z*v[k]
		}
		
		for(k=0; k<nd; ++k){ // step (9)
			assert(I[k]<j);
			HilbertField w = v[I[k]]*v[j+pc];
			mult_add(-w, v[I[k]], v[j+pc]); // v[j+pc] = v[j+pc] - z*v[I[k]]
			t(I[k],j) = w;
		}
		
		HilbertField w =  v[j]*v[j+pc];
		mult_add(-w, v[j], v[j+pc]);	// v[j+pc] = v[j+pc] - w*v[j]
		t(j,j) = w;
		
		// steps (10) and (11) combined
		for(k=0; k<nd; ++k) t(j,I[k]) = conjugate(t(I[k],j));
		
		// step (12) : test for convergence based on the lowest eigenvalues of the projected Hamiltonian
		
		double evalue_test;
		if(j>= 4*phi.size() and j%phi.size()==0){
			matrix<HilbertField> Hpr(j);
			Hpr.assign(t);
			//Moise: in the loop, we don't need the eigen vector, we need it only on output.
			//So we will compute it latter to save some time
			//evec_red.set_size(j);
			//evec_red.assign(Hpr);
			vector<double> evalues_tmp(j);
			evalues_tmp.resize(j);
			Hpr.eigenvalues(evalues_tmp);
			//TODO: we don't need all the eigenvalue nor the eigen vector, to optimize:
			//Only 2 eigenvalues are actually used
			
#ifdef BL_test
			for(size_t i=0; i<j; i++) fev << j << setprecision(10) << '\t' << evalues_tmp[i] << endl;
			fev << endl;
#endif
			
			evalue_test = abs(evalues_tmp[0]-ev_old); ev_old = evalues_tmp[0];
		  if(evalue_test < accur_band_lanczos) converged = true;
			
			int num=0;
			for(size_t i=0; i<v.size(); i++) if(v[i].size()>0) num++;
			if(verb){
				cout.precision(10);
				cout << "--> iteration " << j << ",\tdelta E = " << evalue_test << "\tgap = " << evalues_tmp[1]-evalues_tmp[0] << endl;
			}
			
			if (evalues_tmp[1]-evalues_tmp[0] < band_lanczos_minimum_gap and no_degenerate_BL){
        		qcm_ED_throw("band Lanczos: the gap between the first two eigenvalues is smaller than " + to_string(band_lanczos_minimum_gap));
			}
			if(converged) break;
		}
	}
	M0 = j;
	if(j==0) return false;
	
	//if(!converged)
	{
		matrix<HilbertField> Hpr(j);
		Hpr.assign(t);
		evec_red.set_size(j);
		evalues.resize(j);
		Hpr.eigensystem(evalues, evec_red);
	}
	P0.set_size((int)phi.size(),M0);
	P0.assign(P);
	
#ifdef BL_test
	fev << "\n\n#-----------------------------------\n";
	fev.close();
#endif
	
	if(verb) cout << M0 << " iterations" << endl;
	return true;
}


#endif
