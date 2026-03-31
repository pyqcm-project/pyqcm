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
			QCM_ASSERT(I[k]<j);
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


//! Implementation of the block Lanczos method
/**
 Block Lanczos method. Builds a block-tridiagonal projection of H onto the
 Krylov subspace generated by the starting block phi.

 The block three-term recurrence is:
   H Q_j = Q_{j-1} B_{j-1}^H + Q_j A_j + Q_{j+1} B_j

 where A_j are p×p Hermitian diagonal blocks and B_j are p×p upper-triangular
 off-diagonal blocks obtained from the QR factorisation of the residual at each step.
 The block-tridiagonal projected Hamiltonian thus has diagonal blocks A[j] and
 off-diagonal blocks B[j] (with B[j]^H above the diagonal and B[j] below).

 - Works for both real (HilbertField=double) and complex (HilbertField=Complex) matrices.
 - The matrix H is accessed only through H.mult_add(x, y): y += H*x, matching bandLanczos().
 - Deflation is triggered when a diagonal element of B_j falls below accur_deflation.
 - Convergence is assessed every p steps by tracking the lowest eigenvalue of the
   accumulated block-tridiagonal matrix; accur_band_lanczos and max_iter_BL are used.

 @param H     Object with a mult_add(x, y) method: y += H*x
 @param phi   Starting block of p vectors (each of size dim). Orthonormalized internally;
              the vectors are not modified on output.
 @param A     Diagonal blocks A[j] (p×p Hermitian matrices, allocated and filled on output).
 @param B     Off-diagonal QR blocks B[j] (p×p upper-triangular matrices, output).
 @param M0    Maximum number of block Lanczos steps on input; actual number on output.
 @param verb  Verbose flag.
 @returns true if the lowest eigenvalue converged within M0 steps, false otherwise.
 */
template<typename TYPE, typename HilbertField>
bool blockLanczos(
    TYPE &H,
    vector<vector<HilbertField>> &phi,
    vector<matrix<HilbertField>> &A,
    vector<matrix<HilbertField>> &B,
    int &M0,
    bool verb=false
)
{
    int p = (int)phi.size();
    size_t dim = phi[0].size();

    double accur_deflation   = global_double("accur_deflation");
    double accur_band_lanczos = global_double("accur_band_lanczos");
    if(accur_band_lanczos < 0.0) accur_band_lanczos = 0.0;
    size_t max_iter_BL = global_int("max_iter_BL");
    if(M0 > (int)max_iter_BL) M0 = (int)max_iter_BL;

    if(verb) cout << "\nblock Lanczos: block size p=" << p << ", dim=" << dim << endl;

    A.clear();
    B.clear();

    // Three rolling blocks of p Hilbert-space vectors
    vector<vector<HilbertField>> Q_prev;                              // Q_{j-1}
    vector<vector<HilbertField>> Q_cur(p, vector<HilbertField>(dim)); // Q_j (current)
    vector<vector<HilbertField>> Z(p, vector<HilbertField>(dim));     // workspace / Q_{j+1}

    // --- Initialization: orthonormalize the starting block into Q_cur ---
    for(int k = 0; k < p; ++k) Q_cur[k] = phi[k];
    for(int l = 0; l < p; ++l){
        for(int k = 0; k < l; ++k){
            HilbertField z = Q_cur[k] * Q_cur[l];  // <Q_cur[k]|Q_cur[l]>
            mult_add(-z, Q_cur[k], Q_cur[l]);
        }
        double nrm = norm(Q_cur[l]);
        if(nrm < accur_deflation)
            qcm_ED_throw("blockLanczos: starting block is rank-deficient at column " + to_string(l));
        Q_cur[l] *= 1.0/nrm;
    }

    bool converged = false;
    double ev_old = 1e12;
    int j = 0;

    for(j = 0; j < M0; ++j){
        check_signals();

        // Step 1 — Z[k] = H * Q_cur[k]
        for(int k = 0; k < p; ++k){
            to_zero(Z[k]);
            H.mult_add(Q_cur[k], Z[k]);
        }

        // Step 2 — A_j(k,l) = <Q_cur[k] | Z[l]>   (Hermitian block)
        matrix<HilbertField> Aj(p);
        for(int l = 0; l < p; ++l){
            for(int k = 0; k <= l; ++k){
                HilbertField z = Q_cur[k] * Z[l];
                Aj(k,l) = z;
                Aj(l,k) = conjugate(z);
            }
        }
        A.push_back(Aj);

        // Step 3 — Z[l] -= sum_k A_j(k,l) * Q_cur[k]
        for(int l = 0; l < p; ++l)
            for(int k = 0; k < p; ++k)
                mult_add(-Aj(k,l), Q_cur[k], Z[l]);

        // Step 4 — Z[l] -= sum_k conj(B_{j-1}(l,k)) * Q_prev[k]   (j > 0 only)
        //         i.e.  Z -= Q_prev * B_{j-1}^H  column-by-column
        if(j > 0){
            const matrix<HilbertField> &Bprev = B.back();
            for(int l = 0; l < p; ++l)
                for(int k = 0; k < p; ++k)
                    mult_add(-conjugate(Bprev(l,k)), Q_prev[k], Z[l]);
        }

        // Step 5 — QR factorisation of Z via modified Gram-Schmidt:
        //           Z = Q_{j+1} * B_j  (B_j upper triangular)
        //   After this loop Z[l] holds Q_{j+1}[l] (orthonormal).
        matrix<HilbertField> Bj(p);
        bool singular_block = false;
        for(int l = 0; l < p; ++l){
            // project out already-orthonormalised columns
            for(int k = 0; k < l; ++k){
                HilbertField z = Z[k] * Z[l];  // <Z[k]|Z[l]>  (Z[k] already normalised)
                Bj(k,l) = z;
                mult_add(-z, Z[k], Z[l]);
            }
            double nrm = norm(Z[l]);
            if(nrm < accur_deflation){
                if(verb) cout << "block Lanczos: deflation at column " << l
                              << " of step " << j << endl;
                Bj(l,l) = HilbertField(0);
                // zero remaining columns of Z so they don't contaminate future steps
                for(int ll = l; ll < p; ++ll) to_zero(Z[ll]);
                singular_block = true;
                break;
            }
            Bj(l,l) = HilbertField(nrm);
            Z[l] *= 1.0/nrm;
        }
        B.push_back(Bj);

        // Step 6 — Convergence check every p steps (mirrors bandLanczos strategy)
        if((j+1) >= 2*p && (j+1) % p == 0){
            int sz = (j+1) * p;
            matrix<HilbertField> T(sz, sz);
            // diagonal blocks
            for(int jj = 0; jj <= j; ++jj)
                for(int r = 0; r < p; ++r)
                    for(int c = 0; c < p; ++c)
                        T(jj*p+r, jj*p+c) = A[jj](r,c);
            // off-diagonal blocks: T_{j+1,j} = B[j], T_{j,j+1} = B[j]^H
            for(int jj = 0; jj < j; ++jj)
                for(int r = 0; r < p; ++r)
                    for(int c = 0; c < p; ++c){
                        T((jj+1)*p+r, jj*p+c) = B[jj](r,c);
                        T(jj*p+c, (jj+1)*p+r) = conjugate(B[jj](r,c));
                    }
            vector<double> evals(sz);
            T.eigenvalues(evals);
            double delta = abs(evals[0] - ev_old);
            ev_old = evals[0];
            if(verb){
                cout.precision(10);
                cout << "--> block Lanczos step " << j+1
                     << ", delta E = " << delta << endl;
            }
            if(delta < accur_band_lanczos){ converged = true; break; }
        }

        if(singular_block) break;

        // Advance rolling window: Q_prev <- Q_cur, Q_cur <- Z
        Q_prev = Q_cur;
        Q_cur  = Z;
        for(int k = 0; k < p; ++k) to_zero(Z[k]);
    }

    M0 = j + 1;
    if(verb) cout << A.size() << " block Lanczos floors obtained" << endl;
    return converged;
}


//! Block Lanczos method with Hermitian (polar-decomposition) off-diagonal blocks
/**
 Variant of blockLanczos() in which the off-diagonal coupling blocks B[j] are
 Hermitian positive semi-definite instead of upper-triangular.

 The block three-term recurrence is identical:
   H Q_j = Q_{j-1} B_{j-1}^H + Q_j A_j + Q_{j+1} B_j

 but now B_j = B_j^H (Hermitian PSD), so the projected block-tridiagonal matrix
 has equal sub- and super-diagonal blocks: T[j,j+1] = B_j = T[j+1,j]^H.

 This is achieved by replacing the QR factorisation of the residual block Z with
 a polar decomposition:
   1. Compute the p×p Gram matrix  G_{kl} = <Z[k]|Z[l]>
   2. Eigendecompose  G = V D V^H  (D real non-negative)
   3. B_j = V sqrt(D) V^H        (Hermitian square root of G)
   4. Q_{j+1}[:,l] = sum_k (V D^{-1/2} V^H)_{kl} Z[k]

 Deflation is triggered when a singular value sqrt(D[m]) < accur_deflation.

 @param H     Object with a mult_add(x, y) method: y += H*x
 @param phi   Starting block of p vectors (each of size dim). Orthonormalized internally.
 @param A     Diagonal blocks A[j] (p×p Hermitian matrices, output).
 @param B     Off-diagonal blocks B[j] (p×p Hermitian PSD matrices, output).
 @param M0    Maximum number of block Lanczos steps on input; actual number on output.
 @param verb  Verbose flag.
 @returns true if the lowest eigenvalue converged within M0 steps, false otherwise.
 */
template<typename TYPE, typename HilbertField>
bool blockLanczosSVD(
    TYPE &H,
    vector<vector<HilbertField>> &phi,
    vector<matrix<HilbertField>> &A,
    vector<matrix<HilbertField>> &B,
    int &M0,
    bool verb=false
)
{
    int p = (int)phi.size();
    size_t dim = phi[0].size();

    double accur_deflation    = global_double("accur_deflation");
    double accur_band_lanczos = global_double("accur_band_lanczos");
    if(accur_band_lanczos < 0.0) accur_band_lanczos = 0.0;
    size_t max_iter_BL = global_int("max_iter_BL");
    if(M0 > (int)max_iter_BL) M0 = (int)max_iter_BL;

    if(verb) cout << "\nblock Lanczos (polar/SVD): block size p=" << p << ", dim=" << dim << endl;

    A.clear();
    B.clear();

    // Three rolling blocks of p Hilbert-space vectors
    vector<vector<HilbertField>> Q_prev;
    vector<vector<HilbertField>> Q_cur(p, vector<HilbertField>(dim));
    vector<vector<HilbertField>> Z(p, vector<HilbertField>(dim));

    // --- Initialisation: orthonormalize the starting block into Q_cur ---
    for(int k = 0; k < p; ++k) Q_cur[k] = phi[k];
    for(int l = 0; l < p; ++l){
        for(int k = 0; k < l; ++k){
            HilbertField z = Q_cur[k] * Q_cur[l];
            mult_add(-z, Q_cur[k], Q_cur[l]);
        }
        double nrm = norm(Q_cur[l]);
        if(nrm < accur_deflation)
            qcm_ED_throw("blockLanczosSVD: starting block is rank-deficient at column " + to_string(l));
        Q_cur[l] *= 1.0/nrm;
    }

    bool converged = false;
    double ev_old = 1e12;
    int j = 0;

    for(j = 0; j < M0; ++j){
        check_signals();

        // Step 1 — Z[k] = H * Q_cur[k]
        for(int k = 0; k < p; ++k){
            to_zero(Z[k]);
            H.mult_add(Q_cur[k], Z[k]);
        }

        // Step 2 — A_j(k,l) = <Q_cur[k] | Z[l]>   (Hermitian block)
        matrix<HilbertField> Aj(p);
        for(int l = 0; l < p; ++l){
            for(int k = 0; k <= l; ++k){
                HilbertField z = Q_cur[k] * Z[l];
                Aj(k,l) = z;
                Aj(l,k) = conjugate(z);
            }
        }
        A.push_back(Aj);

        // Step 3 — Z[l] -= sum_k A_j(k,l) * Q_cur[k]
        for(int l = 0; l < p; ++l)
            for(int k = 0; k < p; ++k)
                mult_add(-Aj(k,l), Q_cur[k], Z[l]);

        // Step 4 — Z[l] -= sum_k (B_{j-1}^H)_{k,l} * Q_prev[k]   (j > 0 only)
        //   Note: with Hermitian B_{j-1}, conj(B_{j-1}(l,k)) = B_{j-1}(k,l),
        //   so the formula is identical to the upper-triangular case.
        if(j > 0){
            const matrix<HilbertField> &Bprev = B.back();
            for(int l = 0; l < p; ++l)
                for(int k = 0; k < p; ++k)
                    mult_add(-conjugate(Bprev(l,k)), Q_prev[k], Z[l]);
        }

        // Step 5 — Polar decomposition of the residual block Z:
        //   Gram matrix  G_{kl} = <Z[k]|Z[l]>
        //   G = V D V^H  (eigendecomposition, D real non-negative)
        //   B_j = V diag(sqrt(D)) V^H    (Hermitian PSD square root)
        //   Q_{j+1}[:,l] = sum_k (V diag(D^{-1/2}) V^H)_{kl} Z[k]
        matrix<HilbertField> G(p);
        for(int l = 0; l < p; ++l)
            for(int k = 0; k <= l; ++k){
                HilbertField z = Z[k] * Z[l];
                G(k,l) = z;
                G(l,k) = conjugate(z);
            }

        vector<double> D(p);
        matrix<HilbertField> V(p);
        G.eigensystem(D, V);  // G = V diag(D) V^H, eigenvalues in D (ascending)

        // Check for deflation: singular value sqrt(D[m]) < accur_deflation
        bool singular_block = false;
        for(int m = 0; m < p; ++m){
            if(D[m] < accur_deflation * accur_deflation){
                if(verb) cout << "block Lanczos SVD: deflation at singular value " << m
                              << " (sigma=" << sqrt(max(D[m],0.0)) << ") at step " << j << endl;
                singular_block = true;
                break;
            }
        }

        // Build B_j = V diag(sqrt(D)) V^H  (Hermitian PSD)
        matrix<HilbertField> Bj(p);
        for(int r = 0; r < p; ++r)
            for(int c = 0; c < p; ++c){
                HilbertField s = HilbertField(0);
                for(int m = 0; m < p; ++m)
                    s += V(r,m) * HilbertField(sqrt(max(D[m],0.0))) * conjugate(V(c,m));
                Bj(r,c) = s;
            }
        B.push_back(Bj);

        if(singular_block) break;

        // Build Binv = V diag(D^{-1/2}) V^H  (inverse of B_j)
        matrix<HilbertField> Binv(p);
        for(int r = 0; r < p; ++r)
            for(int c = 0; c < p; ++c){
                HilbertField s = HilbertField(0);
                for(int m = 0; m < p; ++m)
                    s += V(r,m) * HilbertField(1.0/sqrt(D[m])) * conjugate(V(c,m));
                Binv(r,c) = s;
            }

        // New Q_{j+1}[:,l] = sum_k Binv_{kl} Z[k]
        //   (derived from Z_mat = Q_mat Bj  =>  Q_mat = Z_mat Binv)
        vector<vector<HilbertField>> Q_next(p, vector<HilbertField>(dim));
        for(int l = 0; l < p; ++l){
            to_zero(Q_next[l]);
            for(int k = 0; k < p; ++k)
                mult_add(Binv(k,l), Z[k], Q_next[l]);
        }

        // Advance rolling window: Q_prev <- Q_cur, Q_cur <- Q_next
        Q_prev = Q_cur;
        Q_cur  = Q_next;
        for(int k = 0; k < p; ++k) to_zero(Z[k]);

        // Step 6 — Convergence check every p steps (mirrors blockLanczos strategy)
        if((j+1) >= 2*p && (j+1) % p == 0){
            int sz = (j+1) * p;
            matrix<HilbertField> T(sz, sz);
            for(int jj = 0; jj <= j; ++jj)
                for(int r = 0; r < p; ++r)
                    for(int c = 0; c < p; ++c)
                        T(jj*p+r, jj*p+c) = A[jj](r,c);
            for(int jj = 0; jj < j; ++jj)
                for(int r = 0; r < p; ++r)
                    for(int c = 0; c < p; ++c){
                        T((jj+1)*p+r, jj*p+c) = B[jj](r,c);
                        T(jj*p+c, (jj+1)*p+r) = conjugate(B[jj](r,c));
                    }
            vector<double> evals(sz);
            T.eigenvalues(evals);
            double delta = abs(evals[0] - ev_old);
            ev_old = evals[0];
            if(verb){
                cout.precision(10);
                cout << "--> block Lanczos SVD step " << j+1
                     << ", delta E = " << delta << endl;
            }
            if(delta < accur_band_lanczos){ converged = true; break; }
        }
    }

    M0 = j + 1;
    if(verb) cout << A.size() << " block Lanczos SVD floors obtained" << endl;
    return converged;
}


//! Block Lanczos biorthogonalization method (non-symmetric / two-sided version)
/**
 Builds biorthogonal block Krylov subspaces for a Hermitian matrix H from
 separate right starting block phi_R and left starting block phi_L.
 Left and right Lanczos vectors W_j and V_j satisfy:
   <W_j[k] | V_j[l]> = delta_{kl}

 The non-symmetric block three-term recurrences are:
   H V_j = V_{j+1} B_j + V_j A_j + V_{j-1} C_{j-1}^H
   H W_j = W_{j+1} C_j + W_j A_j^H + W_{j-1} B_{j-1}^H

 The projected block-tridiagonal matrix T = W^H H V is:
   T[j*p+k, j*p+l]       = A_j(k,l)              (diagonal, generally non-Hermitian)
   T[j*p+k, (j+1)*p+l]   = C_j^H(k,l)            (superdiagonal)
   T[(j+1)*p+k, j*p+l]   = B_j(k,l)              (subdiagonal)

 When phi_L == phi_R the method reduces to the standard symmetric blockLanczos,
 with A_j Hermitian and C_j = B_j.

 @param H     Object with mult_add(x, y): y += H*x  (H assumed Hermitian: H = H^H)
 @param phi_R Right starting block: p vectors of dimension dim. Not modified on output.
 @param phi_L Left starting block:  p vectors of dimension dim. Not modified on output.
 @param A     Diagonal blocks A_j (output). Not Hermitian in general when phi_L != phi_R.
 @param B     Right off-diagonal blocks B_j (upper triangular, from QR of right residual).
 @param C     Left off-diagonal blocks C_j (from biorthogonal factorisation of left residual).
 @param M0    Maximum number of block steps on input; actual number on output.
 @param verb  Verbose flag.
 @returns true if the off-diagonal block norms fell below accur_band_lanczos.

 Breakdown occurs when C_j becomes singular (right/left Krylov spaces become
 linearly dependent), which is detected and triggers an early exit.
*/
template<typename TYPE, typename HilbertField>
bool blockLanczosBI(
    TYPE &H,
    vector<vector<HilbertField>> &phi_R,
    vector<vector<HilbertField>> &phi_L,
    vector<matrix<HilbertField>> &A,
    vector<matrix<HilbertField>> &B,
    vector<matrix<HilbertField>> &C,
    int &M0,
    bool verb=false
)
{
    int p = (int)phi_R.size();
    if((int)phi_L.size() != p)
        qcm_ED_throw("blockLanczosBI: phi_L and phi_R must have the same block size");
    size_t dim = phi_R[0].size();

    double accur_deflation    = global_double("accur_deflation");
    double accur_band_lanczos = global_double("accur_band_lanczos");
    if(accur_band_lanczos < 0.0) accur_band_lanczos = 0.0;
    size_t max_iter_BL = global_int("max_iter_BL");
    if(M0 > (int)max_iter_BL) M0 = (int)max_iter_BL;

    if(verb) cout << "\nblock Lanczos biorthogonalization: p=" << p << ", dim=" << dim << endl;

    A.clear(); B.clear(); C.clear();

    // Rolling blocks: _prev = level j-1, _cur = level j
    vector<vector<HilbertField>> V_prev, W_prev;
    vector<vector<HilbertField>> V_cur(p, vector<HilbertField>(dim));
    vector<vector<HilbertField>> W_cur(p, vector<HilbertField>(dim));
    vector<vector<HilbertField>> R(p, vector<HilbertField>(dim)); // right residual -> V_next
    vector<vector<HilbertField>> S(p, vector<HilbertField>(dim)); // left residual

    for(int k = 0; k < p; ++k){ V_cur[k] = phi_R[k]; W_cur[k] = phi_L[k]; }

    // --- Biorthogonal modified Gram-Schmidt initialisation ---
    // Goal: <W_cur[k] | V_cur[l]> = delta_{kl}
    // For each pair (l,l): first remove projections from all already-normalised pairs (k<l),
    // then normalise the pair symmetrically so <W_cur[l]|V_cur[l]> = 1.
    for(int l = 0; l < p; ++l){
        for(int k = 0; k < l; ++k){
            // Remove V_cur[k] influence from V_cur[l]: enforce <W_cur[k]|V_cur[l]> = 0
            HilbertField zv = W_cur[k] * V_cur[l];
            mult_add(-zv, V_cur[k], V_cur[l]);
            // Remove W_cur[k] influence from W_cur[l]: enforce <W_cur[l]|V_cur[k]> = 0
            HilbertField zw = W_cur[l] * V_cur[k];
            mult_add(-zw, W_cur[k], W_cur[l]);
        }
        // Normalise: <a*W_cur[l] | b*V_cur[l]> = 1
        // Choosing b = 1/|rho|^{1/2} and a = rho/(|rho|^{3/2}) gives the result.
        // (Here |rho| = sqrt(rho * conj(rho)).)
        HilbertField rho   = W_cur[l] * V_cur[l];
        double       arho2 = realpart(rho * conjugate(rho)); // |rho|^2
        if(arho2 < accur_deflation * accur_deflation)
            qcm_ED_throw("blockLanczosBI: starting block is biorthogonally rank-deficient at column " + to_string(l));
        double arho      = sqrt(arho2);          // |rho|
        double sqrt_arho = sqrt(arho);           // |rho|^{1/2}
        V_cur[l]        *= 1.0 / sqrt_arho;
        HilbertField scale_W = rho / (arho * sqrt_arho);
        for(size_t i = 0; i < dim; ++i) W_cur[l][i] *= scale_W;
    }

    bool converged = false;
    double ev_old  = 1e12;
    int j = 0;

    for(j = 0; j < M0; ++j){
        check_signals();

        // Step 1 — Compute R[k] = H V_cur[k]  and  S[k] = H W_cur[k]
        for(int k = 0; k < p; ++k){
            to_zero(R[k]);  H.mult_add(V_cur[k], R[k]);
            to_zero(S[k]);  H.mult_add(W_cur[k], S[k]);
        }

        // Step 2 — Diagonal block: A_j(k,l) = <W_cur[k] | R[l]>
        matrix<HilbertField> Aj(p);
        for(int k = 0; k < p; ++k)
            for(int l = 0; l < p; ++l)
                Aj(k,l) = W_cur[k] * R[l];
        A.push_back(Aj);

        // Step 3 — Subtract diagonal from residuals
        // Right: R[l] -= sum_k A_j(k,l)         * V_cur[k]
        // Left:  S[m] -= sum_k conj(A_j(m,k))   * W_cur[k]   [= A_j^H(k,m) W_cur[k]]
        for(int l = 0; l < p; ++l)
            for(int k = 0; k < p; ++k)
                mult_add(-Aj(k,l), V_cur[k], R[l]);
        for(int m = 0; m < p; ++m)
            for(int k = 0; k < p; ++k)
                mult_add(-conjugate(Aj(m,k)), W_cur[k], S[m]);

        // Step 4 — Subtract previous off-diagonal contribution (j > 0)
        // Right: R[m] -= sum_l conj(C_{j-1}(m,l)) V_prev[l]   [= C_{j-1}^H row m]
        // Left:  S[m] -= sum_l conj(B_{j-1}(m,l)) W_prev[l]   [= B_{j-1}^H row m]
        if(j > 0){
            const matrix<HilbertField> &Bprev = B.back();
            const matrix<HilbertField> &Cprev = C.back();
            for(int m = 0; m < p; ++m)
                for(int l = 0; l < p; ++l){
                    mult_add(-conjugate(Cprev(m,l)), V_prev[l], R[m]);
                    mult_add(-conjugate(Bprev(m,l)), W_prev[l], S[m]);
                }
        }

        // Step 5 — QR factorisation of R via modified Gram-Schmidt -> V_next stored in R, B_j
        matrix<HilbertField> Bj(p);
        bool singular_R = false;
        for(int l = 0; l < p; ++l){
            for(int k = 0; k < l; ++k){
                HilbertField z = R[k] * R[l];   // Euclidean inner product (R[k] already normalised)
                Bj(k,l) = z;
                mult_add(-z, R[k], R[l]);
            }
            double nrm = norm(R[l]);
            if(nrm < accur_deflation){
                if(verb) cout << "blockLanczosBI: deflation in R at column " << l << " of step " << j << endl;
                Bj(l,l) = HilbertField(0);
                for(int ll = l; ll < p; ++ll) to_zero(R[ll]);
                singular_R = true;
                break;
            }
            Bj(l,l) = HilbertField(nrm);
            R[l] *= 1.0 / nrm;
        }
        B.push_back(Bj);
        if(singular_R) break;

        // Step 6 — Compute C_j: C_j(m,l) = <V_next[m] | S[l]>
        // Derivation: S = W_next * C_j, and W_next^H V_next = I requires C_j^H = S^H V_next,
        // i.e. C_j(m,l) = <V_next[m]|S[l]> = conj(<S[l]|V_next[m]>) = conj(C_j^H(l,m)). ✓
        matrix<HilbertField> Cj(p);
        for(int m = 0; m < p; ++m)
            for(int l = 0; l < p; ++l)
                Cj(m,l) = R[m] * S[l];          // R[m] = V_next[m] (after QR above)
        C.push_back(Cj);

        // Check for left-side breakdown: C_j must be invertible
        double Cj_diag_norm = 0.0;
        for(int k = 0; k < p; ++k) Cj_diag_norm += realpart(Cj(k,k) * conjugate(Cj(k,k)));
        if(sqrt(Cj_diag_norm) < accur_deflation * p){
            if(verb) cout << "blockLanczosBI: left-side breakdown at step " << j << endl;
            break;
        }

        // Step 7 — Compute W_next = S * C_j^{-1}
        // W_next[m] = sum_l (C_j^{-1})_{lm} S[l]
        // Then <W_next[k]|V_next[l]> = delta_{kl} by construction (see derivation above).
        matrix<HilbertField> Cj_inv(Cj);
        Cj_inv.inverse();

        vector<vector<HilbertField>> W_next(p, vector<HilbertField>(dim));
        for(int m = 0; m < p; ++m){
            to_zero(W_next[m]);
            for(int l = 0; l < p; ++l)
                mult_add(Cj_inv(l,m), S[l], W_next[m]);
        }

        // Step 8 — Convergence check every p steps via eigenvalues of the Hermitian part of T
        // (T + T^H)/2 has diagonal blocks (A_j + A_j^H)/2 and off-diagonal blocks (B_j + C_j)/2
        if((j+1) >= 2*p && (j+1) % p == 0){
            int sz = (j+1) * p;
            matrix<HilbertField> T_herm(sz);
            for(int jj = 0; jj <= j; ++jj){
                // Hermitian part of diagonal block: (A[jj] + A[jj]^H) / 2
                for(int r = 0; r < p; ++r)
                    for(int c = 0; c < p; ++c){
                        HilbertField sym = (A[jj](r,c) + conjugate(A[jj](c,r))) * HilbertField(0.5);
                        T_herm(jj*p+r, jj*p+c) = sym;
                    }
                // Off-diagonal: (B[jj] + C[jj]) / 2 and its conjugate transpose
                if(jj < j){
                    for(int r = 0; r < p; ++r)
                        for(int c = 0; c < p; ++c){
                            HilbertField lo = (B[jj](r,c) + C[jj](r,c)) * HilbertField(0.5);
                            T_herm((jj+1)*p+r, jj*p+c) = lo;
                            T_herm(jj*p+c, (jj+1)*p+r) = conjugate(lo);
                        }
                }
            }
            vector<double> evals(sz);
            T_herm.eigenvalues(evals);
            double delta = abs(evals[0] - ev_old);
            ev_old = evals[0];
            if(verb){
                cout.precision(10);
                cout << "--> block Lanczos BI step " << j+1
                     << ", delta E (herm. part) = " << delta << endl;
            }
            if(delta < accur_band_lanczos){ converged = true; break; }
        }

        // Advance rolling window
        V_prev = move(V_cur);  W_prev = move(W_cur);
        V_cur  = move(R);      W_cur  = move(W_next);
        R.assign(p, vector<HilbertField>(dim));
        S.assign(p, vector<HilbertField>(dim));
    }

    M0 = j + 1;
    if(verb) cout << A.size() << " block Lanczos BI steps completed" << endl;
    return converged;
}


#endif
