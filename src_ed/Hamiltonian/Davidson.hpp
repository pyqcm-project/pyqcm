/*
 Implementation of the Davidson method to find the lowest eigenpairs of a large Hermitian matrix
 */
#ifndef Davidson_h
#define Davidson_h


#include <vector>
#include <iomanip>

#include "matrix.hpp"
#include "parser.hpp"

#define FACTOR 4
#define TINY 1e-14

//! Davidson algorithm for finding multiple eigenpairs
/**
 Davidson algorithm
 
 @param	hamil	Hamiltonian (templated)
 @param	 dim		dimension of the matrix
 @param	 K		number of states desired
 @param	val		eigenvalues
 @param	x		eigenstates (initial vectors)
 @param	accuracy		desired accuracy
 
 conditions on the virtual structure hamil:
 - must contain a method hamil.mult_add(vector<HilbertField> &x, vector<HilbertField> &y) that does y -->  y + H.x
 - must contain a method hamil.diag(vector<HilbertField> &d) that provides the diagonal d of H
 
 See "Monster matrices: their eigenvalues and eigenvectors"
 Davidson, Ernest R and Thompson, William J and others
 Computers in Physics 7 (519--522) 1993.
 
 */

template<typename T, typename HilbertField>
void Davidson(T &hamil, size_t dim, size_t K, vector<double> &val, vector<vector<HilbertField>> &x, double accuracy, bool verb=false)
{
	if(K > dim) K=dim;
	val.resize(K);
	
	size_t ncall = 0;
	
	
	// --------------------------------------------------------------------------
	// initialization
	
	size_t J = 3*K;; //! maximum number of vectors to hold in memory
	if(J > dim) J = dim;
	size_t Q; //! number of vectors added at each iteration
	
	if(verb){
		cout << "\nDavidson method for " << K << " vectors in sector " << hamil.B->sec.name() << ", dim = "  << dim << endl;
	}

	// finds the lowest diagonal elements of hamil and puts them into the index array lowest_diag_H
	vector<double> d(dim);
	hamil.diag(d);
	vector<size_t> lowest_diag_H;
	find_lowest(d,K,lowest_diag_H);
	
	// builds vector sets b and h
	vector<vector<HilbertField>> b, h, b_old;
	b.assign(K,vector<HilbertField>(dim));
	h.assign(K,vector<HilbertField>(dim));
	x.assign(K,vector<HilbertField>(dim));
	for(size_t k=0; k<K; k++){
		b[k][lowest_diag_H[k]]=1.0;
		hamil.mult_add(b[k],h[k]);
		ncall++;
	}
	
	//construction de bhb à partir des h trouvés
	matrix<HilbertField> bhb(b.size());
	
	for(size_t i=0; i<b.size(); i++) for(size_t j=0; j<b.size(); j++) bhb(i,j) = h[j][lowest_diag_H[i]];
	
	// --------------------------------------------------------------------------
	// Iterations start here
	
	vector<double> residuals;
	residuals.resize(K);
	
	size_t max_iter_lanczos = global_int("max_iter_lanczos");

	size_t iter;
	for(iter=0; iter<max_iter_lanczos; iter++){
		check_signals();

		//--------------------------------------------------------------------------
		// STEP B
		// eigenvalues and eigenvectors of bhb
		
#ifdef DAVIDSON_DEBUG
		cout << "--------------------------------------------\n";
		cout << "iteration " << iter << endl;
		cout << "b vectors:\n";
		for(size_t k=0; k<b.size(); k++) cout << "no " << k << " : " << b[k] << endl;
		cout << "h vectors:\n";
		for(size_t k=0; k<b.size(); k++) cout << "no " << k << " : " << h[k] << endl;
		cout << "bhb : \n" << bhb << endl;
#endif
		
		vector<double> mu(b.size());
		matrix<HilbertField> a(b.size());
		bhb.eigensystem(mu, a);
		
		// building the current eigenvectors x
		for(size_t k=0; k<K; k++){
			to_zero(x[k]);
			for(size_t p=0; p<b.size(); p++) mult_add(a(p,k), b[p], x[k]);
		}
		
#ifdef DAVIDSON_DEBUG
		cout << "bhb eigenvalues: " << mu << endl;
		cout << "bhb eigenvectors:\n" << a << endl;
		cout << "current eigenvectors:\n";
		for(size_t k=0; k<K; k++) cout << "no " << k << " : " << x[k] << endl;
		cout << "Norm of the residuals:\n";
#endif
		
		//--------------------------------------------------------------------------
		// STEP C
		// residues
		
		map<size_t,vector<HilbertField>> r; //! unconverged vectors
		
		for(size_t k=0; k<K; k++){
			vector<HilbertField> r_tmp(dim);
			for(size_t p=0; p<b.size(); p++) mult_add(a(p,k), h[p], r_tmp);
			mult_add(-mu[k]*HilbertField(1.0), x[k], r_tmp);
			residuals[k] = norm(r_tmp);
#ifdef DAVIDSON_DEBUG
			cout << "no " << k << "\tnorm = " << residuals[k]  << endl;
#endif
			if(residuals[k] > accuracy){
				r[k] = r_tmp;
			}
		}
		Q = r.size();
		if(Q == 0) break; //! all vectors are converged
		
		// if number of vectors in b in too few because of previous deletions, then allocated more:
		for(size_t i=b.size(); i<K+2*Q; i++) b.push_back(vector<HilbertField>(dim));
		
		if(verb && iter%10==0){
			cout.precision(12);
			cout << "--> iteration " << iter << ", evalue = " << mu[0] << ",\tresidual = " << residuals[K-1] << endl;
		}
		
		//--------------------------------------------------------------------------
		// STEP D
		
		// push current unconverged vectors to positions K to K+Q-1
		size_t j=K;
		for(auto &c : r){
			b[c.first].swap(b[j]);
			j++;
		}
		
		//! replacing b by x
		for(size_t k=0; k<K; k++) b[k].swap(x[k]);
		
		//--------------------------------------------------------------------------
		// adding Q vectors
		
		j=K+Q;
		for(auto& c : r){
			size_t k = c.first;
			for(size_t i=0; i<dim; i++){
				c.second[i] /= (-d[i]+mu[k]+TINY);
			}
			b[j].swap(c.second);
			j++;
		}
		r.clear();
		
		b.erase(b.begin()+K+2*Q,b.end());
		
		
		// Gram-Schmidt
		gram_schmidt(b, 0, TINY);
		
		if(b.size() == K) break;
		
#ifdef DAVIDSON_DEBUG
		matrix<HilbertField> bb(b.size());
		for(size_t i=0; i<b.size(); i++){
			bb(i,i) = b[i]*b[i];
			for(size_t k=0; k<i; k++){
				bb(i,k) = b[k]*b[i];
				bb(k,i) = conjugate(bb(i,k));
			}
		}
		
		cout << "b orthogonality matrix:\n" << bb << endl;
#endif
		
		
		//--------------------------------------------------------------------------
		// STEP F
		// building h and bhb for the added b's
		
		h.clear();
		for(size_t k=0; k<b.size(); k++){
			vector<HilbertField> tmp(dim);
			hamil.mult_add(b[k],tmp);
			ncall++;
			h.push_back(tmp);
		}
		
		bhb.set_size(b.size());
		for(size_t i=0; i<b.size(); i++){
			bhb(i,i) = b[i]*h[i];
			for(size_t k=0; k<i; k++){
				bhb(i,k) = b[i]*h[k]; // important: relative place of i and k in the Hermitian case
				bhb(k,i) = conjugate(bhb(i,k));
			}
		}
		
		
		for(size_t k=0; k<K; k++) val[k]=mu[k];
		
		
	}
	if(iter == max_iter_lanczos) qcm_ED_throw("Davidson method failed to converge!");
	else if(verb) cout << ncall << " calls to H and " << iter << " iterations" << endl;
}


#endif

