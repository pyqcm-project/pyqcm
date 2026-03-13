/*
Base abstract class for a Hamiltonian of a given model_instance in a given
Hilbert space sector that serve as a proxy for the multiple implementation of
the Hamiltonian (Dense, legacy CSR, Eigen, ...)
Also provide default method that could be overwrite by implementation
*/

#ifndef Hamiltonian_base_h
#define Hamiltonian_base_h

#include "model.hpp"
#include "state.hpp"
#include "../Operators/Hermitian_operator.hpp"
#include "Q_matrix.hpp"
#include "Lanczos.hpp"
#include "Davidson.hpp"

#ifdef WITH_PRIMME
#include "PRIMME_solver.hpp"
#endif

extern double max_gap;
extern std::normal_distribution<double> normal_dis;

template<typename HilbertField>
class Hamiltonian
{
    //legacy, unused attribute
    //H_FORMAT format; //now implicit with the implementation
    //CSR_hermitian<HilbertField> csr; //private property of the implementation
    //matrix<HilbertField> H_dense; 
    
    public:
    
        void* H_ptr; //base pointer to the hamiltonian
        size_t dim; //!< dimension of the HS sector on which the Hamiltonian acts
        //this should not be belonging to the Hamiltonian class
        sector sec; //!< sector of the HS on which the Hamiltonian acts
        shared_ptr<ED_mixed_basis> B; //!<  pointer to basis of the space on which the Hamiltonian is defined
        shared_ptr<model> the_model; //!< backtrace to the cluster model
        
        Hamiltonian() {};
        virtual ~Hamiltonian() {};
    
        virtual void mult_add(vector<HilbertField> &x, vector<HilbertField> &y) {};
        virtual void diag(vector<double> &d) {};
        virtual double GS_energy();
        virtual vector<shared_ptr<state<HilbertField>>> states(double& GS_energy);
        virtual void print(ostream& fout) {};
        //virtual matrix<HilbertField> to_dense();
        
        virtual Q_matrix<HilbertField> build_Q_matrix(vector<vector<HilbertField>> &phi);
        
    
    private:
    
        vector<double> alpha; //!< main diagonal of the projected Hamiltonian in the Lanczos basis
        vector<double> beta; //!< second diagonal of the projected Hamiltonian in the Lanczos basis

        map<shared_ptr<HS_Hermitian_operator>, double> HS_ops_map(const map<string, double> &value);
        map<shared_ptr<Hermitian_operator>, double> ops_map(const map<string, double> &value);
        //void dense_form();
    
};


// ######################################
// Common method for every implementation
// ######################################

/**
 Applies the Lanczos algorithm for the lowest energy.
 Common with Hamiltonian in CSR and Factorized format.
 @param alpha : first diagonal of the tridiagonal representation (returned by reference)
 @param beta : second diagonal of the tridiagonal representation (returned by reference)
returns the GS energy.
 */
template<typename HilbertField>
double Hamiltonian<HilbertField>::GS_energy()
{
    vector<double> energy;
    size_t niter = 0;
    vector<HilbertField> x(dim);
    random(x, normal_dis);
    LanczosEigenvalue(*this, x, alpha, beta, energy, niter,  global_bool("verb_ED"));
    return energy[0];
}


/**
 Applies the Lanczos or Davidson-Liu algorithm for the lowest-energy states and energies.
 Common with Hamiltonian in CSR and Factorized format.
 returns a vector of pointers to states, to be treated by the parent model_instance.
 @param GS_energy : current ground-state energy of the model_instance, to be updated
 */
template<typename HilbertField>
vector<shared_ptr<state<HilbertField>>> Hamiltonian<HilbertField>::states(double& GS_energy)
{
    vector<shared_ptr<state<HilbertField>>> low_energy_states;
  
    vector<double> evalues;
    evalues.resize(1);
    vector<vector<HilbertField> > evectors;

    int ndouble = sizeof(HilbertField) / sizeof(double);
#ifdef WITH_PRIMME
    //use the previous eigenvectors
    if (global_bool("Ground_state_init_last")) {
        if (the_model->last_eigenvectors[sec].size() == 0) { //initialise random
            the_model->last_eigenvectors[sec].resize(ndouble*dim);
            random(the_model->last_eigenvectors[sec], normal_dis);
        }
        //this create a copy of the previous eigenvector to work on
        //this could be optimized to not duplicate the ground state vector for example=
        evectors.assign(1, vector<HilbertField> (
            (HilbertField*) the_model->last_eigenvectors[sec].data(), 
            (HilbertField*) the_model->last_eigenvectors[sec].data() + dim
        )); 
        evalues[0] = the_model->last_eigenvalues[sec];
    }
    else {
        evectors.resize(1);
        evectors[0].resize(dim);
        random(evectors[0], normal_dis);
    }
#else
    evectors.resize(1);
    evectors[0].resize(dim);
    random(evectors[0], normal_dis);
#endif

    char method = global_char("Ground_state_method");
#ifdef WITH_PRIMME
    if (method != 'P' and global_bool("Ground_state_init_last")) {
        cout << "ED WARNING! : Ground_state_init_last can only be used with the PRIMME solver!" << endl;
        cout << "Switch automatically to PRIMME solver." << endl;
        set_global_char("Ground_state_method", 'P');
        method = 'P';
    }
#endif

    if (method == 'D' or global_int("Davidson_states") > 1) { //Davidson method
        size_t Davidson_states = global_int("Davidson_states");
        Davidson(*this, dim, Davidson_states, evalues, evectors, global_double("accur_Davidson"),  global_bool("verb_ED"));
        if(Davidson_states > 1) {
            if(evalues.back()-evalues[0] < max_gap and global_bool("verb_warning")) {
                cout << "ED WARNING! : not enough Davidson states (" << Davidson_states << ") in sector " << sec.name() << endl;
            }
        }
    }
    else if (method == 'L') { //Default Lanczos method
        Lanczos(*this, dim, evalues[0], evectors[0],  global_bool("verb_ED"));
    }
#ifdef WITH_PRIMME
    else if (method == 'P') { //call PRIMME eigensolver
        PRIMME_state_solver(this, dim, evalues[0], evectors[0],  global_bool("verb_ED"));
    }
#endif
    else {
        qcm_ED_throw("Unknown method in Ground_state_method global parameter. If you set Ground_state_method to 'P', please compiled qcm_wed with PRIMME support (see installation documentation)");
    }
    //change GS_energy value
    if(evalues[0] < GS_energy) GS_energy = evalues[0];
    for(size_t i=0; i<evectors.size(); i++){
        if(evalues[i]-GS_energy > max_gap) continue;
        auto gs = make_shared<state<HilbertField>>(sec,dim);
        gs->energy = evalues[i];
        gs->psi = evectors[i];
        low_energy_states.push_back(gs);
    }
    
    //write the new previous eigenvector
#ifdef WITH_PRIMME
    if (global_bool("Ground_state_init_last")) {
        the_model->last_eigenvectors[sec] = vector<double> (
            (double*) evectors[0].data(),
            (double*) evectors[0].data() + ndouble*dim
        );
        the_model->last_eigenvalues[sec] = evalues[0];
    }
#endif
    return low_energy_states;
}


/**
 Constructs the Q_matrix (Lehmann representation) from the band Lanczos method,
 or full diagonalization if the dimension is small enough.
 @param phi the initial vectors
 */
template<typename HilbertField>
Q_matrix<HilbertField> Hamiltonian<HilbertField>::build_Q_matrix(
    vector<vector<HilbertField>> &phi
) {
    if(this->dim == 0 or phi.size()==0){
        return Q_matrix<HilbertField>(0,0);
    }
  
    int max_banditer = (int)(14*phi.size()*log(1.0*this->dim));
    int M = max_banditer; // essai
    QCM_ASSERT(M>1);

    vector<double> eval; // eigenvalues of the reduced Hamiltonian
    matrix<HilbertField> U;  // eigenvectors of the reduced Hamiltonian
    matrix<HilbertField> P; // matrix of inner products <b[i]|v[j]>
  
    if(bandLanczos(*this, phi, eval, U, P, M,  global_bool("verb_ED"))){
        Q_matrix<HilbertField> Q(phi.size(),M);
        if(Q.M > 0) {
            Q.e = eval; 
            Q.v.product(P,U,phi.size()); //  tempo
        }
        return Q;
    }
    else return Q_matrix<HilbertField>(phi.size(),0);
}


#endif
