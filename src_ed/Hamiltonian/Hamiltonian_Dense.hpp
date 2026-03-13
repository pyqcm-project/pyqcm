/*
 Implementation of the Hamiltonian in dense form
*/

#ifndef Hamiltonian_dense
#define Hamiltonian_dense

#include "Hamiltonian_base.hpp"
#include "model.hpp"
#include "state.hpp"
#include "Q_matrix.hpp"

extern double max_gap;

template<typename HilbertField>
class Hamiltonian_Dense : public Hamiltonian<HilbertField>
{
    public:
    
        matrix<HilbertField> H_dense; //!< dense form of the Hamiltonian (used when the dimension is small)
        
        Hamiltonian_Dense(
            shared_ptr<model> the_model, 
            const map<string, double> &value,
            sector _sec
        );
        void mult_add(vector<HilbertField> &x, vector<HilbertField> &y);
        void diag(vector<double> &d);
        double GS_energy();
        vector<shared_ptr<state<HilbertField>>> states(double& GS_energy);
        void print(ostream& fout);
        matrix<HilbertField> dense_form();
        Q_matrix<HilbertField> build_Q_matrix(vector<vector<HilbertField>> &phi);
    
    private:
    
        map<shared_ptr<HS_Hermitian_operator>, double> sparse_ops; //!< correpondence between terms in H and their coefficients
        void HS_ops_map(const map<string, double> &value);

};

/**
 constructor
 */
template<typename HilbertField>
Hamiltonian_Dense<HilbertField>::Hamiltonian_Dense(
    shared_ptr<model> _the_model,
    const map<string, double> &value, 
    sector _sec
) {
    this->the_model = _the_model;
    this->sec = _sec;
    this->B = _the_model->provide_basis(_sec);
    this->dim = this->B->dim;
    if(this->dim == 0) return;
    HS_ops_map(value);
    H_dense.set_size(this->dim);
    for(auto& i : sparse_ops) i.first->dense_form(H_dense, i.second);
    if(global_bool("print_Hamiltonian")) print(cout);
}


/**
 Applies the Hamiltonian: y = y +H.x
 @param y vector to which H.x is added to
 @param x input vector
 */
template<typename HilbertField>
void Hamiltonian_Dense<HilbertField>::mult_add(
    vector<HilbertField> &x,
    vector<HilbertField> &y
) {
    H_dense.apply_add(x, y);
}


/**
 provides the diagonal d of H
 Used by the Davidson method
 @param d the diagonal of H (pre-allocated)
 */
template<typename HilbertField>
void Hamiltonian_Dense<HilbertField>::diag(vector<double> &d){
    for(auto& h : sparse_ops) h.first->diag(d, h.second);
}


/**
 Momber function for dense Hamiltionain that use LAPACK routine to find the lowest energy.
 @param alpha : first diagonal of the tridiagonal representation (returned by reference)
 @param beta : second diagonal of the tridiagonal representation (returned by reference)
returns the GS energy.
 */
template<typename HilbertField>
double Hamiltonian_Dense<HilbertField>::GS_energy()
{
    vector<double> evalues(this->dim);
    H_dense.eigenvalues(evalues);
    return evalues[0];
}


/**
 Applies the Lanczos or Davidson-Liu algorithm for the lowest-energy states and energies.
 returns a vector of pointers to states, to be treated by the parent model_instance.
 @param GS_energy : current ground-state energy of the model_instance, to be updated
 */
template<typename HilbertField>
vector<shared_ptr<state<HilbertField>>> Hamiltonian_Dense<HilbertField>::states(double& GS_energy)
{
    vector<shared_ptr<state<HilbertField>>> low_energy_states;
    vector<double> evalues(this->dim);
    matrix<HilbertField> U( (int) this->dim );
    H_dense.eigensystem(evalues,U);
    if(evalues[0] < GS_energy) GS_energy = evalues[0];
    for(size_t i=0; i<evalues.size(); i++) {
        if(evalues[i]-GS_energy > max_gap) break;
        auto gs = make_shared<state<HilbertField>>(this->sec, this->dim);
        U.extract_column(i,gs->psi);
        gs->energy = evalues[i];
        low_energy_states.push_back(gs);
    }
    return low_energy_states;
}

/**
 Prints the dense form of the Hamiltonian
 Mostly for debugging purposes on very small systems
 */
template<typename HilbertField>
void Hamiltonian_Dense<HilbertField>::print(ostream& fout)
{
    if(H_dense.v.size() == 0) return;
    banner('~', "Hamiltonian", fout);
    fout << *this->B;
    fout << "Hamiltonian (dense form):\n";
    fout << H_dense;
}

/**
 Puts a dense form of the Hamiltonian matrix in the matrix H_dense
 Mostly for debugging purposes, but also for small dimensions.
 */
template<typename HilbertField>
matrix<HilbertField> Hamiltonian_Dense<HilbertField>::dense_form()
{
    return H_dense;
}


/**
 Constructs the Q_matrix (Lehmann representation) from full diagonalization.
 @param phi the initial vectors
 */
template<typename HilbertField>
Q_matrix<HilbertField> Hamiltonian_Dense<HilbertField>::build_Q_matrix(
    vector<vector<HilbertField>> &phi
) {
    if(this->dim == 0 or phi.size()==0){
        return Q_matrix<HilbertField>(0,0);
    }
  
    if(global_bool("verb_ED")) cout << "Q_matrix : full diagonalization" << endl;
    Q_matrix<HilbertField> Q(phi.size(), this->dim);
    vector<HilbertField> y(this->dim);
    matrix<HilbertField> U(H_dense);
    H_dense.eigensystem(Q.e, U);
    QCM_ASSERT(U.is_unitary(1e-6));
    for(size_t i=0; i<phi.size(); ++i){
        to_zero(y);
        U.left_apply_add(phi[i],y); // does y = phi[i] . U
        Q.v.insert_row(i,y); // inserts y as the ith row of Q
    }
    return Q;
}

/**
 builds HS_operators as needed
 */
template<typename HilbertField>
void Hamiltonian_Dense<HilbertField>::HS_ops_map(const map<string, double> &value)
{
    bool is_complex = false;
    if(typeid(HilbertField) == typeid(Complex)) is_complex = true;
    for(auto& x : value){
        Hermitian_operator& op = *this->the_model->term.at(x.first);
        if(op.HS_operator.find(this->sec) == op.HS_operator.end()){
            op.HS_operator[this->sec] = op.build_HS_operator(this->sec, is_complex); // ***TEMPO***
        }
        sparse_ops[op.HS_operator.at(this->sec)] = value.at(x.first);
    }
    return;
}

#endif
