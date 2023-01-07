/*
 Implementation of the Hamiltonian in CSR format (own implementation)
*/

#ifndef Hamiltonian_csr
#define Hamiltonian_csr

#include "Hamiltonian_base.hpp"
#include "CSR_hermitian.hpp"


template<typename HilbertField>
class Hamiltonian_CSR : public Hamiltonian<HilbertField>
{
    public:
    
        CSR_hermitian<HilbertField> H_csr; //!< CSR matrix of the Hamiltonian
        
        Hamiltonian_CSR(
            shared_ptr<model> the_model, 
            const map<string, double> &value,
            sector _sec
        );
        void mult_add(vector<HilbertField> &x, vector<HilbertField> &y);
        void diag(vector<double> &d);
        //vector<shared_ptr<state<HilbertField>>> states(double& GS_energy);
        
    private:
        map<shared_ptr<HS_Hermitian_operator>, double> sparse_ops; //!< correpondence between terms in H and their coefficients
        void HS_ops_map(const map<string, double> &value);

};


/**
 constructor
 */
template<typename HilbertField>
Hamiltonian_CSR<HilbertField>::Hamiltonian_CSR(
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

    int num=1;
    #ifdef _OPENMP
    num = omp_get_max_threads()/omp_get_num_threads();
    #endif
    map<index_pair,HilbertField> E;
    if(global_bool("CSR_sym_store") and num>1) H_csr.sym_store = true; // set up the CSR format for openMP parallelization
    else H_csr.sym_store = false;
    H_csr.diag.assign(this->dim, 0.0);
    if(global_bool("verb_Hilbert")) {
        if(H_csr.sym_store) cout << "constructing the CSR Hamiltonian (stored symmetrically for openMP with " << num << " threads)..." << endl;
        else cout << "constructing the CSR Hamiltonian..." << endl;
    }
    for(auto& h : sparse_ops){
        h.first->CSR_map(E, H_csr.diag, h.second, H_csr.sym_store);
    }
    size_t row = 0;
    size_t count=0;
    H_csr.Iptr.reserve(this->dim/2);
    H_csr.J.reserve(E.size());
    H_csr.v.reserve(E.size());
    H_csr.Iptr.push_back(0);
    for(auto &x : E) {
        if(x.first.r != row) {
            for(size_t i=row; i<x.first.r; i++) H_csr.Iptr.push_back(count);
            row = x.first.r;
        }
        H_csr.J.push_back(x.first.c);
        H_csr.v.push_back(x.second);
        count++;
    }
    H_csr.Iptr.push_back(count);
    this->H_ptr = &H_csr;
}


/**
 Applies the Hamiltonian: y = y +H.x
 @param y vector to which H.x is added to
 @param x input vector
 */
template<typename HilbertField>
void Hamiltonian_CSR<HilbertField>::mult_add(
    vector<HilbertField> &x, 
    vector<HilbertField> &y
) {
    H_csr.apply(x, y); // applies the CSR matrix
}


/**
 provides the diagonal d of H
 Used by the Davidson method
 @param d the diagonal of H (pre-allocated)
 */
template<typename HilbertField>
void Hamiltonian_CSR<HilbertField>::diag(vector<double> &d){
    for(auto& h : sparse_ops) h.first->diag(d, h.second);
}

/**
 builds HS_operators as needed
 */
template<typename HilbertField>
void Hamiltonian_CSR<HilbertField>::HS_ops_map(const map<string, double> &value)
{
    bool is_complex = false;
    if(typeid(HilbertField) == typeid(Complex)) is_complex = true;
    //create a vector of values keys to have random access iterator
    vector<string> keys;
    keys.reserve(value.size());
    for (auto& x : value) {
        keys.push_back(x.first);
    }
    //construct the hamiltonian in parallel
    #pragma omp parallel for schedule(dynamic, 1)
    //for (auto& x : value){
    for (auto& x : keys) {
        Hermitian_operator& op = *this->the_model->term.at(x);
        if(op.HS_operator.find(this->sec) == op.HS_operator.end()){
            op.HS_operator[this->sec] = op.build_HS_operator(this->sec, is_complex); // ***TEMPO***
        }
    }
    keys.resize(0);
    //then add it to sparse_ops
    for(const auto& x : value){
        Hermitian_operator& op = *this->the_model->term.at(x.first);
        sparse_ops[op.HS_operator.at(this->sec)] = x.second; //value.at(x.first);
    }
}


#endif
