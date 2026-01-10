/*
 Implementation of the Hamiltonian in CSR format with Eigen
*/

#ifndef Hamiltonian_eigen
#define Hamiltonian_eigen

#include "Hamiltonian_base.hpp"
#include <Eigen/SparseCore>
#include <Eigen/Dense>
#include <vector>


template<typename HilbertField>
class Hamiltonian_Eigen : public Hamiltonian<HilbertField>
{
    public:
    
        Eigen::SparseMatrix<HilbertField,Eigen::RowMajor> H_eigen;
        
        Hamiltonian_Eigen(
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
Hamiltonian_Eigen<HilbertField>::Hamiltonian_Eigen(
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
    
    if(global_bool("verb_ED")) cout << "assembling the Hamiltonian sparse matrix" << endl;

    vector<matrix_element<HilbertField>> tripletList;
    for (auto& h : sparse_ops) {
        h.first->Triplet_COO_map(tripletList, h.second, true);
    }
    //create matrix
    H_eigen.resize(this->dim,this->dim);
    H_eigen.setFromTriplets(tripletList.begin(), tripletList.end());
    this->H_ptr = &H_eigen;
}


/**
 Applies the Hamiltonian: y = y +H.x
 @param y vector to which H.x is added to
 @param x input vector
 */
template<typename HilbertField>
void Hamiltonian_Eigen<HilbertField>::mult_add(
    vector<HilbertField> &x, 
    vector<HilbertField> &y
) {
    const Eigen::Map< Eigen::Matrix<HilbertField,Eigen::Dynamic,1> > xe(x.data(), x.size());
    Eigen::Map< Eigen::Matrix<HilbertField,Eigen::Dynamic,1> > ye(y.data(), y.size());
    ye += H_eigen*xe; //this change value of ye in place, so for y
}


/**
 provides the diagonal d of H
 Used by the Davidson method
 @param d the diagonal of H (pre-allocated)
 */
template<typename HilbertField>
void Hamiltonian_Eigen<HilbertField>::diag(vector<double> &d){
    //for (size_t i=0; i<d.size(); i++) d[i] = H_eigen.diagonal()[i];
    for(auto& h : sparse_ops) h.first->diag(d, h.second);
}

/**
 builds HS_operators as needed
 */
template<typename HilbertField>
void Hamiltonian_Eigen<HilbertField>::HS_ops_map(const map<string, double> &value)
{
    
    bool is_complex = false;
    if(typeid(HilbertField) == typeid(Complex)) is_complex = true;
    //create a vector of values keys to have random access iterator
    vector<string> keys;
    keys.reserve(value.size());
    for (auto& x : value) {
        keys.push_back(x.first);
    }
    //construct the Hamiltonian in parallel
    // #pragma omp parallel for
    for(size_t i = 0; i < keys.size(); ++i){
        Hermitian_operator& op = *this->the_model->term.at(keys[i]);
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
