#ifndef HS_general_interaction_operator_h
#define HS_general_interaction_operator_h

#include "HS_nondiagonal_operator.hpp"

//! Represents a general interaction operator in a sector of the Hilbert 
template<typename HS_field>
struct HS_general_interaction_operator : HS_nondiagonal_operator<HS_field>
{
  template<typename op_field>
  HS_general_interaction_operator(shared_ptr<model> _the_model, const string &_name, sector _sec, const vector<matrix_element<op_field>> &elements)
  : HS_nondiagonal_operator<HS_field>(_the_model, _name, _sec)
  {
    check_signals();
    
    shared_ptr<symmetry_group> group = this->B->group;
    HS_field X;
    size_t n = group->N;

    for (auto &x : elements) {
      // unfolding the elements
      int nn = 2 * n;
      int ei = x.r % nn;
      int ej = x.r / nn;
      int ek = x.c % nn;
      int el = x.c / nn;
      if((ei == ej) or (el == ek)) continue;
      // cout << "element (" << ei << ", "  << ej << " | " << ek << ", "<< el << ") : "  << x.v << endl; // TEMPO
      for (size_t I = 0; I < this->B->dim; ++I) {
        int pauli_phase;
        binary_state ssp;

        binary_state ss = this->B->bin(I); // binary form of state 'label'
        auto R = group->Representative(ss, this->B->sec.irrep);
        ssp = ss;
        // cout << "ijkl = (" << ei << ',' << ej << ';' << ek << ',' << el << ")\t";
        // PrintBinaryDouble(cout, ssp.b, n); cout << " -> "; // TEMPO
        pauli_phase = ssp.pair_annihilate(binary_state::mask(el, n), binary_state::mask(ek, n));
        // PrintBinaryDouble(cout, ssp.b, n); cout << " -> "; // TEMPO
        pauli_phase *= ssp.pair_create(binary_state::mask(ei, n), binary_state::mask(ej, n));
        // PrintBinaryDouble(cout, ssp.b, n); cout << "\tphase=" << pauli_phase << endl; // TEMPO
        if (pauli_phase != 0) {
          auto Rp = group->Representative(ssp, this->B->sec.irrep);
          if (pauli_phase == -1) Rp.phase += group->g;
          size_t J = this->B->index(Rp.b);
          
          // cout << "--> " << I << ", "  << J << endl; // TEMPO

          if (J < this->B->dim) {
            // finding the phase
            X = group->phaseX<double>(Rp.phase) * fold_type<HS_field, op_field>(x.v) * sqrt((1.0 * R.length) / Rp.length);
            this->insert(J, I, X);
            // if(I != J) this->insert(I, J, conjugate(X)); // Hermitian conjugate
          }
        }
      }
    }
    this->sort_elements();
  }
};

#endif
