#ifndef Q_matrix_set_h
#define Q_matrix_set_h

#include "Green_function_set.hpp"
#include "Q_matrix.hpp"

//!  Used to store the Lehman representation of a Green function
template<typename HilbertField>
struct Q_matrix_set : Green_function_set
{
  vector<Q_matrix<HilbertField>> q; //!< QMatrices

  Q_matrix_set(const Q_matrix_set &Q);
  Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing);
  Q_matrix_set(istream& fin, shared_ptr<symmetry_group> _group, int mixing);
  Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>>& w, const vector<matrix<HilbertField>> &_q);
  Q_matrix<HilbertField> consolidated_qmatrix();
  void append(Q_matrix_set &Q);
  void check_norm(double threshold, double norm);
  void streamline(bool verb=false);
  void merge(vector<multimap<double, vector<HilbertField>>>& M);

  // realizations of base class virtual methods
  void Green_function(const Complex &z, block_matrix<Complex> &G);
  void integrated_Green_function(block_matrix<Complex> &M);
  void write(ostream& fout);
};


//==============================================================================
// implementation


/**
 Constructor
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing) : Green_function_set(_group, mixing)
{
  q.assign(group->g, Q_matrix<HilbertField>());
}






/**
 deep copy Constructor
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(const Q_matrix_set &Q)
{
  L = Q.L;
  group = Q.group;
  q = Q.q;
}



/**
 constructor from input stream (ASCII file)
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(istream& fin, shared_ptr<symmetry_group> _group, int mixing) : Green_function_set(_group, mixing)
{
  group = _group;
  q.resize(group->g);
  for(int i=0; i<group->g; i++) q[i] = Q_matrix<HilbertField>(fin);
}


/**
 constructor from data from a previous computation
 */
template<typename HilbertField>
Q_matrix_set<HilbertField>::Q_matrix_set(shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>>& _e, const vector<matrix<HilbertField>> &_q) : Green_function_set(_group, mixing)
{
  group = _group;
  q.resize(group->g);
  for(int i=0; i<group->g; i++) q[i] = Q_matrix<HilbertField>(_e[i], _q[i]);
}

/**
 partial Green function evaluation
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::Green_function(const Complex &z, block_matrix<Complex> &G)
{
  for(size_t r=0; r<q.size(); ++r){
    if(q[r].L == 0) continue;
    q[r].Green_function(z,G.block[r]);
  }
}


/**
 partial integrated Green function evaluation
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::integrated_Green_function(block_matrix<Complex> &G)
{
  for(size_t r=0; r<q.size(); ++r){
    if(q[r].L == 0) continue;
    q[r].integrated_Green_function(G.block[r]);
  }
}


/**
 Appends to the Q_matrix another one (increasing the number of rows for the same number of columns
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::append(Q_matrix_set &Q)
{
  QCM_ASSERT(Q.q.size() == q.size());
  for(size_t r=0; r<q.size(); ++r) q[r]->append(*Q.q[r]);
}





/**
 Printing
 */
template<typename HilbertField>
std::ostream& operator<<(std::ostream &flux, const Q_matrix_set<HilbertField> &Q)
{
  
  flux << "\nQMatrix:\n" << std::setprecision(LONG_DISPLAY);
  for(size_t r=0; r<Q.q.size(); ++r){
    flux << "\nPart " << r+1 << '\n';
    flux << Q.q[r];
  }
  return flux;
}






/**
 Checking normalization
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::check_norm(double threshold, double norm)
{
  for(size_t r=0; r<q.size(); ++r) q[r]->check_norm(threshold, norm);
}






/**
 Eliminating the small contributions
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::streamline(bool verb)
{
  for(size_t r=0; r<q.size(); ++r){
    q[r].streamline(verb);
  }
}






/**
 Puts together the elements into a single Q_matrix, in the cluster site basis (i.e. not symmetric operators)
 */
template<typename HilbertField>
Q_matrix<HilbertField> Q_matrix_set<HilbertField>::consolidated_qmatrix()
{
  size_t i_cum = 0, i_tot = 0;
  
  for(size_t r=0; r<q.size(); ++r) i_tot += q[r].M;
  Q_matrix<HilbertField> Q(L, i_tot);
  
  vector<HilbertField> Y(L);
  for(size_t r=0; r<q.size(); ++r){
    vector<HilbertField> X(q[r].L);
    size_t imax = q[r].M;
    for(size_t i=0; i<imax; ++i){
      q[r].v.extract_column(i,X);
      group->to_site_basis(r, X, Y, n_mixed);
      Q.v.insert_column(i+i_cum, Y);
      Q.e[i+i_cum] = q[r].e[i];
    }
    i_cum += imax;
  }
  return Q;
}




/**
 Printing to ASCII file
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::write(ostream&fout)
{
  fout << setprecision((int)global_int("print_precision"));
  for(auto& x : q) fout << x;
}


/**
Merging into a multimap
 */
template<typename HilbertField>
void Q_matrix_set<HilbertField>::merge(vector<multimap<double, vector<HilbertField>>>& M) 
{
  for(size_t r=0; r<q.size(); ++r){
    for(int j=0; j< q[r].M; j++){
      vector<HilbertField> v(q[r].L);
      for(int k=0; k<q[r].L; k++) v[k] = q[r].v(k,j);
      M[r].insert({q[r].e[j], v});
    }
  }
}

#endif /* Q_matrix_set_h */
