//
//  symmetry_group.cpp
//  exact_diagonalization
//
//  Created by David Sénéchal on 16-11-08.
//  Copyright © 2016 David Sénéchal. All rights reserved.
//

#include "symmetry_group.hpp"
	
/**
 Constructor
 */
symmetry_group::symmetry_group(int _N, int _n_sites, const vector<vector<int>> &gen, bool _bath_irrep) : N(_N), n_sites(_n_sites), generator(gen), bath_irrep(_bath_irrep)
{
  bath_irrep = _bath_irrep;
  if(_bath_irrep) L = _n_sites;
  else L = N;
  
  if(global_bool("nosym") or generator.size() == 0){
    bath_irrep = false;
    generator.resize(1);
    generator[0].resize(N);
    for(int i=0; i<N; i++) generator[0][i] = i;
    build();
  }
  else build();
}


/**
 checks whether element v is a valid element
 */
bool symmetry_group::is_valid_element(const vector<int>& v, bool full){
  for(int i=0; i<L; ++i){
    if(v[i] >= L or v[i] < 0) return false;
    for(int j=0; j<i; ++j) if(v[i] == v[j]) return false;
  }
  if(full){
    for(int i=L; i<N; ++i){
    if(v[i] >= g) return false;
    }
  }
  return true;
}

bool symmetry_group::diff(const vector<int>& x, const vector<int>& y, bool full){
  for(int i=0; i<L; ++i) if(x[i] != y[i]) return true;
  if(full) for(int i=L; i<N; ++i) if(x[i] != y[i]) return true;
  return false;
}

//! checks whether this is the trivial (identity) group element
bool symmetry_group::is_identity(const vector<int>& v, bool full){
  for(int i=0; i<L; ++i) if(v[i] != i) return false;
  if(full){
    for(int i=L; i<N; ++i) if(v[i] != 0) return false;
  }	
  return true;
}


//! checks whether this is the trivial (identity) group element
vector<int> symmetry_group::identity(){
  vector<int> v(N);
  for(int i=0; i<L; ++i) v[i] = i;
  for(int i=L; i<N; ++i) v[i] = 0;
  return v;
}


//! multiplies two group elements
vector<int> symmetry_group::product(const vector<int>& x, const vector<int>& y)
{
  vector<int> v(N);
  for(int i=0; i<L; ++i) v[i] = x[y[i]];
  if(g) for(int i=L; i<N; ++i) v[i]= (x[i] + y[i])%g;
  return v;
}

//! returns the inverse of a group element
vector<int> symmetry_group::inverse(const vector<int>& x){
  vector<int> v(N);
  for(int i=0; i<L; ++i) v[x[i]] = i;
  for(int i=L; i<N; ++i) v[i]= (-x[i] + g)%g;
  return v;
}

// returns the signature of the permutation
int symmetry_group::sign(const vector<int>& x){
  int n1 = L-1;
  int s=0;
  for(int i=0; i<n1; ++i){
    for(int j=i+1; j<L; ++j) if (x[i] > x[j]) s++;
  }
  if(s & 1) return(-1);
  else return(1);
}


// applies the group element to the bits of b
uint32_t symmetry_group::apply_single(const vector<int>& x, const uint32_t b){
  int bp = 0;
  for(int i=0; i<L; ++i) if(b&(1<<i)) bp += (1<<x[i]);
  return bp;
}


/**
applies the group element to the bits of b, and calculates the phase, which includes the fermionic phase from the permutation and the group phase.
the phase is an int from 0 to 2*g that becomes the complex number exp(i*phase*pi/(2*g))
returns {updated b, phase}
*/
pair<uint32_t, int> symmetry_group::apply(const vector<int>& v, const uint32_t b){
  uint32_t bp = 0;
  uint32_t mask = 1;
  int q[32];
  int j=0;
  int phase = 0;
  
  memset(q,0,N*sizeof(*q));
  for(int i=0; i<L; i++){
    if(b & mask){
      bp += (1 << v[i]);
      q[j++] = v[i];
    }
    mask <<= 1;
  }
  for(int i=L; i<N; i++){
    if(b & mask){
      phase += 2*v[i];
      bp += mask;
    }
    mask <<= 1;
  }
  // computing the signature
  int n1 = j-1;
  int m=0;
  for(int i=0; i<n1; ++i){
    for(int k=i+1; k<j; ++k) if (q[i] > q[k]) m++;
  }
  phase += (m%2)*g; // adds g (i.e. x -1) if the permutation is odd
  return {bp, phase};
}


// applies the group element to a vector
template<typename T>
vector<T> symmetry_group::apply_to_vector(const vector<int>& v, const vector<T>& x)
{
  QCM_ASSERT(x.size()==L);
  vector<T> xp(L);
  for(int i=0; i<L; i++) xp[i] = x[v[i]];
  return xp;
}



/**
 returns the label of the orbital obtained by applying group element v to orbital 'a', along with the phase 'phase'
@param a orbital label (from 0 to 2N-1)
@param phase exponent (in multiple of 2g/(2pi)), added to the previous value
returns an orbital label between 0 and 2N-1
*/
pair<int, int> symmetry_group::map_orbital(const vector<int>& v, int a){
  if(a<L) return {v[a], 0};
  else if(a<N) return {a, 2*v[a]};
  else if(a<N+L) return {v[a-N]+N, 0};
  else return {a, 2*v[a%N]};
}





/**
 Finishes the construction of the struct following reading the generators
 */
void symmetry_group::build()
{
  //..............................................................................
  // Building and checking the generators
  
  if(generator.size() > MAX_GENERATORS){
    qcm_ED_throw("Number of group generators exceeds the limit of " + to_string(MAX_GENERATORS));
  }
  
  for(int i=0; i<(int)generator.size(); ++i){
    if(!is_valid_element(generator[i])){
      cout << "generator " << generator[i] << " problematic" << endl;
      qcm_ED_throw("generator " + to_string(i+1) + " is not valid!");
    }
  }

  //..............................................................................
  // Checking that the generators commute with each other
  
  for(int i=0; i<generator.size(); ++i){
    for(int j=0; j<i; ++j){
      auto p1 = product(generator[i], generator[j]);
      auto p2 = product(generator[j], generator[i]);
      if(diff(p1, p2)) {
        qcm_ED_throw("generators " + to_string(i+1) + " and " + to_string(j+1) + " do not commute!");
      }
    }
  }
  
  //..............................................................................
  // Lengths of the generators
  
  g=1;
  has_complex_irrep = false;
  vector<int> dim;
  dim.reserve(MAX_GENERATORS);
  for(int i=0; i<generator.size(); ++i){
    auto p = identity();
    int length=1;
    while(true){
      p = product(p, generator[i]);
      if(is_identity(p)) break;
      ++length;
    }
    dim.push_back(length);
    g *= length;
    if(length>2) has_complex_irrep = true;
  }
  Index index(dim);

  //..............................................................................
  // Checking (again) that the generators commute with each other
  
  for(int i=0; i<generator.size(); ++i){
    for(int j=0; j<i; ++j){
      auto p1 = product(generator[i], generator[j]);
      auto p2 = product(generator[j], generator[i]);
      if(diff(p1, p2, true)) {
        qcm_ED_throw("generators " + to_string(i+1) + " and " + to_string(j+1) + " do not commute!");
      }
    }
  }

  //..............................................................................
  // Constructing the elements
  
  e.reserve(g);
  for(int i=0; i<g; ++i){
    auto p = identity();
    index(i);

    for(int j=0; j<generator.size(); ++j){
      for(int l=0; l<index.ind[j]; ++l) p = product(p, generator[j]);
    }
    // Problem if this is the identity. Complain and abort
    if(is_identity(p, true) and i != 0){
      qcm_ED_throw("The identity is generated as the non trivial element " + index.str());
    }
    if(!is_valid_element(p,true)){
      cout << "element " << generator[i] << " problematic" << endl;
      qcm_ED_throw("element " + to_string(i+1) + " is not valid!");
    }
    e.push_back(p);
  }

  //..............................................................................
  // Constructing the phases of the characters and the character table
  
  phi.set_size(g, g);
  Index irrep(index);
  for(int i=0; i<g; ++i){
    irrep(i);
    for(int j=0; j<g; ++j){
      index(j);
      phi(i,j) = irrep*index;
    }
  }
  
  complex_irrep.assign(g, false);
  chi.set_size(g, g);
  for(int i=0; i<g; ++i){
    for(int j=0; j<g; ++j){
      double phase = phi(i,j)*2*M_PI/g;
      Complex z(cos(phase),sin(phase));
      chi(i,j) = chop(z,1e-12);
      if(abs(sin(phase)) > 1e-12) complex_irrep[i] = true;
    }
  }
  
  //..............................................................................
  // Constructing the transformation matrix for symmetric orbitals
  
  site_irrep_dim.resize(g);
  site_irrep_dim_cumul.resize(g);
  site_irrep_dim_cumul[0] = 0;
  S.set_size(n_sites);
  
  int n_sym_orb = 0;
  vector<Complex> orb(n_sites);
  for(int irrep = 0; irrep < g; irrep++){
    int n_sym_orb_in_irrep = 0;
    vector<bool> skip_site(n_sites);
    for(int site=0; site < n_sites; site++){ // loop over orbits (i.e. representatives)
      if(skip_site[site])	continue;  // orbit already covered for that irrep
      skip_site[site] = true;
      to_zero(orb);
      for(int i=0; i < g; i++){
        skip_site[e[i][site]] = true;
        orb[(int)e[i][site]] += chi(irrep,i);
      }
      if(normalize(orb)){
        for(int i=0 ; i<n_sites; i++) S(n_sym_orb,i) = orb[i];
        n_sym_orb ++;
        n_sym_orb_in_irrep++;
      }
    }
    site_irrep_dim[irrep] = n_sym_orb_in_irrep;
    if(irrep>0) site_irrep_dim_cumul[irrep] = site_irrep_dim_cumul[irrep-1] + n_sym_orb_in_irrep;
  }
  
  if(n_sym_orb != n_sites){
    qcm_ED_throw("Problem with construction of symmetric orbitals: "
                   + to_string(n_sym_orb) + " found instead of " + to_string(n_sites));
  }
  
  if(has_complex_irrep == false){
    S_real.set_size(n_sites);
    S_real.real_part(S);
  }
  
  //..............................................................................
  // construction of the tensor products
  tensor.set_size(g);
  for (int r1=0; r1<g; r1++) { // loop over irreps
    for (int r2=0; r2<g; r2++){ // loop over irreps
      tensor(r1,r2) = index(r1,r2); // group or representation product
    }
  }
  //..............................................................................
  // identifying the conjugate irreps
  
  conjugate.resize(g);
  //	cout << "conjugate representations:\n";
  for(int r=0; r<g; r++){
    vector<int> chi_tmp(g);
    for(int i=0; i<g; i++) chi_tmp[i] = 2*g-phi(r,i); // adding 2*g to make result positive
    for(int rp=0; rp<g; rp++){
      int i;
      for(i=0; i<g; i++) if((chi_tmp[i]-phi(rp,i)+2*g)%g) break;
      if(i==g){
        conjugate[r] = rp;
        break;
      }
    }
  }

  //..............................................................................
  // precomputing the phases
  phaseR.resize(2*g);
  phaseC.resize(2*g);
  for(int i=0; i<2*g; i++) phaseR[i] =-1; phaseR[0] = 1;
  for(int i=0; i<2*g; i++){
    double phi = (M_PI*i)/g;
    Complex z(cos(phi),sin(phi));
    phaseC[i] = z;
  }
}




/**
 Applies group element elem to the binary_state b
 @param elem group element
 @param b input binary_state state
 returns {new binary state, phase}
 the phase is the fermionic phase (+1 or -1) + phase from the signed permutations + group phase
 The transformation is applied separately on the left and right parts.
 This should be changed if we extend symmetries to those that mix the up and down spin parts.
 */
pair<binary_state, int> symmetry_group::apply(const int &elem, const binary_state &b)
{
  // first the right part of b
  auto right = apply(e[elem], b.right());
  // second the left part of b
  auto left  = apply(e[elem], b.left());
  return {binary_state(left.first, right.first), left.second+right.second};
}






/**
 Applies group element elem to the binary_state b, in representation irrep
 @param elem group element
 @param irrep irreducible representation
 @param b input binary_state state
 returns {new binary state, phase}
 the phase is the net phase coming from both the fermionic phase, the signed permutations and the projection operator (characters of the irrep)
 */
pair<binary_state, int> symmetry_group::apply(const int &elem, const int &irrep, const binary_state &b)
{
  auto X = apply(elem, b);
  X.second += 2*phi(elem, irrep);
  X.second = X.second%(2*g);
  return X;
}






/**
 Transforms the binary state b into its representative in the orbit under the group, in the representation irrep
 @param b input binary_state
 @param irrep the irreducible representation
 returns {new binary state, phase, length}
 phase : resulting phase (if multiplied by (2 pi / 2 g) )
 length : the length of the cycle, zero if the state is null (=0,1,2,4,8)

 Note: phases are in multiples of pi/g and not 2pi/g, because of groups of odd length and the necessity to
 represent -1 (from fermionic signs) in those terms.
 */
rep_map symmetry_group::Representative(const binary_state &b, int irrep)
{
  int h=1; // size of the subgroup that leaves the state invariant
  rep_map M = {b, 0, 0};

  for(int elem=1; elem<g; ++elem){ // loop over group elements to find the representative
    auto X = apply(elem, b);
    X.second += 2*phi(elem,irrep);
    X.second = X.second%(2*g);
    
    if(X.first == b){
      h++;
      if(X.second != 0 and (2*g)%X.second == 0){
        M.length = 0;
        return M;
      }
    }
    if (X.first.b < M.b.b) {
      M.b = X.first;
      M.phase = X.second;
    }
  }
  M.length = g/h;
  return M;
}





/**
 Converts a vector B in the basis of symmetric orbitals
 into a vector G in the original site basis
 @param r label of the irrep
 @param x vector expressed in the irrep r (dimension site_irrep_dim[r]*n_mixed)
 @param y vector in the larger basis (dimension n_sites*n_mixed)
 */
void symmetry_group::to_site_basis(int r, vector<Complex> &x, vector<Complex> &y, int n_mixed)
{
  to_zero(y);
  QCM_ASSERT(y.size() == n_mixed*n_sites);
  QCM_ASSERT(x.size() == n_mixed*site_irrep_dim[r]);
  int offset = site_irrep_dim_cumul[r];
  for(int mr=0; mr<n_mixed; mr++){
    int y_offset = mr*n_sites;
    for(int a=0; a<n_sites; a++){
      int dr = site_irrep_dim[r];
      Complex z(0.0);
      int x_offset = dr*mr;
      for(int alpha=0; alpha<dr; alpha++){
        z += x[alpha+x_offset]*S(alpha+offset,a);
      }
      y[a+y_offset] += z;
    }
  }
}

void symmetry_group::to_site_basis(int r, vector<double> &x, vector<double> &y, int n_mixed)
{
  to_zero(y);
  QCM_ASSERT(y.size() == n_mixed*n_sites);
  QCM_ASSERT(x.size() == n_mixed*site_irrep_dim[r]);
  int offset = site_irrep_dim_cumul[r];
  for(int mr=0; mr<n_mixed; mr++){
    int y_offset = mr*n_sites;
    for(int a=0; a<n_sites; a++){
      int dr = site_irrep_dim[r];
      double z(0.0);
      int x_offset = dr*mr;
      for(int alpha=0; alpha<dr; alpha++){
        z += x[alpha+x_offset]*S_real(alpha+offset,a);
      }
      y[a+y_offset] += z;
    }
  }
}





/**
 Checks whether a sector is valid
 output : true if valid, false otherwise
 */
bool symmetry_group::sector_is_valid(const sector& sec){
  if(sec.N < sector::odd){
    if(sec.N < 0) return false;
    if(sec.N > 2*N) return false;
    if(sec.S < sector::odd){
      if(sec.S > sec.N) return false;
      if(sec.S < -sec.N) return false;
      if((sec.N+sec.S)%2) return false;
    }
  }
  if(sec.S < sector::odd){
    int norb((int)N);
    if(sec.S > norb) return false;
    if(sec.S < -norb) return false;
  }
  if(sec.irrep >= g) return false;
  else return true;
}






// Shifts a sector by adding (pm=1) or removing (pm=-1) a particle of spin "spin" (=0 or 1) in irrep _irrep
sector symmetry_group::shift_sector(const sector& _sec, int pm, int spin, int _irrep){
  
  sector sec = _sec;
  if(sec.N == sector::odd) sec.N = sector::even;
  else if(sec.N == sector::even) sec.N = sector::odd;
  else sec.N += pm;

  if(sec.S == sector::odd) sec.S = sector::even;
  else if(sec.S == sector::even) sec.S = sector::odd;
  else{
    if(spin) sec.S -= pm;
    else sec.S += pm;
  }

  if(pm==1) sec.irrep = tensor(sec.irrep,_irrep);
  else sec.irrep = tensor(sec.irrep, conjugate[_irrep]);

  return sec;
}




bool symmetry_group::operator!=(const symmetry_group &y){
  if(N != y.N) return true;
  if(n_sites != y.n_sites) return true;
  if(generator.size() != y.generator.size()) return true;
  for(int i=0; i<generator.size(); ++i) if(generator[i] != y.generator[i]) return true;
  return false;
}





/**
 Converts a block-diagonal matrix B in the basis of symmetric orbitals
 into a full matrix G in the original site basis
 @param B the block diagonal matrix
 @param G the full matrix, unrolled into a vector
 */
void symmetry_group::to_site_basis(block_matrix<Complex> &B, matrix<Complex> &G, int n_mixed)
{
  for(int mr=0; mr<n_mixed; mr++){
    int G_offset_r = mr*n_sites;
    for(int mc=0; mc<n_mixed; mc++){
      int G_offset_c = mc*n_sites;
      for(int a=0; a<n_sites; a++){
        for(int b=0; b<n_sites; b++){
          int offset = 0;
          for(int r=0; r<g; r++){
            int dr = site_irrep_dim[r];
            Complex z(0.0);
            matrix<Complex>& sym_g = B.block[r];
            int sym_g_offset_r = dr*mr;
            int sym_g_offset_c = dr*mc;
            for(int alpha=0; alpha<dr; alpha++){
              for(int beta=0; beta<dr; beta++){
                z += sym_g(alpha+sym_g_offset_r,beta+sym_g_offset_c)*S(beta+offset,b)*conj(S(alpha+offset,a));
              }
            }
            offset += dr;
            G(a+G_offset_r,b+G_offset_c) += z;
          }
        }
      }
    }
  }
}



/**
 Building the set of symmetric orbitals
 @param mixing mixing state of the instance
 */
vector<vector<symmetric_orbital>> symmetry_group::build_symmetric_orbitals(int mixing)
{
  int control = 0;
  int nambu_max = (mixing&HS_mixing::anomalous)? 2:1;
  int spin_max = (mixing&HS_mixing::spin_flip)? 2:1;
  vector<vector<symmetric_orbital>> sym_orb(g);
  for(int r=0; r<g; r++){
    vector<symmetric_orbital> tmp;
    tmp.reserve(nambu_max*spin_max*site_irrep_dim[r]);
    for(int i_nambu = 0; i_nambu < nambu_max; i_nambu++){
      for(int i_spin = 0; i_spin < spin_max; i_spin++){
        for(int i=0; i< site_irrep_dim[r]; i++){
          if(i_nambu){
            symmetric_orbital tmp_orb;
            if(spin_max==2) tmp_orb = symmetric_orbital(conjugate[r], i, i_spin, i_nambu, site_irrep_dim);
            else tmp_orb = symmetric_orbital(conjugate[r], i, (i_spin+1)%2, i_nambu, site_irrep_dim);
            tmp.push_back(tmp_orb);
          }
          else{
            symmetric_orbital tmp_orb(r, i, i_spin, i_nambu, site_irrep_dim);
            tmp.push_back(tmp_orb);
          }
          control++;
        }
      }
    }
    sym_orb[r] = tmp;
  }
  return sym_orb;
}




std::ostream & operator<<(std::ostream &flux, const symmetry_group &x)
{
  flux << "generators\n";
  for(int i=0; i<x.generator.size(); i++) flux << i+1 << " : " << x.generator[i] << '\n';
  flux << "\ngroup elements\n";
  for(int i=0; i<x.g; i++) flux << i+1 << " : " << x.e[i] << '\n';
  flux << "\nSymmetric orbitals:\n" << x.S << '\n';
  flux << "\nphases according to the different irreps:\n" << x.phi << endl;
  flux << "\nphaseR:\n" << x.phaseR << endl;
  flux << "\nphaseC:\n" << x.phaseC << endl;
  return flux;
}