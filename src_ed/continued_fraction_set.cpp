#include "continued_fraction_set.hpp"

/**
 Constructor
 */
continued_fraction_set::continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, bool _is_complex) :
Green_function_set(_group, mixing), sec(_sec), is_complex(_is_complex)
{
  e.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  h.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  for(size_t r=0; r<group->site_irrep_dim.size(); ++r){
    size_t n = group->site_irrep_dim[r];
    if(sec.S >= sector::odd) n *= 2;
    if(sec.N >= sector::odd) n *= 2;
    e[r].set_size(n);
    h[r].set_size(n);
  }
}

/**
 Constructor from arrays
 */
continued_fraction_set::continued_fraction_set(sector _sec, shared_ptr<symmetry_group> _group, int mixing, const vector<vector<double>> &A, const vector<vector<double>> &B, bool _is_complex) :
Green_function_set(_group, mixing), sec(_sec), is_complex(_is_complex)
{
  QCM_ASSERT(group->site_irrep_dim.size() == 2*A.size());
  QCM_ASSERT(group->site_irrep_dim.size() == 2*B.size());
  e.resize(group->site_irrep_dim.size());
  h.resize(group->site_irrep_dim.size());
  for(size_t r=0; r<group->site_irrep_dim.size(); ++r){
    size_t n = group->site_irrep_dim[r];
    if(sec.S >= sector::odd) n *= 2;
    if(sec.N >= sector::odd) n *= 2;
    e[r].set_size(n);
    h[r].set_size(n);
  }
  int j=0;
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a=0; a<nr; ++a){
      int bmax = (is_complex)? nr : a+1;
      for(size_t b=0; b<bmax; ++b){
        e[r](a, b) = continued_fraction(A[j], B[j]);
        j++;
        h[r](a, b) = continued_fraction(A[j], B[j]);
        j++;
      }
    }
  }

}

/**
 Constructor from input stream (ASCII file)
 */
continued_fraction_set::continued_fraction_set(istream &fin, sector _sec, shared_ptr<symmetry_group> _group, int mixing, bool _is_complex) :
Green_function_set(_group, mixing), sec(_sec), is_complex(_is_complex)
{
  e.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  h.assign(group->site_irrep_dim.size(), matrix<continued_fraction>());
  for(size_t r=0; r<group->site_irrep_dim.size(); ++r){
    size_t n = group->site_irrep_dim[r];
    if(sec.S >= sector::odd) n *= 2;
    if(sec.N >= sector::odd) n *= 2;
    e[r].set_size(n);
    h[r].set_size(n);
  }
  string tmp;
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a=0; a<nr; ++a){
      int bmax = (is_complex)? nr : a+1;
      for(size_t b=0; b<bmax; ++b){
        size_t r2, a2, b2;
        fin >> tmp >> r2 >> tmp >> a2 >> tmp >> b2;
        if(r2 != r or a2 != a or b2 != b) qcm_ED_throw("error while reading continued fraction");
        fin >> e[r](a,b) >> h[r](a,b);
      }
    }
  }
}

/**
write in a ASCII file
 @param flux : output flux
 */
void continued_fraction_set::write(ostream& flux)
{
  
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    for(size_t a=0; a<nr; ++a){
      int bmax = (is_complex)? nr : a+1;
      for(size_t b=0; b<bmax; ++b){
        flux << "irrep: " << r << "\ta: " << a << "\tb: " << b << "\n";
        flux << e[r](a,b) << h[r](a,b);
      }
    }
  }
}


/**
 constructs the cluster Green function matrix \a G at frequency \a z
 @param z complex frequency
 @param G Green function matrix (added to)
 */
void continued_fraction_set::Green_function(const Complex &z, block_matrix<Complex> &G)
{
  Complex v;
  
  // loop over irreps
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    if(nr==0) continue;
    vector<Complex> Gtmp(nr);
    
    // diagonal terms (direct frequency)
    for(size_t o1 = 0; o1 < nr; o1++){
      Gtmp[o1] = e[r](o1,o1).evaluate(z) + h[r](o1,o1).evaluate(z);
      G.block[r](o1,o1) += Gtmp[o1];
    }
    
    // off diagonal terms
    if(is_complex){
      for(size_t o1 = 0; o1 < nr; o1++){
        for(size_t o2 = 0; o2 <o1; o2++){
          Complex tmp1 = e[r](o1,o2).evaluate(z) +  h[r](o1,o2).evaluate(z);
          Complex tmp2 = Complex(0,1)*(e[r](o2,o1).evaluate(z) +  h[r](o2,o1).evaluate(z));
          G.block[r](o2,o1) += 0.5*(tmp1 + tmp2 - Complex(1, 1)*(Gtmp[o1] + Gtmp[o2]));
          G.block[r](o1,o2) += 0.5*(tmp1 - tmp2 - Complex(1,-1)*(Gtmp[o1] + Gtmp[o2]));
        }
      }
    }
    else{
      for(size_t o1 = 0; o1 < nr; o1++){
        for(size_t o2 = 0; o2 <o1; o2++){
          Complex tmp = 0.5*(e[r](o1,o2).evaluate(z) + h[r](o1,o2).evaluate(z) - Gtmp[o1] - Gtmp[o2]);
          G.block[r](o1,o2) += tmp;
          G.block[r](o2,o1) += tmp;
        }
      }
    }
  }
}




/**
 frequency-integrated Green function
 @param [out] G integrated Green function (adds to previous value)
 */
void continued_fraction_set::integrated_Green_function(block_matrix<Complex> &G)
{
  Complex v;
  
  // loop over irreps
  for(size_t r=0; r<e.size(); ++r){
    size_t nr = e[r].r;
    if(nr==0) continue;
    vector<Complex> Gtmp(nr);
    
    // diagonal terms (direct frequency)
    for(size_t o1 = 0; o1 < nr; o1++){
      G.block[r](o1,o1) += h[r](o1,o1).b[0];
    }
    
    // off diagonal terms
    for(size_t o1 = 0; o1 < nr; o1++){
      for(size_t o2 = 0; o2 <o1; o2++){
        Complex tmp = 0.5*(h[r](o1,o2).b[0] - G.block[r](o1,o1) - G.block[r](o2,o2));
        G.block[r](o1,o2) += tmp;
        G.block[r](o2,o1) += tmp;
      }
    }
  }
}
