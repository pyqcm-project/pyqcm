#ifdef _OPENMP
  #include <omp.h>
#endif
#include "model_instance.hpp"
#include "matrix.hpp"


template<>
matrix<Complex> model_instance<double>::hopping_matrix(bool spin_down)
{
  return ((mixing&HS_mixing::up_down and spin_down) ? to_complex_matrix(tc_down) : to_complex_matrix(tc));
}
template<>
matrix<Complex> model_instance<Complex>::hopping_matrix(bool spin_down)
{
  return ((mixing&HS_mixing::up_down and spin_down) ? tc_down : tc);
}




/**
 Constructs the continued fraction representation from the Lanczos method
 or full diagonalization if the dimension is small enough.
 
 @param Omega	state on which the contribution to the Green function is built
 @param spin_down		true if we are building the spin down component of the GF
 */

template<>
void model_instance<double>::build_cf(state<double> &Omega, bool spin_down)
{
  auto& sym_orb = the_model->sym_orb[mixing];
  auto cf = make_shared<continued_fraction_set>(Omega.sec, the_model->group, mixing, false);
  if(spin_down) Omega.gf_down = cf;
  else Omega.gf = cf;
  
  for(size_t r=0; r<sym_orb.size(); r++){
    for(int pm = -1; pm<2; pm += 2){ // loop over destruction (pm = -1) and creation (pm = +1)
                                     // constructing the target sector
      sector target_sec = the_model->group->shift_sector(Omega.sec, pm, spin_down, r);
      if(!the_model->group->sector_is_valid(target_sec)) continue; // target sector is null
      
      // Assembling the Hamiltonian
      Hamiltonian<double>* H = create_hamiltonian(the_model, value, target_sec);       
      vector<symmetric_orbital>& sorb = sym_orb[r];
      if(H->B->dim==0) continue;
      
      // building the list of pairs
      vector<pair<size_t,size_t> > sorb_pair;
      sorb_pair.reserve(sorb.size()*sorb.size());
      
      for(size_t o1=0; o1< sorb.size(); o1++){ // double loop over symmetric operators
        for(size_t o2=0; o2<=o1; o2++){
          sorb_pair.push_back(pair<size_t,size_t>(o1,o2));
        }
      }

      #pragma omp parallel for schedule(dynamic,1)
      for(size_t s=0; s< sorb_pair.size(); s++){ // double loop over symmetric operators
        size_t o1 = sorb_pair[s].first;
        size_t o2 = sorb_pair[s].second;
        vector<double> psi(H->B->dim);
        double norm;
        
        if(global_bool("verb_ED")) cout << "element " << sorb[o1].str() << " , " << sorb[o2].str() << endl;
        
        symmetric_orbital sorb1 = sym_orb[r][o1];
        symmetric_orbital sorb2 = sym_orb[r][o2];
        if(spin_down) sorb1.spin = sorb2.spin = 1;
        if(!the_model->create_or_destroy(pm, sorb1, Omega, psi, 1.0)) continue;
        if(o1 != o2){
          the_model->create_or_destroy(pm, sorb2, Omega, psi, 1.0);
        }
        
        // normalisation of |x> and storing its norm in "norm"
        norm = norm2(psi);
        if(norm > 1e-14){
          psi *= 1.0/sqrt(norm);
          
          // tridiagonalisation and calculation of the projection
          pair<vector<double>, vector<double>> V = LanczosGreen(*H, psi);
          continued_fraction cont_fraction(V.first, V.second, Omega.energy, norm*Omega.weight, pm==1);
          if(pm == 1){
            cf->e[r](o1,o2) = cont_fraction;
            // if(global_bool("verb_ED")){
            //   cout << "coefficients of the electron continued fraction:\n";
            //   cout << "alphas\tbetas\n";
            //   cout << cont_fraction; 
            // }
          }
          else{
            cf->h[r](o1,o2) = cont_fraction;
            // if(global_bool("verb_ED")){
            //   cout << "coefficients of the hole continued fraction:\n";
            //   cout << "alphas\tbetas\n";
            //   cout << cont_fraction; 
            // }
          }
        }
      }
      delete H;
    }
  }
}


template<>
void model_instance<Complex>::build_cf(state<Complex> &Omega, bool spin_down)
{
  // qcm_throw("the use of continued fractions with complex-valued Hamiltonians is not yet ready");
  auto& sym_orb = the_model->sym_orb[mixing];

  auto cf = make_shared<continued_fraction_set>(Omega.sec, the_model->group, mixing, true);
  if(spin_down) Omega.gf_down = cf;
  else Omega.gf = cf;
  
  for(size_t r=0; r<sym_orb.size(); r++){
    // loop over destruction (pm = -1) and creation (pm = +1)
    for(int pm = -1; pm<2; pm += 2){ 
      // constructing the target sector
      sector target_sec = the_model->group->shift_sector(Omega.sec, pm, spin_down, r);
      if(!the_model->group->sector_is_valid(target_sec)) continue; // target sector is null
      
      // Assembling the Hamiltonian
      Hamiltonian<Complex> *H = create_hamiltonian(the_model, value, target_sec);
      if(H->B->dim==0) continue;
      
      vector<symmetric_orbital>& sorb = sym_orb[r];
      
      // building the list of pairs (for easier parallelization)
      vector<pair<size_t,size_t> > sorb_pair;
      sorb_pair.reserve(sorb.size()*sorb.size());
      
      for(size_t o1=0; o1< sorb.size(); o1++){ // double loop over symmetric operators
        for(size_t o2=0; o2< sorb.size(); o2++){
          sorb_pair.push_back(pair<size_t,size_t>(o1,o2));
        }
      }

      #pragma omp parallel for schedule(dynamic,1) // TEMPO
      for(size_t s=0; s< sorb_pair.size(); s++){ // single loop over symmetric operators pairs
        size_t o1 = sorb_pair[s].first;
        size_t o2 = sorb_pair[s].second;
        vector<Complex> psi(H->B->dim);
        double norm;
        
        if(global_bool("verb_ED")) cout << "element " << sorb[o1].str() << " , " << sorb[o2].str() << endl;
        
        symmetric_orbital sorb1 = sym_orb[r][o1];
        symmetric_orbital sorb2 = sym_orb[r][o2];
        if(spin_down) sorb1.spin = sorb2.spin = 1;
        if(!the_model->create_or_destroy(pm, sorb1, Omega, psi, Complex(1.0))) continue;
        if(o1 > o2){
          the_model->create_or_destroy(pm, sorb2, Omega, psi, Complex(1.0,0.0));
        }
        else if(o1 < o2){
          if(pm==1) the_model->create_or_destroy(pm, sorb2, Omega, psi, Complex(0.0,1.0));
          else the_model->create_or_destroy(pm, sorb2, Omega, psi, Complex(0.0,-1.0));
        }
        
        // normalisation of |x> and storing its norm in "norm"
        norm = norm2(psi);
        if(norm > 1e-14){
          psi *= 1.0/sqrt(norm);
          
          // tridiagonalisation and calculation of the projection
          pair<vector<double>, vector<double>> V = LanczosGreen(*H, psi);
          continued_fraction cont_fraction(V.first, V.second, Omega.energy,norm*Omega.weight,pm==1);
          if(pm == 1){
            cf->e[r](o1,o2) = cont_fraction;
            // if(global_bool("verb_ED")){
            //   cout << "coefficients of the electron continued fraction:\n";
            //   cout << "alphas\tbetas\n";
            //   cout << cont_fraction; 
            // }
          }
          else{
            cf->h[r](o1,o2) = cont_fraction;
            // if(global_bool("verb_ED")){
            //   cout << "coefficients of the hole continued fraction:\n";
            //   cout << "alphas\tbetas\n";
            //   cout << cont_fraction; 
            // }
          }
        }
      }
      delete H;
    }
  }
}






/**
 Performs a polynomial fit of a series of points
 CF Numerical Recipes p. 119 , Neville algorithm
 */
void polynomial_fit(
                    vector<double> &xa, //!< array of abcissas
                    vector<double> &ya, //!< array of values
                    const double x,	//!< projected abcissa
                    double &y,	//!< requested extrapolated value
                    double &dy	//!< requested error on extrapolated value
){
  size_t i,m,ns=0;
  double den,dif,dift,ho,hp,w;
  
  size_t n=xa.size();
  vector<double> c(n,0.0), d(n,0.0);
  dif=fabs(x-xa[0]);
  for (i=0;i<n;++i) {
    if ((dift=fabs(x-xa[i])) < dif) {
      ns=i;
      dif=dift;
    }
    c[i]=ya[i];
    d[i]=ya[i];
  }
  y=ya[ns--];
  for (m=1;m<n;++m) {
    for (i=0;i<n-m;++i) {
      ho=xa[i]-x;
      hp=xa[i+m]-x;
      w=c[i+1]-d[i];
      if ((den=ho-hp) == 0.0) qcm_ED_throw("Error in routine polint");
      den=w/den;
      d[i]=hp*den;
      c[i]=ho*den;
    }
    y += (dy=(2*(ns+1) < (n-m) ? c[ns+1] : d[ns--]));
  }
}




