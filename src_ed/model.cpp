#include "model.hpp"
#include "Operators/Hermitian_operator.hpp"
#include <fstream>
#ifdef _OPENMP
  #include <omp.h>
#endif

/**
 Constructor
 @param _name   name of system (cluster). Used to distinguish different cluster shapes within a larger calculation
 @param _n_orb	number of orbitals (bath included) in the model
 @param _n_bath	number of bath orbitals in the model
 @param gen sequence of generators (permutations of sites)
  */

model::model(const string &_name, const size_t _n_orb, const size_t _n_bath, const vector<vector<int>> &gen, bool bath_irrep)
: name(_name),
n_orb(_n_orb),
n_bath(_n_bath),
n_sites(_n_orb-_n_bath),
is_closed(false),
is_factorized(false),
mixing(-1)
{
  group = make_shared<symmetry_group>(n_orb, n_sites, gen, bath_irrep);
  in_bath.assign(2*n_orb, false);
  for(size_t i=n_sites; i<n_orb; i++){
    in_bath[i] = true;
    in_bath[i+n_orb] = true;
  }

  // builds symmetric orbitals for all mixings:
  sym_orb.reserve(6);
  for(int m=0; m<6; m++) sym_orb.push_back(group->build_symmetric_orbitals(m));
}







/**
 Prints the definition of the model into a stream
 */
void model::print(ostream& fout)
{
  banner('-', "cluster " + name, fout);
  fout << *group << endl;
  for(auto& x: term) x.second->print(fout);
}



/**
 returns the basis required for the sector sec. Builds it if needed.
 @param sec sector of the state 
*/
shared_ptr<ED_mixed_basis> model::provide_basis(const sector& sec)
{
  if(basis.find(sec) == basis.end()) basis[sec] = make_shared<ED_mixed_basis>(sec, group);
  return basis[sec];
}




/**
 returns the basis required for the sector sec. Builds it if needed.
 @param sec sector of the state 
*/
shared_ptr<ED_factorized_basis> model::provide_factorized_basis(const sector& sec)
{
  if(factorized_basis.find(sec) == factorized_basis.end()) 
    factorized_basis[sec] = make_shared<ED_factorized_basis>(sec, n_orb);
  return factorized_basis[sec];
}



/**
 Builds the operators necessary in a given sector
 @param GS_sector sector of the state 
 @param complex true if the HS operators must be complex-valued
*/
void model::build_HS_operators(const sector& sec, bool is_complex)
{
  vector<shared_ptr<Hermitian_operator>> keys;
  keys.reserve(term.size());
  for (auto& x : term) {
      keys.push_back(x.second);
  }
  //#pragma omp parallel for schedule(dynamic,1)
  for(auto& x : keys){
    if(!x->is_active) continue;
    if(x->HS_operator.find(sec) == x->HS_operator.end())
      cout << "building operator " << x->name << endl;
      x->HS_operator[sec] = x->build_HS_operator(sec, is_complex);
  }
}




/**
Prints a graph representation of the model, using the dot language.
The fixed positions of the cluster sites per se (not the bath) are provided in 'pos'
*/
void model::print_graph(const vector<vector<double>> &pos){
  ofstream fout(name+".dot");

  fout << "graph {\nK=1.3;\n";
  for(int i=0; i<pos.size(); i++){
    fout << i+1 << " [shape=square color=\"blue\" pos=\"" << pos[i][0] << "," << pos[i][1] << "!\"]\n";
  }
  for(int i=pos.size(); i<n_orb; i++){
    fout << i+1 << " [shape=circle color=\"red\"]\n";
  }
  for(auto& x : term){
    if(x.second->is_interaction) continue;
    auto elem = x.second->matrix_elements();
    for(auto& e : elem){
      if(e.r >= n_orb or e.c >= n_orb) continue;
      if(e.r >= e.c) continue;
      string lab = x.second->name;
      if(abs(e.v - 1.0)<1e-4) lab = lab;
      else if(abs(e.v + 1.0)<1e-4) lab = '-'+lab;
      else lab = to_string(e.v)+lab;
      if(e.c >= n_sites) fout << e.r+1 << " -- " << e.c+1 << " [color = \"green\" headlabel=\"" << lab << "\" labeldistance=3 labelangle=0 fontsize=10 fontcolor=\"#990000\"];\n";
      else fout << e.r+1 << " -- " << e.c+1 << " [label=\"" << lab << "\" fontsize=10];\n";
    }
  }
  fout << "}\n";
  fout.close();
}



/**
 Applies a creation or annihilation operator
 |y> += z * c_a |x>  ou  |y> += z * c^+_a |x>
 @param pm		if = 1: creation; if = -1: destruction.
 @param a		symmetric orbital
 @param x		in state
 @param y		out state (allocated before)
 @param z		coefficient of the out state
 */
bool model::create_or_destroy(int pm, const symmetric_orbital &a, state<double> &x, vector<double> &y, double z)
{
  if(a.nambu) pm = -pm;
  sector target_sec = group->shift_sector(x.sec, pm, a.spin, a.irrep);

  if(is_factorized){
    auto B = provide_factorized_basis(x.sec);
    auto T = provide_factorized_basis(target_sec);
    if(y.size() != T->dim) y.resize(T->dim);
    if(pm==1){ // creation
      for(uint32_t I=0; I<T->dim; ++I){
        auto P = Destroy(a.orb+n_orb*a.spin, I, *T, *B);
        if(!get<3>(P)) continue;
        get<1>(P) = 1-2*(get<1>(P)%2);
        y[I] += z*get<1>(P)*x.psi[get<0>(P)];
      }
    }
    else{ // destruction
      for(uint32_t I=0; I<B->dim; ++I){
        auto P = Destroy(a.orb+n_orb*a.spin, I, *B, *T);
        if(!get<3>(P)) continue;
        get<1>(P) = 1-2*(get<1>(P)%2);
        y[get<0>(P)] += z*get<1>(P)*x.psi[I];
      }
    }
  }
  else{
    auto B = provide_basis(x.sec);
    sector target_sec = group->shift_sector(x.sec, pm, a.spin, a.irrep);
    auto T = provide_basis(target_sec);
    if(y.size() != T->dim) y.resize(T->dim);

    if(pm==1){ // creation
      if(Hamiltonian_format == H_FORMAT::H_format_onthefly){
        for(uint32_t I=0; I<T->dim; ++I){
          auto P = Destroy(a.orb+n_orb*a.spin, I, *T, *B);
          if(!get<3>(P)) continue;
          get<1>(P) = get<1>(P)%(2*group->g);
          y[I] += z*group->phaseX<double>(get<1>(P))*x.psi[get<0>(P)];
        }
      }
      else{
        destruction_identifier D(target_sec,a);
        if(destruction.find(D)==destruction.end()){
          destruction[D] = make_shared<destruction_operator<double>>(T, B, D.sorb);
        }
        destruction.at(D)->multiply_add_conjugate(x.psi,y,z);
      }
      // cout << "creation at " << a.label << '\n' << x.psi << '\n' << y << "\n\n"; // tempo
    }
    else{ // destruction
      if(Hamiltonian_format == H_FORMAT::H_format_onthefly){
        for(uint32_t I=0; I<B->dim; ++I){
          auto P = Destroy(a.orb+n_orb*a.spin, I, *B, *T);
          if(!get<3>(P)) continue;
          get<1>(P) = get<1>(P)%(2*group->g);
          y[get<0>(P)] += z*group->phaseX<double>(get<1>(P))*x.psi[I];
        }
      }
      else{
        destruction_identifier D(x.sec,a);
        if(destruction.find(D)==destruction.end()){
          destruction[D] = make_shared<destruction_operator<double>>(B, T, D.sorb);
        }
        destruction.at(D)->multiply_add(x.psi,y,z);
      }
    }
  }
  return true;
}






/**
 Applies a creation or annihilation operator
 |y> += z * c_a |x>  ou  |y> += z * c^+_a |x>
 @param pm		if = 1: creation; if = -1: destruction.
 @param a		symmetric orbital
 @param x		in state
 @param y		out state (allocated before)
 @param z		coefficient of the out state
 */
bool model::create_or_destroy(int pm, const symmetric_orbital &a, state<Complex> &x, vector<Complex> &y, Complex z)
{
  if(a.nambu) pm = -pm;
  sector target_sec = group->shift_sector(x.sec, pm, a.spin, a.irrep);

  if(is_factorized){
    auto B = provide_factorized_basis(x.sec);
    auto T = provide_factorized_basis(target_sec);
    if(y.size() != T->dim) y.resize(T->dim);
    if(pm==1){ // creation
      for(uint32_t I=0; I<T->dim; ++I){
        auto P = Destroy(a.orb+n_orb*a.spin, I, *T, *B);
        if(!get<3>(P)) continue;
        get<1>(P) = get<1>(P)%(2*group->g);
        y[I] += z*group->phaseX<Complex>(get<1>(P))*x.psi[get<0>(P)];
      }
    }
    else{ // destruction
      for(uint32_t I=0; I<B->dim; ++I){
        auto P = Destroy(a.orb+n_orb*a.spin, I, *B, *T);
        if(!get<3>(P)) continue;
        get<1>(P) = get<1>(P)%(2*group->g);
        y[get<0>(P)] += z*group->phaseX<Complex>(get<1>(P))*x.psi[I];
      }
    }
  }
  else{
    auto B = provide_basis(x.sec);
    sector target_sec = group->shift_sector(x.sec, pm, a.spin, a.irrep);
    auto T = provide_basis(target_sec);
    if(T->dim == 0) return false;
    if(y.size() != T->dim) y.resize(T->dim);

    if(pm==1){ // creation
      if(Hamiltonian_format == H_FORMAT::H_format_onthefly){
        for(uint32_t I=0; I<T->dim; ++I){
          auto P = Destroy(a.orb+n_orb*a.spin, I, *T, *B);
          if(!get<3>(P)) continue;
          get<1>(P) = get<1>(P)%(2*group->g);
          y[I] += z*group->phaseX<Complex>(get<1>(P))*x.psi[get<0>(P)];
        }
      }
      else{
        destruction_identifier D(target_sec,a);
        if(destruction_complex.find(D)==destruction_complex.end()){
          destruction_complex[D] = make_shared<destruction_operator<Complex>>(T, B, D.sorb);
        }
        destruction_complex.at(D)->multiply_add_conjugate(x.psi,y,z);
      }
      // cout << "creation at " << a.label << '\n' << x.psi << '\n' << y << "\n\n"; // tempo
    }
    else{ // destruction
      if(Hamiltonian_format == H_FORMAT::H_format_onthefly){
        for(uint32_t I=0; I<B->dim; ++I){
          auto P = Destroy(a.orb+n_orb*a.spin, I, *B, *T);
          if(!get<3>(P)) continue;
          get<1>(P) = get<1>(P)%(2*group->g);
          y[get<0>(P)] += z*group->phaseX<Complex>(get<1>(P))*x.psi[I];
        }
      }
      else{
        destruction_identifier D(x.sec,a);
        if(destruction_complex.find(D)==destruction_complex.end()){
          destruction_complex[D] = make_shared<destruction_operator<Complex>>(B, T, D.sorb);
        }
        destruction_complex.at(D)->multiply_add(x.psi,y,z);
      }
      // cout << "destruction at " << a.label << '\n' << x.psi << '\n' << y << "\n\n"; // tempo
    }
  }
  return true;
}

