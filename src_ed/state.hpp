#ifndef state_h
#define state_h

#include <cstdio>

#include "sector.hpp"
#include "Q_matrix_set.hpp"
#include "continued_fraction_set.hpp"
#include "parser.hpp"
#include "ED_basis.hpp"

//! state (e.g. the ground state) from which a contribution to the Green function is computed
template<typename HilbertField>
struct state
{
	sector sec; //!< sector to which the state belongs
	vector<HilbertField> psi; 	//!< the Hilbert-space vector representing the state
	double energy; //!< energy of the state
	double weight; //!< its weight in the density matrix
	
  shared_ptr<Green_function_set> gf;
  shared_ptr<Green_function_set> gf_down;
  
  /**
   default constructor
   */
  state() : sec("R0") {}

  
  
  /**
	 constructor from sector and dimension
	 */
	state(sector _sec, size_t dim) : sec(_sec)
	{
		psi.resize(dim);
	}
  
  
  /**
   constructor from ASCII file
   */
  state(istream& fin, shared_ptr<symmetry_group> group , int mixing, GF_FORMAT GF_solver)
  {
    vector<string> input = read_strings(fin);
    if(input.size()!=3) qcm_ED_throw("failed to read a state from input file. Need sector, energy and weight in header line");
    sec = sector(input[0]);
    energy = from_string<double>(input[1]);
    weight = from_string<double>(input[2]);
    if(GF_solver == GF_format_BL){
      try{gf = make_shared<Q_matrix_set<HilbertField>>(fin, group, mixing);}
      catch(const string &s) {qcm_ED_catch(s);}
    }
    else if(GF_solver == GF_format_CF){
      try{gf = shared_ptr<continued_fraction_set>(new continued_fraction_set(fin, sec, group, mixing, typeid(HilbertField) == typeid(Complex)));}
      catch(const string &s) {qcm_ED_catch(s);}
    }
    else qcm_ED_throw("unkown Green function solver (GF_solver)");
    if(mixing&HS_mixing::up_down){
      if(GF_solver == GF_format_BL){
        try{gf_down = shared_ptr<Q_matrix_set<HilbertField>>(new Q_matrix_set<HilbertField>(fin, group, mixing));}
        catch(const string &s) {qcm_ED_catch(s);}
      }
      else if(GF_solver == GF_format_CF){
        try{gf_down = shared_ptr<continued_fraction_set>(new continued_fraction_set(fin, sec, group, mixing, typeid(HilbertField) == typeid(Complex)));}
        catch(const string &s) {qcm_ED_catch(s);}
      }
    }
  }
  


  /**
   writing the Green function representation to an ASCII file
   */
  void write(ostream& fout)
  {
    fout << "state\n" << sec << '\t' << energy << '\t' << weight << endl;
    if(gf != nullptr) gf->write(fout);
    if(gf_down != nullptr) gf_down->write(fout);
  }


  /**
   writing the wavefunction to an ASCII file
   */
  void write_wavefunction(ostream& fout, const ED_basis &B)
  {
    fout << "state\n" << sec << '\t' << energy << '\t' << weight << endl;
    if(B.dim <= global_int("max_dim_print")){
      for(int i=0; i<B.dim; i++){
        fout << abs(psi[i])*abs(psi[i]) << '\t' << psi[i] << '\t';
        B.print_state(fout,i);
        fout << '\n';
      }
    }
  }


};

template<typename HilbertField>
std::ostream& operator<<(std::ostream &flux, const state<HilbertField> &x)
{
  flux << "E = " << x.energy << " (" << x.sec << ") weight = " << x.weight << endl;
  return flux;
}



#endif

