#include <cstring>
#include <fstream>
// #include<unordered_map>
#include<map>

#include "global_parameter.hpp"
#include "parser.hpp"

bool is_global_parameter_initialized = false;

// unordered_map<string, global_parameter<bool>> GP_bool;
// unordered_map<string, global_parameter<size_t>> GP_int;
// unordered_map<string, global_parameter<double>> GP_double;
// unordered_map<string, global_parameter<char>> GP_char;
map<string, global_parameter<bool>> GP_bool;
map<string, global_parameter<size_t>> GP_int;
map<string, global_parameter<double>> GP_double;
map<string, global_parameter<char>> GP_char;

H_FORMAT Hamiltonian_format;

void new_global_bool(bool def, const string& name, const string& des)
{
  GP_bool[name] = global_parameter<bool>(def, name, des);
}

void new_global_int(size_t def, const string& name, const string& des)
{
  GP_int[name] = global_parameter<size_t>(def, name, des);
}

void new_global_double(double def, const string& name, const string& des)
{
  GP_double[name] = global_parameter<double>(def, name, des);
}

void new_global_char(char def, const string& name, const string& des)
{
  GP_char[name] = global_parameter<char>(def, name, des);
}

bool global_bool(const string& name)
{
  if(GP_bool.find(name) == GP_bool.end()) qcm_ED_throw("global parameter "+name+" undefined");
  return GP_bool.at(name).value;
}

size_t global_int(const string& name)
{
  return GP_int.at(name).value;
}

double global_double(const string& name)
{
  return GP_double.at(name).value;
}

char global_char(const string& name)
{
  return GP_char.at(name).value;
}


bool is_global_bool(const string& name) {return (GP_bool.find(name) != GP_bool.end());}
bool is_global_int(const string& name) {return (GP_int.find(name) != GP_int.end());}
bool is_global_double(const string& name) {return (GP_double.find(name) != GP_double.end());}
bool is_global_char(const string& name) {return (GP_char.find(name) != GP_char.end());}



void print_options(int to_file)
{
  if(to_file==1) Print_global_parameters_latex();
  else if(to_file==2) Print_global_parameters_RST();
  else{
    cout << "\n\nOPTIONS";
    Print_global_parameters(cout);
    cout << "\n\n";
  }
}


/**
 Prints the meaning and values of all global parameters
 @param out output stream
 */
void Print_global_parameters(ostream &out)
{
  out << "BOOLEAN OPTIONS\n\n";
  for (auto& x: GP_bool){
    out << x.second << endl;
  }
  out << "\nINTEGER-VALUED OPTIONS\n\n";
  for (auto& x: GP_int){
    out << x.second << endl;
  }
  out << "\nREAL-VALUED OPTIONS\n\n";
  for (auto& x: GP_double){
    out << x.second << endl;
  }
  out << "\nSTRING-VALUED OPTIONS\n\n";
  for (auto& x: GP_char){
    out << x.second << endl;
  }
}






/**
 Prints the meaning and values of all global parameters
 */
void Print_global_parameters_latex()
{
  ofstream fout("QCM_options_def.tex");
  
  if(GP_bool.size()){
    fout << "\\subsection{Boolean options}\n\\begin{longtable}{|l|l|p{10cm}|}\n\\hline\n";
    fout << "Option & default & description \\\\ \\hline\n";
    for (auto& x: GP_bool){
      x.second.print_latex(fout);
    }
    fout << "\\hline\\end{longtable}\n";
  }
  
  if(GP_int.size()){
    fout << "\\subsection{Integer-valued options}\n\\begin{longtable}{|l|l|p{10cm}|}\n\\hline\n";
    fout << "Option & default & description \\\\ \\hline\n";
    for (auto& x: GP_int){
      x.second.print_latex(fout);
    }
    fout << "\\hline\\end{longtable}\n";
  }
  
  if(GP_double.size()){
    fout << "\\subsection{Real-valued options}\n\\begin{longtable}{|l|l|p{10cm}|}\n\\hline\n";
    fout << "Option & default & description \\\\ \\hline\n";
    for (auto& x: GP_double){
      x.second.print_latex(fout);
    }
    fout << "\\hline\\end{longtable}\n";
  }
  
  if(GP_char.size()){
    fout << "\\subsection{Char-valued options}\n\\begin{longtable}{|l|l|p{10cm}|}\n\\hline\n";
    fout << "Option & default & description \\\\ \\hline\n";
    for (auto& x: GP_char){
      x.second.print_latex(fout);
    }
    fout << "\\hline\\end{longtable}\n\n\n";
  }
  fout.close();
}


/**
 Prints the meaning and values of all global parameters
 */
void Print_global_parameters_RST()
{
  ofstream fout("options.rst");
  
  fout << ".. include:: options_intro.txt\n\n";
  
  if(GP_bool.size()){
    fout << "Boolean options\n=========================\n";
    fout << ".. csv-table::\n    :header: \"name\", \"default\", \"description\"\n    :widths: 15, 10, 50\n\n";
    for (auto& x: GP_bool){
      x.second.print_RST(fout);
    }
    fout << "\n\n\n";
  }
  
  if(GP_int.size()){
    fout << "Integer-valued options\n=========================\n";
    fout << ".. csv-table::\n    :header: \"name\", \"default\", \"description\"\n    :widths: 15, 10, 50\n\n";
    for (auto& x: GP_int){
      x.second.print_RST(fout);
    }
    fout << "\n\n\n";
  }
  
  if(GP_double.size()){
    fout << "Real-valued options\n=========================\n";
    fout << ".. csv-table::\n    :header: \"name\", \"default\", \"description\"\n    :widths: 15, 10, 50\n\n";
    for (auto& x: GP_double){
      x.second.print_RST(fout);
    }
    fout << "\n\n\n";
  }
  
  if(GP_char.size()){
    fout << "Char-valued options\n=========================\n";
    fout << ".. csv-table::\n    :header: \"name\", \"default\", \"description\"\n    :widths: 15, 10, 50\n\n";
    for (auto& x: GP_char){
      x.second.print_RST(fout);
    }
    fout << "\n\n\n";
  }
  
  fout.close();
}

void set_global_bool(const string& param, bool value)
{
  if(GP_bool.find(param) == GP_bool.end()){
    qcm_ED_throw("global parameter "+param+" does not exist.");
  }
  else GP_bool.at(param).value = value;
}

void set_global_int(const string& param, size_t value)
{
  if(GP_int.find(param) == GP_int.end()){
    qcm_ED_throw("global parameter "+param+" does not exist.");
  }
  else GP_int.at(param).value = value;
}

void set_global_double(const string& param, double value)
{
  if(GP_double.find(param) == GP_double.end()){
    qcm_ED_throw("global parameter "+param+" does not exist.");
  }
  else GP_double.at(param).value = value;
}

void set_global_char(const string& param, char value)
{
  if(GP_char.find(param) == GP_char.end()){
    qcm_ED_throw("global parameter "+param+" does not exist.");
  }
  else GP_char.at(param).value = value;
}

namespace ED {
  void global_parameter_init()
  {
    if(is_global_parameter_initialized) return;
    is_global_parameter_initialized = true;
        
    new_global_bool(false,"check_lanczos_residual","checks the Lanczos residual at the end of the eigenvector computation");
    new_global_bool(false,"no_degenerate_BL","forbids band lanczos to proceed when the eigenstates have degenerate energies");
    new_global_bool(false,"nosym", "does not take cluster symmetries into account");
    new_global_bool(false,"one_body_solution","Only solves the one-body part of the problem, for the Green function");
    new_global_bool(false,"print_Hamiltonian","Prints the Hamiltonian on the screen, if small enough");
    new_global_bool(false,"strip_anomalous_self","sets to zero the anomalous part of the self-energy");
    new_global_bool(false,"continued_fraction","Uses the continued fraction solver for the Green function instead of the band Lanczos method");
    new_global_bool(false,"verb_ED","prints ED information and progress");
    new_global_bool(false,"print_variances","prints the variance of the operators in files");
    new_global_bool(false,"merge_states","merges states in the mixed state case");


    new_global_double(1e-12,"accur_band_lanczos","energy difference tolerance for stopping the BL process");
    new_global_double(0.01,"accur_continued_fraction","value of beta below which the simple Lanczos process stops");
    new_global_double(1.0e-5,"accur_Davidson","maximum norm of residuals in the Davidson-Liu algorithm");
    new_global_double(1e-7,"accur_deflation","norm below which a vector is deflated in the band Lanczos method");
    new_global_double(1e-12,"accur_lanczos","tolerance of the Ritz residual estimate in the Lanczos method");
    new_global_double(1.0e-5,"accur_Q_matrix","tolerance in the normalization of the Q matrix");
    new_global_double(1e-5,"band_lanczos_minimum_gap","gap between the lowest two states in BL below which the method fails");
    new_global_double(0.01,"minimum_weight","minimum weight in the density matrix");
    new_global_double(1.0e-4, "Qmatrix_tolerance", "minimum value of a Qmatrix coefficient");
    new_global_double(0.0,"temperature", "Temperature of the system.");

    new_global_int(2,"Davidson_states","Number of states requested in the Davidson-Liu algorithm");
    new_global_int(64,"max_dim_print","Maximum dimension for printing vectors and matrices");
    new_global_int(256,"max_dim_full","Maximum dimension for using full diagonalization");
    new_global_int(600,"max_iter_BL","Maximum number of iterations in the band Lanczos procedure");
    new_global_int(400,"max_iter_CF","Maximum number of iterations in the continuous fraction Lanczos procedure");
    new_global_int(1000,"max_iter_lanczos","Maximum number of iterations in the Lanczos procedure");
    new_global_int(0,"seed","seed of the random number generator");

    new_global_char('E', "Hamiltonian_format", "Desired Hamiltonian format: S (CSR matrix), O (individual operators), F (factorized), N (none = on the fly)");
  }
}

