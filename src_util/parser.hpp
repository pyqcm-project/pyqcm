#ifndef parser_h
#define parser_h

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include "types.hpp"

#define SHORT_DISPLAY 7
#define NORMAL_DISPLAY 9
#define LONG_DISPLAY 14

using namespace std;

namespace parser{
  extern	bool no_rewind;
  inline void next_line(std::istream &flux){flux.ignore(numeric_limits<streamsize>::max(), '\n');};
  istream & find_next(istream &flux, const char* search);
}


//-----------------------------------------------------------------------------
// Templates

//! converts a string to a generic type
template<typename T>
T from_string(const string &s){
  istringstream sin(s);
  T x;
  sin >> x;
  if(sin.fail()){
    cerr << "Fatal error in 'from_string()', string '" << s << "' cannot be interpreted as type requested\n";
    exit(1);
  }
  return x;
}

//! converts a generic type to a string
template<typename T>
string to_string(const T &x){
  ostringstream sout;
  sout << x;
  return sout.str();
}

//-----------------------------------------------------------------------------
// Declarations

int cluster_index_from_string(string& S);
istream & operator==(istream &input, const char* search);
istream & operator==(istream &input, const string &search);
istream & operator>>(istream &input, const char* search);
istream & operator>>(istream &input, const string &search);
vector<std::string> split_string(const string &s, char delim);
vector<string> read_strings(istream &s);
void banner(const char c, const char s[128], std::ostream &fout = std::cout);
void banner(const char c, const string &s, ostream &fout = std::cout);
void check_name(const string& S);
void check_signals();
struct qcm_error : public std::runtime_error {
    explicit qcm_error(const std::string& msg) : std::runtime_error(msg) {}
};

#define QCM_ASSERT(cond) \
    do { if (!(cond)) qcm_throw(std::string("Assertion failed: " #cond \
        " in " __FILE__ ":") + std::to_string(__LINE__)); } while(0)

void qcm_catch(const std::exception& e);
void qcm_ED_catch(const std::exception& e);
void qcm_ED_throw(const std::string& s);
void qcm_throw(const std::string& s);
void qcm_warning(const std::string& s);

//-----------------------------------------------------------------------------
// Declarations and code

/**
 Chops a real number \a x if its absolute value is smaller than the limit \a c
 */
inline double chop(double x, double c=1e-6){return (fabs(x)<c) ? 0 : x;}

/**
 Chops a complex number \a x separately for its real and imaginary parts
 */
inline Complex chop(Complex x, double c=1e-6){return Complex(fabs(x.real()) < c ? 0 : x.real(),fabs(x.imag()) < c ? 0:x.imag());}


#endif
