#ifndef parameter_h
#define parameter_h

#include <string>
#include <set>
#include <map>
#include <vector>
#include <memory>
#include <tuple>

struct lattice_model;
struct lattice_model_instance;

using namespace std;

enum class print_format {names, values};

//! parameter of the model (lattice or cluster) with its dependency
struct parameter{
  string name;
  double value;
  size_t label; //!< label of the cluster on which this operator is defined (0 if a lattice operator)
  double multiplier;
  shared_ptr<parameter> ref;
  set<shared_ptr<parameter>> deref;
  const static char separator = '_';
  
  parameter(const string& _name, int _label, double v=0.0);
  void check_name(const string &str);
};

//! set of parameters for a model, including dependencies
struct parameter_set{
  shared_ptr<lattice_model> model; //!< backtrace to the model
  map<string,shared_ptr<parameter>> param; //!< map of strings to parameters
  vector<string> CDMFT_variational; //!< list of CDMFT variational parameters

  parameter_set();
  parameter_set(shared_ptr<lattice_model> _model, vector<pair<string, double>> values, vector<tuple<string, double, string>> equiv);
  map<string,double> value_map();
  void set_value(shared_ptr<parameter> P, const double &v);
  void set_value(const string& name, const double &v);
  void set_multiplier(const string& name, const double &v);
  void check_existence(string& S);
  void copy_to(vector<double> &x);
  void copy_from(const vector<double> &x);
  void print(ostream& out);
  bool is_dependent(const string &S);
  void CDMFT_variational_set(const vector<string>& vars);

  static bool parameter_set_defined;
};
#endif /* parameter_h */
