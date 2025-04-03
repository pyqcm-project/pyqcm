#include <fstream>
#include <set>

#include "parameter_set.hpp"
#include "parser.hpp"
#include "global_parameter.hpp"
#include "lattice_model.hpp"

#define FIELD_WIDTH 26

//==============================================================================
// implementation of struct 'parameter'


/**
constructor
@param _name [in] name of the parameter
@param _label [in] label (i.e. cluser label, or lattice if = 0)
@param v [in] value of the parameter
*/
parameter::parameter(const string& _name, int _label, double v)
  : name(_name), label(_label), value(v), ref(), multiplier(1.0)
{
  if(name == "") qcm_throw("parameter must have a name");
}




/**
Checks the admissibility of the name
@param str [in] name of the parameter
*/
void parameter::check_name(const string &str)
{
  size_t pos = str.rfind(separator);
  if(pos != string::npos) qcm_throw("The character '" + string(1,separator) + "' is forbidden in parameter names ("+str+")");
}


//==============================================================================
// implementation of struct 'parameter_set'

bool parameter_set::parameter_set_defined = false;

/**
 default constructor
 */
parameter_set::parameter_set()
{
}



/**
 constructor from arguments
 @param _model [in] backtrace to the lattice model
 @param values [in] values of the different parameters, as a map
 @param equiv [in] array of equivalence of parameters : <name of parameter, multipler, name of reference parameter>
 */
parameter_set::parameter_set(shared_ptr<lattice_model> _model, vector<pair<string, double>> values, vector<tuple<string, double, string>> equiv)
: model(_model)
{
  if(parameter_set_defined){
    qcm_throw("parameter_set() can only be called once!");
  }
  parameter_set_defined = true;

  model->close_model(); // once the parameters are initialized, the model can no longer be modified

  set<string> param_list;

  for(auto& x : values){
    check_existence(x.first);
    if(param_list.find(x.first) != param_list.end()){
      qcm_throw(x.first+" has been assigned more than once. This is forbidden for safety reasons.");
    }
    param_list.insert(x.first);
    auto P = model->name_and_label(x.first);
    if(P.second > 0 && model->clusters[P.second-1].ref != P.second-1){
      qcm_throw(x.first+" : values of parameters on dependent clusters should not be set!");
    }
    param[P.first] = make_shared<parameter>(x.first, P.second, x.second);
  }
  for(auto& x : equiv){
    string name1;
    string name2;
    double mult;
    tie(name1, mult, name2) = x;
    check_existence(name1);
    
    if(param_list.find(name1) != param_list.end()){
      qcm_throw(name1+" has been assigned more than once. This is forbidden for safety reasons.");
    }
    param_list.insert(name1);

    auto P = model->name_and_label(name1);
    if(param.find(name2) == param.end()){
      qcm_throw(name2+" has not been assigned so far and cannot be used as a reference");
    }
    auto X = make_shared<parameter>(P.first, P.second, mult*param.at(name2)->value);
    param[name1] = X;
    X->ref = param[name2];
    X->multiplier = mult;
    param[name2]->deref.insert(X);
  }

  // completing with implicit dependent operators
  set<shared_ptr<parameter>> param_tmp;
  for(auto& x : param){
    if(x.second->label == 0){
      auto& op = model->term.at(x.first);
      for(int i=0; i<model->clusters.size(); i++){
        if(!op->in_cluster[i]) continue;
        string name = x.first+parameter::separator+to_string(i+1);
        if(param.find(name) != param.end()) continue;
        auto tmp = make_shared<parameter>(name, i+1, x.second->value);
        param_tmp.insert(tmp);
        tmp->ref = x.second;
      }
    }
  }
  for(auto& x : param_tmp){
    param[x->name] = x;
    x->ref->deref.insert(x);
  }
}



/**
 sets the value of the parameter and its children
 @param name [in] name of parameter
 @param v [in] value to set the parameter to
 */
void parameter_set::set_value(const string& name, const double &v)
{
  if(param.find(name) == param.end()) qcm_throw("parameter "+name+" does not exist! impossible to set value");
  if(param[name]->ref != nullptr)
    qcm_throw("parameter "+name+" is dependent! impossible to set value");
  set_value(param[name], v);
}



/**
 sets the value of the parameter and its children
 @param P [in] pointer to parameter
 @param v [in] value to set the parameter to
 */
void parameter_set::set_value(shared_ptr<parameter> P, const double &v)
{
  P->value = v;
  for(auto& x : P->deref){
    set_value(x, v*x->multiplier);
  }
}


/**
 sets the multiplier of a dependent parameter (changes its initial value)
 @param name [in] name of parameter
 @param v [in] new value of the multiplier
 */
void parameter_set::set_multiplier(const string& name, const double &v)
{
  if(param.find(name) == param.end()) qcm_throw("parameter "+name+" does not exist! impossible to set multiplier");
  if(param[name]->ref == nullptr)
    qcm_throw("parameter "+name+" is independent! impossible to set multiplier");
  param[name]->multiplier = v;
}



/**
 checks the existence of a parameter name in the parameter_set
 @param name [in] name of the parameter
 */
void parameter_set::check_existence(string& name)
{
  auto P = model->name_and_label(name, true);
  if(P.second){
    if(!ED::exists(model->clusters[P.second-1].name, P.first)) qcm_throw("operator "+name+" does not exist in model");
  }
}




/**
 returns a map <string, double> providing the value of each parameter
 @returns a map <string, double>
*/
map<string,double> parameter_set::value_map()
{
  map<string,double> X;
  for(auto& x : param){
    X[x.first] = x.second->value;
  }
  return X;
}





/**
 Prints parameters to a stream
 @param out [in] output stream
 */
void parameter_set::print(ostream& out)
{
  out << setprecision(10);
  for(auto& x : param){
    if(x.second->ref!=nullptr){
      ostringstream sout;
      sout << x.first << " := " << x.second->multiplier << " x " << x.second->ref->name;
      out << left << setw(FIELD_WIDTH) << sout.str();
    }
    else{
      ostringstream sout; sout << x.first << " = " << x.second->value;
      out << left << setw(FIELD_WIDTH) << sout.str();
    }
    if(x.second->deref.size()){
      auto it = x.second->deref.begin();
      out << "--> " << (*it)->name;
      it++;
      while(it != x.second->deref.end()){
        out << ", " << (*it)->name;
        it++;
      }
    }
    out << endl;
  }
}




/**
 returns true is the parameter named S is dependent
 */
bool parameter_set::is_dependent(const string &S)
{
  if(param.find(S)==param.end())
    qcm_throw(S + " is not in the set of parameters. Its dependence cannot even be checked!");
  return (param.at(S)->ref != nullptr);
}


/**
sets the CDMFT variational parameters to vars
@param vars [in] array of parameter names
*/
void parameter_set::CDMFT_variational_set(const vector<string>& vars) {
  for(auto& s : vars){
    if(param.find(s)==param.end())
      qcm_throw(s + " is not in the set of parameters: it cannot be a variational parameter.");
    if(is_dependent(s)) qcm_throw("parameter "+s+" is dependent. Cannot be a variational parameter.");
  }
  CDMFT_variational = vars;
}



