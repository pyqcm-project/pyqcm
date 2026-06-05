/**
 * @file lattice_operator.cpp
 * @brief Implementation of the lattice_operator class (Hamiltonian terms).
 */
#include "lattice_operator.hpp"
#include "lattice_model.hpp"


template<>
double native<double>(Complex x) { return real(x); }

template<>
Complex native<Complex>(Complex x) { return x; }


map<string,latt_op_type> lattice_operator::op_type_map = {
  {"one-body", latt_op_type::one_body},
  {"singlet", latt_op_type::singlet},
  {"dz", latt_op_type::dz},
  {"dy", latt_op_type::dy},
  {"dx", latt_op_type::dx},
  {"Hubbard", latt_op_type::Hubbard},
  {"Hund", latt_op_type::Hund},
  {"Heisenberg", latt_op_type::Heisenberg},
  {"X", latt_op_type::X},
  {"Y", latt_op_type::Y},
  {"Z", latt_op_type::Z}
};


/**
constructor.
@param _model [in] backtrace to the lattice model
@param _name [in] name of the operator
@param _type [in] type of operator (refer to the enumeration latt_op_type in the header file)
 */
lattice_operator::lattice_operator(lattice_model& _model, const string& _name, latt_op_type _type) :
model(_model),
name(_name),
type(_type),
mixing(0),
is_active(false),
is_density_wave(false),
is_closed(false),
is_interaction(false),
is_complex(false),
norm(1.0),
nambu_correction(0.0),
nambu_correction_full(0.0),
average(0.0)
{
  if(model.is_closed) qcm_throw("It is too late to add an operator: the model is closed");
  elements.reserve(2*model.sites.size());
}

/**
constructor.
@param _model [in] backtrace to the lattice model
@param _name [in] name of the operator
@param _type [in] type of operator (refer to the map lattice_operator::op_type_map defined above)
 */
lattice_operator::lattice_operator(lattice_model& _model, const string& _name, const string& _type) :
model(_model),
name(_name),
mixing(0),
is_active(false),
is_density_wave(false),
is_closed(false),
is_interaction(false),
is_complex(false),
norm(1.0),
nambu_correction(0.0),
nambu_correction_full(0.0)
{
  if(model.is_closed) qcm_throw("It is too late to add an operator: the model is closed");
  type = op_type_map[_type];
  elements.reserve(2*model.sites.size());
}



/**
 adds a matrix element to the operator
 @param pos [in] array of positions
 @param link [in] array of bonds
 @param v [in] complex amplitude
 @param opt [in] option, that depends on the type of operator
 */
void lattice_operator::add_matrix_element(const vector3D<int64_t>& pos, const vector3D<int64_t>& link, Complex v, const string& opt)
{
  if(is_closed) qcm_throw("It is too late to modifiy the operator: it is closed");
  if(model.is_closed) qcm_throw("It is too late to modifiy an operator: the model is closed");
  
  if(model.position_map.find(pos) == model.position_map.end()){
    qcm_throw("position "+to_string<vector3D<int64_t>>(pos)+" does not exist!");
  }
  int s1 = (int)model.position_map.at(pos);
  auto match = model.find_second_site(s1, link);
  if(!match) return;
  auto [s2, ni, ni_opp] = *match;
  if(type == latt_op_type::one_body){
    if(opt.size()!=2) qcm_throw("one-body matrix element requires an optional two-character string");
    int tau = (int)(opt[0]-'0');
    int sigma = (int)(opt[1]-'0');
    if(tau == 0){
      for(int alpha=0; alpha<2; alpha++){
        for(int beta=0; beta<2; beta++){
          Complex ec = model.pauli[sigma][alpha][beta]*v;
          if(ec.imag() == 0.0 and ec.real() == 0.0) continue;
          elements.push_back({s1,alpha,s1,beta,ni,ec});
        }
      }
    }
    else{
      for(int i=0; i<2; i++){
        int si = s1+i*(s2-s1);
        for(int j=0; j<2; j++){
          int sj = s1+j*(s2-s1);
          for(int alpha=0; alpha<2; alpha++){
            for(int beta=0; beta<2; beta++){
              Complex ec = model.pauli[tau][i][j]*model.pauli[sigma][alpha][beta]*v;
              if(ec.imag() == 0.0 and ec.real() == 0.0) continue;
              if(j<i) elements.push_back({si,alpha,sj,beta,ni_opp,ec});
              else elements.push_back({si,alpha,sj,beta,ni,ec});
            }
          }
        }
      }
    }
  }
  
  switch(type){
    case latt_op_type::one_body :
      break;
    case latt_op_type::singlet :
    case latt_op_type::dz :
    case latt_op_type::dy :
    case latt_op_type::dx :
      model.add_anomalous_elements(elements, s1, s2, ni, ni_opp, v, type);
      break;
    case latt_op_type::Hubbard :
      if(ni) qcm_throw("faulty matrix element for Hubbard interaction");
      if(s1 != s2){
        elements.push_back({s1, 0, s2, 0, 0, v});
        elements.push_back({s1, 1, s2, 0, 0, v});
        elements.push_back({s1, 0, s2, 1, 0, v});
        elements.push_back({s1, 1, s2, 1, 0, v});
      }
      else elements.push_back({s1, 0, s2, 1, 0, v});
      break;
    case latt_op_type::Hund :
      if(s1 == s2 and ni==0) qcm_throw("faulty matrix element for Hund interaction");
      elements.push_back({s1, 0, s2, 0, ni, v});
      break;
    case latt_op_type::Heisenberg :
      if(s1 == s2 and ni==0) qcm_throw("faulty matrix element for Heisenberg interaction");
      elements.push_back({s1, 0, s2, 0, ni, v});
      break;
    case latt_op_type::X :
      if(s1 == s2 and ni==0) qcm_throw("faulty matrix element for spin X interaction");
      elements.push_back({s1, 0, s2, 0, ni, v});
      break;
    case latt_op_type::Y :
      if(s1 == s2 and ni==0) qcm_throw("faulty matrix element for spin Y interaction");
      elements.push_back({s1, 0, s2, 0, ni, v});
      break;
    case latt_op_type::Z :
      if(s1 == s2 and ni==0) qcm_throw("faulty matrix element for spin Z interaction");
      elements.push_back({s1, 0, s2, 0, ni, v});
      break;
  }
}




/**
 closes the operator
 */
void lattice_operator::close()
{
  is_closed = true;
  consolidate();
  if(is_complex) model.build_cluster_operators<Complex>(*this);
  else model.build_cluster_operators<double>(*this);
}






/**
 sets some properties of the operator, once its matrix elements have been set.
 */
void lattice_operator::consolidate()
{

  // consolidate duplicate elements
  // creating the map, emptying the list of elements and creating it
  map<lattice_index_pair, complex<double>> element_map;
  for(auto& x : elements){
    element_map[lattice_index_pair(x.site1, x.spin1, x.site2, x.spin2, x.neighbor)] += x.v;
  }
  elements.clear();
  elements.reserve(element_map.size());
  for(auto& [y, v] : element_map){
    if(abs(v) < SMALL_VALUE)  continue;
    if(fabs(v.imag()) > SMALL_VALUE) is_complex = true;
    else v = v.real();
    elements.push_back(lattice_matrix_element(y.site1, y.spin1, y.site2, y.spin2, y.neighbor, v));
  }


  for(auto& x : elements){
    if(fabs(x.v.imag()) > SMALL_VALUE){is_complex = true; break;}
  }

  // finding the mixing state induced by the operator
  if(type==latt_op_type::one_body){
    for(auto& x : elements) if(x.spin1 != x.spin2) {mixing |= HS_mixing::spin_flip; break;}
  }
  if(mixing&HS_mixing::anomalous){
    for(auto& x : elements) if(x.spin1 == x.spin2) {mixing |= HS_mixing::spin_flip; break;}
  }
  check_spin_symmetry();
  
  //  computes the Nambu correction needed for lattice averages
  if(type==latt_op_type::one_body){
    for(auto& x : elements){
      if(x.site1 == x.site2 and x.spin1 == x.spin2 and x.neighbor == 0){
        nambu_correction_full += real(x.v);
        if(x.spin1 == 1) nambu_correction += real(x.v);
      }
    }
  }
}



/**
 determines whether the operator is symmetric under the exchange of up and down spins
 sets the bool variable is_spin_symmetric accordingly
 */
void lattice_operator::check_spin_symmetry()
{
  if(mixing&HS_mixing::spin_flip) return; // do not bother if the operator is already spin flip
  vector<bool> pass(elements.size(), false);
  for(size_t i = 0; i<elements.size(); i++){
    if(pass[i]) continue;
    for(size_t j = i; j<elements.size(); j++){
      if(pass[j]) continue;
      if(elements[i].v == elements[j].v
         and elements[i].site1 == elements[j].site1
         and elements[i].site2 == elements[j].site2
         and elements[i].neighbor == elements[j].neighbor
         and elements[i].spin1 == elements[j].spin1
         ){
        pass[j] = pass[i] = true; break;
      }
    }
    if(pass[i]) continue;
  }
  size_t i=0;
  for(i = 0; i<elements.size(); i++) if(pass[i] == false) break;
  if(i==elements.size()) mixing |= HS_mixing::up_down; // adds the bit associated with spin asymmetry
}






