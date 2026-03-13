#include <fstream>
#include "lattice_model.hpp"

//! superconducting pairing link (for graphical purposes)
struct SC_link{
  size_t s1;
  size_t s2;
  Complex v;
  int type;// 0:singlet, 1:triplet, 2:dz
};

//==============================================================================
/**
 produces an asymptote file describing the lattice operator op
 @param op [in] lattice operator
 @param asy_labels [in] if true, prints the labels on each site
 @param asy_orb [in] if true, prints the lattice orbital labels on each site
 @param asy_neighbors [in] if true, draws the sites of the neighboring clusters
 @param asy_working_basis [in] if true, works in the working basis, not the physical basis
 */
void lattice_model::asy_print(const lattice_operator &op, bool asy_labels, bool asy_orb, bool asy_neighbors, bool asy_working_basis)
{
  // if(op.type == latt_op_type::Hubbard) return;
  
  ofstream fout(op.name+".asy");
  if (!fout.good()) qcm_throw("failed to open file " + op.name + ".asy");

  double asy_scale = global_double("asy_scale");
  double asy_spin_scale = global_double("asy_spin_scale");
  
  vector3D<double> r;
  bool superlattice_draw = true;
  
  if(superlattice.D == 1 and superlattice.e[0].x == 0 and superlattice.e[0].y == 0) superlattice_draw = false;
  if(superlattice.D == 0) superlattice_draw = false;  
  vector3D<double>::dimension=2;
  
  string header;
  header = R"(
  unitsize(1.5cm);
  dotfactor=10;
  pen axes_pen=red+fontsize(7)+linewidth(0.3);
  pen dof_label_pen=red+fontsize(6);
  pen label_pen=heavygreen+fontsize(8);
  pen link_pen=blue+fontsize(7)+linewidth(0.5);
  pen link2_pen=lightblue+dotted+fontsize(5)+linewidth(0.5);
  pen neg_pen=red;
  pen pos_pen=blue;
  pen sc_pen=orange + linewidth(0.5);
  pen site_pen=black+linewidth(1);
  pen site2_pen=heavygreen;
  pen spin_pen_clus = heavygreen+linewidth(0.7);
  pen spin_pen=heavyred+linewidth(1);
  pen value_pen=black + linewidth(0.25);
  real hsvcorr = 30;
  real sc=0.15;
  )";

  fout << header;

  header = R"(
    void circledlabel(string s, pair r){
      filldraw(shift(r)*scale(0.2)*unitcircle,white,pos_pen);
      label(s,r);
    }
    void framedlabel(string s, pair r){
      filldraw(shift(r-(0.1,0.1))*scale(0.2)*unitsquare,neg_pen,neg_pen);
      label(s,r,fontsize(9));
    }
    )";
  fout << header;
  
  if (op.is_complex){
    header = R"(
      void circlevalue(pair z, pair r){
        pair zs = sc*z;
        filldraw(shift(r)*scale(abs(zs))*unitcircle,white,value_pen);
        draw((r-zs)--(r+zs),EndArrow(15*abs(zs)));
      }
      void drawtriplet(pair r1, pair r2, pair z){
        draw(r1--r2,sc_pen,MidArrow(HookHead,3));
        circlevalue(z,0.5*(r1+r2));
      }
      void drawsinglet(pair r1, pair r2, pair z){
        draw(r1--r2,sc_pen);
        circlevalue(z,0.5*(r1+r2));
      }
      void drawonebody(pair r1, pair r2, pair z, bool v, int s){
        pen p;
        if(s==1) p += linetype("3 3");
        draw(r1--r2, p, EndArrow);
        circlevalue(z,0.5*(r1+r2));
      }
      )";
  }
  else{
    header = R"(
      void circlevalue(real z, pair r){
        pair zs = sc*z;
        if(z > 0) filldraw(shift(r)*scale(abs(zs))*unitcircle,pos_pen,value_pen);
        else filldraw(shift(r)*scale(abs(zs))*unitcircle,neg_pen,value_pen);
      }
      void drawtriplet(pair r1, pair r2, real z){
        draw(r1--r2,sc_pen,MidArrow(HookHead,3));
        circlevalue(z,0.5*(r1+r2));
      }
      void drawsinglet(pair r1, pair r2, real z){
        draw(r1--r2,sc_pen);
        circlevalue(z,0.5*(r1+r2));
      }
      void drawonebody(pair r1, pair r2, real z, bool v, int s){
        pen p;
        if(s==1) p += linetype("3 3");
        draw(r1--r2, p);
        circlevalue(z,0.5*(r1+r2));
      }
      void drawinteraction(pair r1, pair r2, int s){
        pen p;
        p = value_pen + linewidth(1);
        if(s==1) p += linetype("3 3");
        draw(r1--r2, p);
      }
      )";
  }

  fout << header;
  
  
  if(superlattice_draw and superlattice.D > 0){
    r = asy_working_basis ?  vector3D<double>(superlattice.e[0]) : phys.to(vector3D<double>(superlattice.e[0]));
    fout << "draw((0,0)--(" << r.x << "," << r.y << "), axes_pen, EndArrow);\n";
  }
  if(superlattice_draw and superlattice.D > 1){
    r = asy_working_basis ?  vector3D<double>(superlattice.e[1]) : phys.to(vector3D<double>(superlattice.e[1]));
    fout << "draw((0,0)--(" << r.x << "," << r.y << "), axes_pen, EndArrow);\n";
  }
  if(superlattice_draw and superlattice.D > 2){
    r = asy_working_basis ?  vector3D<double>(superlattice.e[2]) : phys.to(vector3D<double>(superlattice.e[2]));
    fout << "draw((0,0)--(" << r.x << "," << r.y << "), axes_pen, EndArrow);\n";
  }
  
  //-----------------------------------------------------------------------------------
  if(op.type == latt_op_type::Hubbard){
    for(auto& x : op.elements){
      if(x.spin1 or x.spin2) continue;
      if(x.site1 == x.site2) continue;
      vector3D<double> r1 = asy_working_basis ?  vector3D<double>(sites[x.site1].position) : phys.to(sites[x.site1].position);
      vector3D<double> r2 = asy_working_basis ?  vector3D<double>(sites[x.site2].position+neighbor[x.neighbor]) : phys.to(sites[x.site2].position+neighbor[x.neighbor]);
      if(sites[x.site1].cluster == sites[x.site2].cluster and x.neighbor == 0){
        fout << "drawinteraction(" << r1 << ", " << r2 << ", 0);\n";
      }
      else if(asy_neighbors){
        fout << "drawinteraction(" << r1 << ", " << r2 << ", 1);\n";
      }
    }
  }
  //-----------------------------------------------------------------------------------



  if(op.type == latt_op_type::one_body){
    for(auto& x : op.elements){
      if(x.neighbor) continue;

      // if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      // if(x.spin1 or x.spin2) continue;
      if(x.spin1) continue;
      if(x.site1 == x.site2) continue;
      vector3D<double> r1 = asy_working_basis ?  vector3D<double>(sites[x.site1].position) : phys.to(sites[x.site1].position);
      vector3D<double> r2 = asy_working_basis ?  vector3D<double>(sites[x.site2].position) : phys.to(sites[x.site2].position);
      if(sites[x.site1].cluster == sites[x.site2].cluster){
        if(op.is_complex) fout << "drawonebody(" << r1 << ", " << r2 << ", " << x.v << ", false, " << x.spin2 << ");\n";
        else fout << "drawonebody(" << r1 << ", " << r2 << ", " << x.v.real() << ", false, " << x.spin2 << ");\n";
      }
      else{
        if(op.is_complex) fout << "drawonebody(" << r1 << ", " << r2 << ", " << x.v << ", true, " << x.spin2 << ");\n";
        else fout << "drawonebody(" << r1 << ", " << r2 << ", " << x.v.real() << ", true, " << x.spin2 << ");\n";
      }
    }
    
    // looking for spin operators
    
    vector<double> Sz(sites.size(),0.0);
    vector<double> charge(sites.size(), 0.0);
    vector<double> Sx(sites.size(),0.0);
    for(auto& x : op.elements){
      if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      if(x.site1 != x.site2) continue;
      if(x.spin1 == x.spin2){
        Sz[x.site1] += (2*(int)x.spin1-1)*x.v.real();
        charge[x.site1] += x.v.real();
      }
      else Sx[x.site1] += x.v.real();
    }
    for(auto& x : op.elements){
      if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      if(x.site1 != x.site2) continue;
      vector3D<double> r = asy_working_basis ?  vector3D<double>(sites[x.site1].position) : phys.to(sites[x.site1].position);

      vector3D<double> S(Sx[x.site1],Sz[x.site1],0.0);
      if(!S.is_null()){
        S *= asy_spin_scale;
        fout << "draw((" << r.x-S.x << ',' << r.y-S.y << ")--("<< r.x+S.x << ',' << r.y+S.y << "),spin_pen,EndArrow(4));\n";
      }
      double d = charge[x.site1];
      if(abs(d) > 0.001){
        fout << "circlevalue(" << d << "," << r << ");\n";
      }
    }
  }
  
  else if(op.type == latt_op_type::Hund 
    or op.type == latt_op_type::Heisenberg
    or op.type == latt_op_type::X
    or op.type == latt_op_type::Y
    or op.type == latt_op_type::Z
  ){
    for(auto& x : op.elements){
      if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      if(x.spin1 or x.spin2) continue;
      vector3D<double> r1 = asy_working_basis ?  vector3D<double>(sites[x.site1].position) : phys.to(sites[x.site1].position);
      vector3D<double> r2 = asy_working_basis ?  vector3D<double>(sites[x.site2].position) : phys.to(sites[x.site2].position);
      if(sites[x.site1].cluster == sites[x.site2].cluster){
        fout << "draw(" << r1 << "--"<< r2 << ", link_pen);\n";
      }
      else{
        fout << "draw(" << r1 << "--"<< r2 << ", link2_pen);\n";
      }
    }
  }

  else if(op.mixing&HS_mixing::anomalous){
    vector<SC_link> VL;
    for(auto& x : op.elements){
      // if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      if(x.neighbor) continue;
      if(x.site1 == x.site2){
        if(x.spin1 < x.spin2) VL.push_back({x.site1, x.site1, x.v, 0});
        else VL.push_back({x.site1, x.site1, -x.v, 0});
      }
      if(x.site1 < x.site2){
        if(x.spin1 == 0 and x.spin2 == 0) VL.push_back({x.site1, x.site2, x.v, 2});
        if(x.spin1 < x.spin2) VL.push_back({x.site1, x.site2, x.v, 0});
        else VL.push_back({x.site1, x.site2, -x.v, 0});
      }
    }
    for(auto& x : op.elements){
      if(x.neighbor) continue;
      // if(x.neighbor or (sites[x.site1].cluster != sites[x.site2].cluster)) continue;
      if(x.site1 > x.site2){
        for(auto& y : VL){
          if(y.s1 == x.site2 and y.s2 == x.site1){
            if((x.spin1 < x.spin2 and x.v == -y.v) or (x.spin1 > x.spin2 and x.v == y.v)) {y.type = 1; break;}
          }
        }
      }
    }
    for(auto& y : VL){
      vector3D<double> r1 = asy_working_basis ?  vector3D<double>(sites[y.s1].position) : phys.to(sites[y.s1].position);
      vector3D<double> r2 = asy_working_basis ?  vector3D<double>(sites[y.s2].position) : phys.to(sites[y.s2].position);
      if(op.is_complex){
        if(y.type==0) fout << "drawsinglet(" << r1 << "," << r2 << ", (" << asy_scale*real(y.v) << "," << asy_scale*imag(y.v) << "));\n";
        else fout << "drawtriplet(" << r1 << "," << r2 << ", (" << asy_scale*real(y.v) << ", " << asy_scale*imag(y.v) << "));\n";
      }
      else{
        if(y.type==0) fout << "drawsinglet(" << r1 << "," << r2 << "," << asy_scale*real(y.v) << ");\n";
        else fout << "drawtriplet(" << r1 << "," << r2 << "," << asy_scale*real(y.v) << ");\n";
      }
    }

  }
  

  //..............................................................................
  // then, the sites only
  // draw dots or open circles for z > 0

  // first find the range of z
  int zmax = 0;
  int zmin = 0;
  for(int i=0; i<sites.size(); i++){
    if(sites[i].position.z > zmax) zmax = sites[i].position.z;
    if(sites[i].position.z < zmin) zmin = sites[i].position.z;
  }
  double zscale[zmax-zmin+1];
  for(int i=0; i<zmax-zmin+1; i++) zscale[i] = (12.0 + 4*i)*5/(6*72);

  for(int i=0; i<sites.size(); i++){
    vector3D<double> r = asy_working_basis ?  vector3D<double>(sites[i].position) : phys.to(sites[i].position);
    int z = sites[i].position.z;
    if(z == zmin)
      fout << "filldraw(shift" << r << "*scale(" << zscale[0] << ")*unitcircle,site_pen); // site\n";
    else{
      fout << "draw(shift" << r << "*scale(" << zscale[z-zmin] << ")*unitcircle,site_pen); // site\n";
    }

    if(asy_orb){
      if(n_band>1) fout << "label('" << sites[i].orb+1 << "', " << r << ", NE, label_pen); // site\n";
    }
    if(asy_labels){
      fout << "circledlabel('" << i+1 << "', " << r << "); // site\n";
    }
  }
  
  //..............................................................................
  // then, the sites on the neighboring clusters
  
  if(asy_neighbors){
    for(int ix=1; ix<neighbor.size(); ++ix){
      vector3D<int64_t> s = neighbor[ix];
      for(int i=0; i<sites.size(); i++){
        vector3D<double> r = asy_working_basis ?  vector3D<double>(sites[i].position+s) : phys.to(vector3D<double>(sites[i].position+s));
        fout << "dot(" << r << ",site2_pen);\n";
      }
    }
  }
  
  //..............................................................................
  
  fout.close();
  vector3D<double>::dimension=3;
}

