/*! \file
 \brief Methods of the struct \a lattice
 */

#include "parser.hpp"
#include "lattice3D.hpp"

#define LARGE_UNIT 2048 // ! WARNING !! must not be too large for fear of overflow!




lattice3D::lattice3D(vector<int64_t> _e)
{
  if(_e.size()%3 or _e.size()>9) qcm_throw("the superlattice array should have 0, 3, 6 or 9 elements");
  D = _e.size()/3;
  e.assign(D, vector3D<int64_t>());
  int j=0;
  for(int i=0; i<D; i++){
    e[i].x = _e[j++];
    e[i].y = _e[j++];
    e[i].z = _e[j++];
  }
  init();
}




/**
 reads the superlattice vectors from a stream and completes the initialization
 @param fin input stream
 */
void lattice3D::read(istream &fin)
{
	e = read_vectors<int64_t>(fin);
	init();
}






/**
 Completes the initialization of the basis
 */
void lattice3D::init(){
  if(global_bool("zero_dim")) e.clear();
  D = (int)e.size();
  QCM_ASSERT(D<4);
  
  if(D==0){ // define a large repeatable unit, even though not repeated
    e.push_back(vector3D<int64_t>(LARGE_UNIT,0,0));
    e.push_back(vector3D<int64_t>(0,LARGE_UNIT,0));
    e.push_back(vector3D<int64_t>(0,0,LARGE_UNIT));
  }
  else if(D==1){
    if(e[0].x != 0){
      if(e[0].y != 0 or e[0].z != 0)
        qcm_throw("in a 1D model, two of the components (x,y or z) must be absent from the lattice vectors");
      e.push_back(vector3D<int64_t>(0,LARGE_UNIT,0));
      e.push_back(vector3D<int64_t>(0,0,LARGE_UNIT));
    }
    else if(e[0].y != 0){
      if(e[0].z != 0)
        qcm_throw("in a 1D model, two of the components (x,y or z) must be absent from the lattice vectors");
      e.push_back(vector3D<int64_t>(LARGE_UNIT,0,0));
      e.push_back(vector3D<int64_t>(0, 0, LARGE_UNIT));
    }
    else if(e[0].z != 0){
      e.push_back(vector3D<int64_t>(LARGE_UNIT,0,0));
      e.push_back(vector3D<int64_t>(0,LARGE_UNIT,0));
    }
    else
      qcm_throw("in a 1D model, all lattice vectors cannot be zero!");
}
  else if(D==2){
    if (e[0].z == 0 and e[1].z == 0){
      e.push_back(vector3D<int64_t>(0,0,LARGE_UNIT));
    }
    else if (e[0].x == 0 and e[1].x == 0){
      e.push_back(vector3D<int64_t>(LARGE_UNIT,0,0));
    }
    else if (e[0].y == 0 and e[1].y == 0){
      e.push_back(vector3D<int64_t>(0,LARGE_UNIT,0));
    }
    else
      qcm_throw("in a 2D model, one of the components (x,y or z) must be absent from the lattice vectors");
  }

  M.set_size(3);
	
	vol = triple_product(e[0],e[1],e[2]);
	if(vol < 0){
		e[0] = -e[0];
		vol = -vol;
	}
	M(0,0) = e[0].x;
	M(1,0) = e[0].y;
	M(2,0) = e[0].z;
	
	M(0,1) = e[1].x;
	M(1,1) = e[1].y;
	M(2,1) = e[1].z;
	
	M(0,2) = e[2].x;
	M(1,2) = e[2].y;
	M(2,2) = e[2].z;
	
	M.inverse();
}







/**
 folds a position in the superlattice : r = R + S
 R belongs to the superlattice, and S is a remainder that belongs to the original lattice's unit cell.
 @param r [in] integer position, in the working basis
 @param R [out] integer position in the superlattice, in the working basis
 @param S [out] remainder in the unit cell, in the working basis
 */
void lattice3D::fold(const vector3D<int64_t> &r, vector3D<int64_t> &R, vector3D<int64_t> &S)
{
	vector3D<int64_t> Q;
	
	// use Cramer's rule to find R such that r = (R.x e[0] + R.y e[1] + R.z e[2])/vol
	R.x =  r.x*(e[1].y*e[2].z - e[2].y*e[1].z) - r.y*(e[1].x*e[2].z - e[2].x*e[1].z) + r.z*(e[1].x*e[2].y - e[1].y*e[2].x);
	R.y = -r.x*(e[0].y*e[2].z - e[2].y*e[0].z) + r.y*(e[0].x*e[2].z - e[2].x*e[0].z) - r.z*(e[0].x*e[2].y - e[0].y*e[2].x);
	R.z =  r.x*(e[0].y*e[1].z - e[1].y*e[0].z) - r.y*(e[0].x*e[1].z - e[1].x*e[0].z) + r.z*(e[0].x*e[1].y - e[0].y*e[1].x);
	
	// now r = (R.x e[0] + R.y e[1] + R.z e[2]) + (Q.x e[0] + Q.y e[1] + Q.z e[2])/vol
	Q.x = R.x % vol;
  R.x /= vol;
  if(Q.x < 0) {
    Q.x += vol;
    R.x--;
  }

  Q.y = R.y % vol;
  R.y /= vol;
  if(Q.y < 0) {
    Q.y += vol;
    R.y--;
  }
	
  Q.z = R.z % vol;
  R.z /= vol;
  if(Q.z < 0) {
    Q.z += vol;
    R.z--;
  }
	
	S = e[0]*Q.x + e[1]*Q.y + e[2]*Q.z;
	S.x /= vol;
	S.y /= vol;
	S.z /= vol;
	Q = R;
	R = e[0]*Q.x + e[1]*Q.y + e[2]*Q.z;
  
  // then compensate between R and S to make the folding d-dimensional if d<3
  if(D < 3) {
    R -= e[2]*Q.z;
    S += e[2]*Q.z;
  }
  if(D < 2) {
    R -= e[1]*Q.y;
    S += e[1]*Q.y;
  }
}







/**
 Defines a default lattice in dimension D
 */

void lattice3D::trivial(){
	D = 3;
	e.push_back(vector3D<int64_t>(1,0,0));
	e.push_back(vector3D<int64_t>(0,1,0));
	e.push_back(vector3D<int64_t>(0,0,1));
	init();
}







/**
 from an integer vector V expressed in the working basis, provides as output its real components
 in the basis of the current superlattice
 @param V input vector
 */
vector3D<double> lattice3D::to(vector3D<int64_t> V)
{
	vector3D<double> W(V);
	W.transform(M);
	return W;
}






/**
 from a vector V expressed in the basis of the current superlattice, provides as output its components in the working basis
 @param V input vector
 */
vector3D<double> lattice3D::from(vector3D<double> V)
{
	vector3D<double> W;
	
	W.x = V.x*e[0].x + V.y*e[1].x + V.z*e[2].x;
	W.y = V.x*e[0].y + V.y*e[1].y + V.z*e[2].y;
	W.z = V.x*e[0].z + V.y*e[1].z + V.z*e[2].z;
	
	return W;
}







/**
 constructs a basis3D, 2PI dual of the basis of the current superlattice
 @param x reference to the basis to be constructed
 */
void lattice3D::dual(basis3D &x)
{
	double L1 = 2.0*M_PI/vol;
	basis3D D;
	
	x.e[0] = vector3D<double>(vector_product(e[1],e[2]))*L1;
	x.e[1] = vector3D<double>(vector_product(e[2],e[0]))*L1;
	x.e[2] = vector3D<double>(vector_product(e[0],e[1]))*L1;
}







std::ostream & operator<<(std::ostream &flux, lattice3D &latt){
	flux << "Dimension " << latt.D << endl;
	for(int i=0; i<latt.D; i++) flux << latt.e[i] << endl;
	flux << "lattice transformation matrix:\n" << latt.M << endl;
	return flux;
}






