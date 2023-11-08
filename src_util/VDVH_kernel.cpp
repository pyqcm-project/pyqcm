#include <VDVH_kernel.hpp>
#include <vector>
#include <cstring>
#include "types.hpp"

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/*
Some refs:
https://en.algorithmica.org/hpc/algorithms/matmul/
https://sci-hub.st/https://dl.acm.org/doi/10.1145/1356052.1356053
https://ia601407.us.archive.org/23/items/cnx-org-col11136/high-performance-computing.pdf
https://people.freebsd.org/~lstewart/articles/cpumemory.pdf
*/


// 
// NAIVE, ORIGINAL VERSION
//

//Double version
void VDVH_naive(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  for(size_t i=0; i<M; ++i){
    for(size_t a=0; a<L; ++a){
      Complex vai = V[a+i*L]*D[i];
      for(size_t b=0; b<L; ++b){
        G[b + a*L] += vai*V[b+i*L]; // original was G(a,b) but was wrong with complex operators
      }
    }
  }
}
//Complex version
void VDVH_naive(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M) {
  for(size_t i=0; i<M; ++i){
    for(size_t a=0; a<L; ++a){
      Complex vai = V[a+i*L]*D[i];
      for(size_t b=0; b<L; ++b){
        G[b + a*L] += vai*conjugate(V[b+i*L]); // original was G(a,b) but was wrong with complex operators
      }
    }
  }
}



#ifdef HAVE_AVX2

//
// KERNEL AVX2 version
//

// Double
#define KERNEL_HEIGH_D 4
#define KERNEL_WIDTH_D 2
void kernel_avx2(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_D][KERNEL_WIDTH_D] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  Complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_D; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm256_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real());
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_D; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = y+2*j + k*L;
        reg_temp2 = _mm256_set_pd(V[index+1],V[index+1],V[index],V[index]);
        //res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D; i++) {
      if (x+i > y+2*j+1) {
        _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2]; //lower triangle
        _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
        _G[2*(y+2*j+1 + (x+i)*L)] += res[i][j][2]; //upper triangle
        _G[2*(y+2*j+1 + (x+i)*L)+1] += res[i][j][3];
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //lower triangle
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
        _G[2*(y+2*j + (x+i)*L)] += res[i][j][0]; //upper triangle
        _G[2*(y+2*j + (x+i)*L)+1] += res[i][j][1];
      }
      else if (x+i == y+2*j+1) {
        _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2]; //diagonal
        _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //lower triangle
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
        _G[2*(y+2*j + (x+i)*L)] += res[i][j][0]; //upper triangle
        _G[2*(y+2*j + (x+i)*L)+1] += res[i][j][1];
      }
      else if (x+i == y+2*j) {
        _G[2*(x+i + (y+2*j)*L)] += res[i][j][0]; //diagonal
        _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
      }
    }
  }
}

void kernel_avx2_hor(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_WIDTH_D] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  Complex temp;
  
  for(int k=l; k<r; k++) {
    temp = V[x + k*L] * D[k];
    reg_temp = _mm256_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real());
    for (int j = 0; j < KERNEL_WIDTH_D; j++) {
      int index = y+2*j + k*L;
      reg_temp2 = _mm256_set_pd(V[index+1],V[index+1],V[index],V[index]);
      res[j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[j]);
    }
  }
  
  // write the results back to G considering symmetry
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    if (x > y+2*j+1) {
      _G[2*(x + (y+2*j+1)*L)] += res[j][2]; //lower triangle
      _G[2*(x + (y+2*j+1)*L)+1] += res[j][3];
      _G[2*(y+2*j+1 + x*L)] += res[j][2]; //upper triangle
      _G[2*(y+2*j+1 + x*L)+1] += res[j][3];
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //lower triangle
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
      _G[2*(y+2*j + x*L)] += res[j][0]; //upper triangle
      _G[2*(y+2*j + x*L)+1] += res[j][1];
    }
    else if (x == y+2*j+1) {
      _G[2*(x + (y+2*j+1)*L)] += res[j][2]; //diagonal
      _G[2*(x + (y+2*j+1)*L)+1] += res[j][3];
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //lower triangle
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
      _G[2*(y+2*j + x*L)] += res[j][0]; //upper triangle
      _G[2*(y+2*j + x*L)+1] += res[j][1];
    }
    else if (x == y+2*j) {
      _G[2*(x + (y+2*j)*L)] += res[j][0]; //diagonal
      _G[2*(x + (y+2*j)*L)+1] += res[j][1];
    }
  }
}


void VDVH_kernel_avx2(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  //G is LpadH * L
  const int LpadH = (L + 2*KERNEL_WIDTH_D-1) / (2*KERNEL_WIDTH_D) * (2*KERNEL_WIDTH_D);
  
  //padding the output matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(L * LpadH, 0.);
  
  //using the main kernel
  for (int x = 0; x <= L-KERNEL_HEIGH_D; x += KERNEL_HEIGH_D)
    for (int y = 0; y < x+KERNEL_HEIGH_D; y += 2*KERNEL_WIDTH_D)
      kernel_avx2(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
  
  //using the 1xKERNEL_WIDTH_D kernel to finish
  for (int x = L/KERNEL_HEIGH_D*KERNEL_HEIGH_D; x < L; x += 1)
    for (int y = 0; y <= x; y += 2*KERNEL_WIDTH_D)
      kernel_avx2_hor(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
  
  for (int i = 0; i < L*L; i++) G[i] += _G[i];//std::copy(_G.begin()+ i*L, _G.begin()+i*L+L, G.begin()+i*L);
  
  _G.resize(0);
}

// Complex
#define KERNEL_HEIGH_C 3
#define KERNEL_WIDTH_C 2
void kernel_avx2(Complex* G, const Complex* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_C][KERNEL_WIDTH_C] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  Complex temp;
  double* _V = (double*) V;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_C; i++) {
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm256_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real());
      for (int j = 0; j < KERNEL_WIDTH_C; j++) {
        int index = 2*(y+2*j + k*L);
        reg_temp2 = _mm256_set_pd(_V[index+2],_V[index+2],_V[index],_V[index]); //real part
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
      reg_temp = _mm256_set_pd(-temp.real(),temp.imag(),-temp.real(),temp.imag());
      for (int j = 0; j < KERNEL_WIDTH_C; j++) {
        int index = 2*(y+2*j + k*L);
        reg_temp2 = _mm256_set_pd(_V[index+3],_V[index+3],_V[index+1],_V[index+1]); //imag part (conjugate)
        res[i][j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  
  // write the results back to G
  for (int j = 0; j < KERNEL_WIDTH_C; j++) {
    for (int i = 0; i < KERNEL_HEIGH_C; i++) {
      G[(x+i) + (y+2*j)*L] = res[i][j][0] + Complex(0., res[i][j][1]);
      G[(x+i) + (y+2*j+1)*L] += res[i][j][2] + Complex(0., res[i][j][3]);
    }
  }
}

void kernel_avx2_hor(Complex* G, const Complex* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_WIDTH_C] = {0.}; //hold two complex type
  __m256d reg_temp = {0.}, reg_temp2 = {0.};
  Complex temp;
  double* _V = (double*) V;
  
  for(int k=l; k<r; k++) { 
    temp = V[x + k*L] * D[k];
    reg_temp = _mm256_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real());
    for (int j = 0; j < KERNEL_WIDTH_C; j++) {
      int index = 2*(y+2*j + k*L);
      reg_temp2 = _mm256_set_pd(_V[index+2],_V[index+2],_V[index],_V[index]); //real part
      res[j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[j]);
    }
    reg_temp = _mm256_set_pd(-temp.real(),temp.imag(),-temp.real(),temp.imag());
    for (int j = 0; j < KERNEL_WIDTH_C; j++) {
      int index = 2*(y+2*j + k*L);
      reg_temp2 = _mm256_set_pd(_V[index+3],_V[index+3],_V[index+1],_V[index+1]); //imag part (conjugate)
      res[j] = _mm256_fmadd_pd(reg_temp2, reg_temp, res[j]);
    }
  }
  
  // write the results back to G
  for (int j = 0; j < KERNEL_WIDTH_C; j++) {
    G[x + (y+2*j)*L] = res[j][0] + Complex(0.,res[j][1]);
    G[x + (y+2*j+1)*L] += res[j][2] + Complex(0., res[j][3]);
  }
}

void VDVH_kernel_avx2(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M) {
  //G is LpadH * L
  const int LpadH = (L + 2*KERNEL_WIDTH_C-1) / (2*KERNEL_WIDTH_C) * (2*KERNEL_WIDTH_C);
  
  //padding the output matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(L * LpadH, 0.);
  
  //using the main kernel
  for (int x = 0; x <= L-KERNEL_HEIGH_C; x += KERNEL_HEIGH_C)
    for (int y = 0; y < x+KERNEL_HEIGH_C; y += 2*KERNEL_WIDTH_C)
      kernel_avx2(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
  
  //using the 1xKERNEL_WIDTH_C kernel to finish
  for (int x = L/KERNEL_HEIGH_C*KERNEL_HEIGH_C; x < L; x += 1)
    for (int y = 0; y <= LpadH; y += 2*KERNEL_WIDTH_C)
      kernel_avx2_hor(_G.data(), V.data(), D.data(), x, y, 0, M, M, L);
  
  for (int i = 0; i < L*L; i++) G[i] += _G[i];
  
  _G.resize(0);
}

#endif



