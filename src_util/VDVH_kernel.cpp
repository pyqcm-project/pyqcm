#include <VDVH_kernel.hpp>
#include <immintrin.h>
#include <vector>
#include <cstring>
#include "types.hpp"


/*
How I designed this kernel ?
I used AVX2 processor which contained 16 vector registers (256 bits)
I would like to maximize their use.
https://en.algorithmica.org/hpc/algorithms/matmul/
TODO: I did not use the fact that G is symetric
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





//
// KERNEL AVX2 version
//

// Double
#define KERNEL_HEIGH_D 3
#define KERNEL_WIDTH_D 3
void kernel_avx2(__restrict__ Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res[KERNEL_HEIGH_D][KERNEL_WIDTH_D]{}; //hold two complex type
  __m256d reg_temp, reg_temp2;
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
  
  // write the results back to G
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D; i++) { //heigh inner loop becase G is column oriented
      _G[2*(x+i + (y+2*j)*L)] += res[i][j][0];
      _G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
      _G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2];
      _G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
    }
  }
}


/*
#define KERNEL_HEIGH_D 3
#define KERNEL_WIDTH_D 3
//unrool loops by hand to count register
//this give an gcc bug when optimizing with -O2 and -funroll-loops (res01 and res10 wrong)
void kernel_avx2(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m256d res00, res01, res02, //each result register contains 2 complex
          res10, res11, res12,
          res20, res21, res22;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    __m256d D_k = _mm256_set_pd(D[k].imag(),D[k].real(),D[k].imag(),D[k].real());
    //__m256d D_k = _mm256_broadcast_pd((double*) &D[k]); //broacast D[k]
    
    //compute V(x+i,k) * D(k)
    __m256d VD0, VD1, VD2; 
    VD0 = _mm256_mul_pd(_mm256_broadcast_sd(&V[x+k*L]), D_k);
    VD1 = _mm256_mul_pd(_mm256_broadcast_sd(&V[x+1+k*L]), D_k);
    VD2 = _mm256_mul_pd(_mm256_broadcast_sd(&V[x+2+k*L]), D_k);
    //at this point, we can even reuse D_k
    
    //Store V^T(k,y+j)
    __m256d VT0, VT1, VT2; 
    VT0 = _mm256_set_pd(V[y + k*L +1],V[y + k*L +1],V[y + k*L],V[y + k*L]);
    VT1 = _mm256_set_pd(V[y+2 + k*L +1],V[y+2 + k*L +1],V[y+2 + k*L],V[y+2 + k*L]);
    VT2 = _mm256_set_pd(V[y+4 + k*L +1],V[y+4 + k*L +1],V[y+4 + k*L],V[y+4 + k*L]);
    
    //update result
    res00 = _mm256_fmadd_pd(VD0, VT0, res00);
    res01 = _mm256_fmadd_pd(VD0, VT1, res01);
    res02 = _mm256_fmadd_pd(VD0, VT2, res02);
    res10 = _mm256_fmadd_pd(VD1, VT0, res10);
    res11 = _mm256_fmadd_pd(VD1, VT1, res11);
    res12 = _mm256_fmadd_pd(VD1, VT2, res12);
    res20 = _mm256_fmadd_pd(VD2, VT0, res20);
    res21 = _mm256_fmadd_pd(VD2, VT1, res21);
    res22 = _mm256_fmadd_pd(VD2, VT2, res22);
  }
  
  // write the results back to G
  double* _G = (double*) G;
  //_G[2*(x+i + (y+2*j)*L)] += res[i][j][0];
  //_G[2*(x+i + (y+2*j)*L)+1] += res[i][j][1];
  //_G[2*(x+i + (y+2*j+1)*L)] += res[i][j][2];
  //_G[2*(x+i + (y+2*j+1)*L)+1] += res[i][j][3];
  //i=0, j=0
  _G[2*(x + y*L)]       += res00[0]; _G[2*(x + y*L)+1]       += res00[1];
  _G[2*(x + (y+1)*L)]   += res00[2]; _G[2*(x + (y+1)*L)+1]   += res00[3];
  //i=1, j=0
  _G[2*(x+1 + y*L)]     += res10[0]; _G[2*(x+1 + y*L)+1]     += res10[1];
  _G[2*(x+1 + (y+1)*L)] += res10[2]; _G[2*(x+1 + (y+1)*L)+1] += res10[3];
  //i=2, j=0
  _G[2*(x+2 + y*L)]     += res20[0]; _G[2*(x+2 + y*L)+1]     += res20[1];
  _G[2*(x+2 + (y+1)*L)] += res20[2]; _G[2*(x+2 + (y+1)*L)+1] += res20[3];
  
  //i=0, j=1
  _G[2*(x + (y+2)*L)]   += res01[0]; _G[2*(x + (y+2)*L)+1]   += res01[1];
  _G[2*(x + (y+3)*L)]   += res01[2]; _G[2*(x + (y+3)*L)+1]   += res01[3];
  //i=1, j=1
  _G[2*(x+1 + (y+2)*L)] += res11[0]; _G[2*(x+1 + (y+2)*L)+1] += res11[1];
  _G[2*(x+1 + (y+3)*L)] += res11[2]; _G[2*(x+1 + (y+3)*L)+1] += res11[3];
  //i=2, j=1
  _G[2*(x+2 + (y+2)*L)] += res21[0]; _G[2*(x+2 + (y+2)*L)+1] += res21[1];
  _G[2*(x+2 + (y+3)*L)] += res21[2]; _G[2*(x+2 + (y+3)*L)+1] += res21[3];
  
  //i=0, j=2
  _G[2*(x + (y+4)*L)]   += res02[0]; _G[2*(x + (y+4)*L)+1]   += res02[1];
  _G[2*(x + (y+5)*L)]   += res02[2]; _G[2*(x + (y+5)*L)+1]   += res02[3];
  //i=1, j=2
  _G[2*(x+1 + (y+4)*L)] += res12[0]; _G[2*(x+1 + (y+4)*L)+1] += res12[1];
  _G[2*(x+1 + (y+5)*L)] += res12[2]; _G[2*(x+1 + (y+5)*L)+1] += res12[3];
  //i=2, j=2
  _G[2*(x+2 + (y+4)*L)] += res22[0]; _G[2*(x+2 + (y+4)*L)+1] += res22[1];
  _G[2*(x+2 + (y+5)*L)] += res22[2]; _G[2*(x+2 + (y+5)*L)+1] += res22[3];
  
}
*/

void VDVH_kernel_avx2(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  //note Mpad is the size of the inner padded matrix
  //G is LpadH * LpadV
  const int LpadH = (L + 2*KERNEL_WIDTH_D-1) / (2*KERNEL_WIDTH_D) * (2*KERNEL_WIDTH_D);
  const int LpadV = (L + KERNEL_HEIGH_D-1) / KERNEL_HEIGH_D * KERNEL_HEIGH_D;
  const int Mpad = (M + KERNEL_HEIGH_D-1) / KERNEL_HEIGH_D * KERNEL_HEIGH_D;
  
  //padding the input matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(LpadV * LpadH);
  std::vector<double> _V; _V.resize(LpadV * Mpad);
  std::vector<Complex> _D; _D.resize(Mpad);
  
  std::copy(D.begin(), D.end(), _D.begin());
  for (int i = 0; i < M; i++) {
    std::copy(V.begin() + i*L, V.begin() + (i+1)*L, _V.begin() + i * LpadV);
  }
  
  for (int x = 0; x < LpadV; x += KERNEL_HEIGH_D)
    for (int y = 0; y < LpadH; y += 2*KERNEL_WIDTH_D)
      kernel_avx2(_G.data(), _V.data(), _D.data(), x, y, 0, M, Mpad, LpadV);
 
  for (int i = 0; i < L; i++) std::copy(_G.begin()+ i*LpadV, _G.begin()+i*LpadV+L, G.begin()+i*L);
  
  _G.resize(0); _V.resize(0); _D.resize(0);
}

// Complex
#define KERNEL_HEIGH_C 0
#define KERNEL_WIDTH_C 0
void VDVH_kernel_avx2(std::vector<Complex> &G, std::vector<Complex> &V, std::vector<Complex> &D, const int L, const int M){};
void kernel_avx2(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L) {};





//
// KERNEL AVX512 version
//

// Double version
#define KERNEL_HEIGH_D_AVX512 5
#define KERNEL_WIDTH_D_AVX512 4
void kernel_avx512(__restrict__ Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L)
{
  __m512d res[KERNEL_HEIGH_D_AVX512][KERNEL_WIDTH_D_AVX512]{}; //hold four complex type
  __m512d reg_temp, reg_temp2;
  Complex temp;
  
  for(int k=l; k<r; k++) { //k inner dim to reduce (V column, square size of D)
    //loops must be unrooled
    for (int i = 0; i<KERNEL_HEIGH_D_AVX512; i++) {
      //broadcast lines of V(x+i,k) * D(k) into a register
      temp = V[x+i + k*L] * D[k];
      reg_temp = _mm512_set_pd(temp.imag(),temp.real(),temp.imag(),temp.real(),temp.imag(),temp.real(),temp.imag(),temp.real()); //TODO
      //now multiply the temp register by column of B
      for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
        //we should take indice V^T(k,y+j) and V^T(k,y+j+1)
        //so for V:             V(y+j,k)   and V(y+j+1,k)
        int index = y+4*j + k*L;
        reg_temp2 = _mm512_set_pd(V[index+3],V[index+3],V[index+2],V[index+2],V[index+1],V[index+1],V[index],V[index]);
        //res[i][j] += reg_temp2 * reg_temp; // as a vec register and FMA
        res[i][j] = _mm512_fmadd_pd(reg_temp2, reg_temp, res[i][j]);
      }
    }
  }
  // write the results back to G
  double* _G = (double*) G;
  for (int j = 0; j < KERNEL_WIDTH_D_AVX512; j++) {
    for (int i = 0; i < KERNEL_HEIGH_D_AVX512; i++) { //heigh inner loop becase G is column oriented
      _G[2*(x+i + (y+4*j)*L)] += res[i][j][0];
      _G[2*(x+i + (y+4*j)*L)+1] += res[i][j][1];
      _G[2*(x+i + (y+4*j+1)*L)] += res[i][j][2];
      _G[2*(x+i + (y+4*j+1)*L)+1] += res[i][j][3];
      _G[2*(x+i + (y+4*j+2)*L)] += res[i][j][4];
      _G[2*(x+i + (y+4*j+2)*L)+1] += res[i][j][5];
      _G[2*(x+i + (y+4*j+3)*L)] += res[i][j][6];
      _G[2*(x+i + (y+4*j+3)*L)+1] += res[i][j][7];
    }
  }
}

void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M) {
  //note Mpad is the size of the inner padded matrix
  //G is LpadH * LpadV
  const int LpadH = (L + 4*KERNEL_WIDTH_D_AVX512-1) / (4*KERNEL_WIDTH_D_AVX512) * (4*KERNEL_WIDTH_D_AVX512);
  const int LpadV = (L + KERNEL_HEIGH_D_AVX512-1) / KERNEL_HEIGH_D_AVX512 * KERNEL_HEIGH_D_AVX512;
  const int Mpad = (M + KERNEL_HEIGH_D_AVX512-1) / KERNEL_HEIGH_D_AVX512 * KERNEL_HEIGH_D_AVX512;
  
  //padding the input matrix to fit the kernel (to remove later)
  std::vector<Complex> _G; _G.resize(LpadV * LpadH);
  std::vector<double> _V; _V.resize(LpadV * Mpad);
  std::vector<Complex> _D; _D.resize(Mpad);
  
  std::copy(D.begin(), D.end(), _D.begin());
  for (int i = 0; i < M; i++) {
    std::copy(V.begin() + i*L, V.begin() + (i+1)*L, _V.begin() + i * LpadV);
  }
  
  for (int x = 0; x < LpadV; x += KERNEL_HEIGH_D_AVX512)
    for (int y = 0; y < LpadH; y += 4*KERNEL_WIDTH_D_AVX512)
      kernel_avx512(_G.data(), _V.data(), _D.data(), x, y, 0, M, Mpad, LpadV);
 
  for (int i = 0; i < L; i++) std::copy(_G.begin()+ i*LpadV, _G.begin()+i*LpadV+L, G.begin()+i*L);
  
  _G.resize(0); _V.resize(0); _D.resize(0);
}

// Complex version
#define KERNEL_HEIGH_C_AVX512 0
#define KERNEL_WIDTH_C_AVX512 0
void VDVH_kernel_avx512(std::vector<Complex> &G, std::vector<Complex> &V, std::vector<Complex> &D, const int L, const int M){};
void kernel_avx512(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L) {};

