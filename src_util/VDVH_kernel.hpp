#ifndef VDVH_kernel_h
#define VDVH_kernel_h

#include "types.hpp"
#include <vector>


//Naive version
void VDVH_naive(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M);
void VDVH_naive(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M);


#ifdef HAVE_AVX2
// AVX2 VERSION
void VDVH_kernel_avx2(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx2(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

void VDVH_kernel_avx2(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx2(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L);
#endif

#if 0
// AVX512 VERSION
void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<double> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx512(Complex* G, const double* V, const Complex* D, const int x, const int y, const int l, const int r, const int M, const int L);

void VDVH_kernel_avx512(std::vector<Complex> &G, const std::vector<Complex> &V, const std::vector<Complex> &D, const int L, const int M);
void kernel_avx512(void* G, Complex* V, Complex* D, int x, int y, int l, int r, int M, int L);
#endif



#endif
