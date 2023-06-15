#ifndef types_h
#define types_h


#include <complex>


typedef std::complex<double> Complex;


inline double conjugate(const double x) { return x;}
inline Complex conjugate(const Complex x) { return conj(x);}

inline double realpart(const double x) { return x;}
inline double realpart(const Complex x) { return real(x);}

template<typename HS_field, typename op_field>
HS_field fold_type(op_field x);

template<>
inline double fold_type(double x) {return x;}
template<>
inline Complex fold_type(Complex x) {return x;}
template<>
inline Complex fold_type(double x) {return x;}
template<>
inline double fold_type(Complex x) {return real(x);}

struct run_statistics{
    size_t n_GF_recycled = 0L; // number of times the GF is recycled 
    size_t n_GF_computed = 0L; // number of times the GF is computed
};

#endif /* types_h */
