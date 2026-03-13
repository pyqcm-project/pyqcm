#ifndef matrix_h
#define matrix_h
#include <iostream>

#include <complex>
#include <cstring>
#include <ctype.h>

#include "parser.hpp"
#include "vector_num.hpp"

using namespace std;

#define matrix_SMALL_VALUE 1e-10
//#define BOUND_CHECK

//! A general matrix class (of ints, double and Complex types).
template <typename T> struct matrix {
  size_t r;    //!< number of rows
  size_t c;    //!< number of columns
  vector<T> v; //!< array of values (column ordered)

  auto data() ->decltype(v.data()) { return v.data(); }
  auto size() ->decltype(v.size()) { return v.size(); }

  //! default constructor (private)
  matrix() : r(0), c(0) {}

  //! constructor with a given size (square matrix)
  matrix(size_t _r) : r(_r), c(_r) { v.resize(r * c); }

  //! constructor with a given size (non square)
  matrix(size_t _r, size_t _c) : r(_r), c(_c) { v.resize(r * c); }

  //! constructor with a given size from a vector
  matrix(const size_t _r, const size_t _c, const vector<T> &_v) : r(_r), c(_c), v(_v) {}

  //! constructor with a given size from a vector
  matrix(const size_t _r, const vector<T> &_v) : r(_r), c(_r), v(_v) {}

  //! deep copy constructor
  matrix(const matrix<T> &A) : r(A.r), c(A.c), v(A.v) {}

  //! concatenation constructor (puts the matrix side by side)
  matrix(const matrix<T> &A, const matrix<T> &B) : r(A.r), c(A.c + B.c) {
    v.resize(A.v.size() + B.v.size());
    copy(A.v.data(), A.v.data() + A.v.size(), v.data());
    copy(B.v.data(), B.v.data() + B.v.size(), &v[A.v.size()]);
  }


  //! assignment operator
  template <typename S> matrix<T> &assign(const matrix<S> &A) {
    if ((void *)this == (void *)&A)
      return *this;
    else if (v.size() == 0) {
      set_size(A.r, A.c);
      for (size_t i = 0; i < A.v.size(); ++i) v[i] = A.v[i];
    } else if (r == A.r and c == A.c) {
      for (size_t i = 0; i < A.v.size(); ++i) v[i] = A.v[i];
    } else if (v.size() > A.v.size()) {
      for (size_t i = 0; i < A.r; ++i) {
        for (size_t j = 0; j < A.c; ++j) { v[i + r * j] = A.v[i + A.r * j]; }
      }
    } else {
      for (size_t i = 0; i < r; ++i) {
        for (size_t j = 0; j < c; ++j) { v[i + r * j] = A.v[i + A.r * j]; }
      }
    }
    return *this;
  }



  //! element access (for rhs)
  inline const T &operator()(const size_t &i, const size_t &j) const { return (v[i + r * j]); }

  //! element access (for lhs)
  T &operator()(const size_t &i, const size_t &j) { return (v[i + r * j]); }

  //! concatenation (puts the matrix side by side)
  void concatenate(const matrix<T> &A, const matrix<T> &B) {
    set_size(A.r, A.c + B.c);
    copy(&A.v[0], &A.v[0] + A.v.size(), &v[0]);
    copy(&B.v[0], &B.v[0] + B.v.size(), &v[A.v.size()]);
  }


  //! extracts a square submatrix A, starting at row i
  void sub_matrix(size_t i, matrix<T> &A) {
    QCM_ASSERT(r == c and A.r == A.c and i + A.r <= r);
    for (size_t k = 0; k < A.r; ++k) {
      for (size_t l = 0; l < A.r; ++l) { A(k, l) = v[i + k + r * (i + l)]; }
    }
  }



  //! extracts a rectangular submatrix of size (R,C), starting at (i,j) of this, and copies it to position (I, J) of A
  template <typename S>
  void move_sub_matrix(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0) {
    QCM_ASSERT(j + C <= c and i + R <= r and I + R <= A.r and J + C <= A.c);
    for (size_t k = 0; k < R; ++k) {
      for (size_t l = 0; l < C; ++l) { A(I + k, J + l) += z * v[i + k + r * (j + l)]; }
    }
  }

  //! extracts a rectangular submatrix of size (R,C), starting at (i,j) of this, and copies its complex conjugate to position (I,
  //! J) of A
  template <typename S>
  void move_sub_matrix_conjugate(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0) {
    QCM_ASSERT(j + C <= c and i + R <= r and I + R <= A.r and J + C <= A.c);
    for (size_t k = 0; k < R; ++k) {
      for (size_t l = 0; l < C; ++l) { A(I + k, J + l) += z * conjugate(v[i + k + r * (j + l)]); }
    }
  }


  //! extracts a rectangular submatrix of size (R,C), starting at (i,j) of this, and copies its hermitian conjugate to position
  //! (I, J) of A
  template <typename S>
  void move_sub_matrix_HC(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0) {
    QCM_ASSERT(j + C <= c and i + R <= r and I + C <= A.r and J + R <= A.c);
    for (size_t k = 0; k < R; ++k) {
      for (size_t l = 0; l < C; ++l) { A(I + l, J + k) += z * conjugate(v[i + k + r * (j + l)]); }
    }
  }


  //! extracts a rectangular submatrix of size (R,C), starting at (i,j) of this, and copies its transpose to position (I, J) of A
  template <typename S>
  void move_sub_matrix_transpose(size_t R, size_t C, size_t i, size_t j, size_t I, size_t J, matrix<S> &A, double z = 1.0) {
    QCM_ASSERT(j + C <= c and i + R <= r and I + C <= A.r and J + R <= A.c);
    for (size_t k = 0; k < R; ++k) {
      for (size_t l = 0; l < C; ++l) { A(I + l, J + k) += z * v[i + k + r * (j + l)]; }
    }
  }



  //! inserts A as the ith row of the matrix
  void insert_row(size_t i, const vector<T> &A) {
    for (size_t j = 0; j < c; j++) v[i + r * j] = A[j];
  }



  //! inserts A as the ith column of the matrix
  void insert_column(size_t i, const vector<T> &A) {
    for (size_t j = 0; j < r; j++) v[j + r * i] = A[j];
  }



  //! extracts a row into a vector
  void extract_row(size_t i, vector<T> &A) {
    for (size_t j = 0; j < c; j++) A[j] = v[i + r * j];
  }

  //! extracts a row
  vector<T> extract_row(size_t i) {
    vector<T> A(c);
    for (size_t j = 0; j < c; j++) A[j] = v[i + r * j];
    return A;
  }


  //! extracts a column into a vector
  void extract_column(size_t i, vector<T> &A) { copy(&v[i * r], &v[i * r] + r, &A[0]); }

  vector<T> extract_column(size_t i) {
    vector<T> A(r);
    copy(&v[i * r], &v[i * r] + r, &A[0]);
    return A;
  }


  //! sets to zero
  inline void zero() { to_zero(v); }


  //! sets to zero the components addressed by the mask
  void apply_mask(const vector<size_t> &mask) {
    for (auto &i : mask) v[i] = 0.0;
  }


  //! sets size and allocate
  inline void set_size(size_t _r, size_t _c) {
    r = _r;
    c = _c;
    v.clear();
    v.resize((size_t)(r * c));
  }
  inline void set_size(size_t _r) { set_size(_r, _r); }
  template <typename S> inline void set_size(matrix<S> mat) { set_size(mat.r, mat.c); }


  //! imports data to already allocated matrix
  inline void import(T *x) { copy(x, v.size(), &v[0]); }



  //! takes the real part of another matrix
  void real_part(const matrix<Complex> &A) {
    for (size_t i = 0; i < v.size(); i++) v[i] = real(A.v[i]);
  }


  //! initializes to the identity matrix
  void identity() {
    zero();
    for (size_t i = 0; i < r; i++) v[i + i * r] = T(1.0);
  }


  //! sets the diagonal of A equal to the vector x
  template <typename S> inline void diagonal(const vector<S> &x) {
    for (size_t i = 0; i < r; ++i) v[i + i * r] = x[i];
  }


  //! adds a constant to the diagonal
  template <typename S> inline void add_to_diagonal(const S x) {
    for (size_t i = 0; i < r; ++i) v[i + i * r] += x;
  }

  //! adds a vector to the diagonal
  template <typename S> inline void add_to_diagonal(const vector<S> &x) {
    for (size_t i = 0; i < r; ++i) v[i + i * r] += x[i];
  }


  //! replaces the matrix by its transpose
  void transpose() {
    vector<T> tmp(v);
    swap(r, c);
    for (size_t i = 0; i < r; ++i)
      for (int j = 0; j < c; ++j) v[i + r * j] = tmp[j + c * i];
  }


  //! replaces the matrix by its Hermitean conjugate
  void hermitian_conjugate() {
    vector<T> tmp(v);
    swap(r, c);
    for (size_t i = 0; i < r; ++i)
      for (size_t j = 0; j < c; ++j) v[i + r * j] = conjugate(tmp[j + c * i]);
  }


  //! computes the trace of a matrix
  T trace() {
    T z(0.0);
    for (size_t i = 0; i < r; i++) z += v[i + r * i];
    return z;
  }


  //! computes the partial trace of a matrix
  T trace(size_t start_row, size_t end_row) {
    T z(0.0);
    for (size_t i = start_row; i < end_row; i++) z += v[i + r * i];
    return z;
  }


  //! computes the trace of the product with a matrix A
  template <typename S> T trace_product(matrix<S> A, bool transpose = false) {
    T z(0.0);
    if (transpose) {
      QCM_ASSERT(A.r == r and A.c == c);
      for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) { z += v[i + r * j] * A.v[i + r * j]; }
      }
    } else {
      QCM_ASSERT(A.r == c and A.c == r);
      for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) { z += v[i + r * j] * A.v[j + c * i]; }
      }
    }
    return z;
  }



  //! substraction of a number times the unit matrix
  template <typename S> matrix<T> &operator-=(const S &a) {
    for (size_t i = 0; i < r; i++) v[i + r * i] -= a;
    return *this;
  }



  //! substraction of a matrix
  template <typename S> matrix<T> &operator-=(const matrix<S> &A) {
    for (size_t i = 0; i < v.size(); i++) v[i] = v[i] - A.v[i];
    return *this;
  }

  //! addition of a number times the unit matrix
  template <typename S> matrix<T> &operator+=(const S &a) {
    for (size_t i = 0; i < r; i++) v[i + r * i] += a;
    return *this;
  }



  //! addition of a matrix
  template <typename S> matrix<T> &operator+=(const matrix<S> &A) {
    for (size_t i = 0; i < v.size(); i++) v[i] = v[i] + A.v[i];
    return *this;
  }



  //! calculates the squared difference between a matrix and this: |this -A|^2
  double diff_sq(const matrix<T> &A, double a = 1.0) {
#ifdef BOUND_CHECK
    QCM_ASSERT(r == A.r and c == A.c);
#endif
    double z = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
      double zz = abs(v[i] - A.v[i] * a);
      z += zz * zz;
    }
    return (z);
  }



  //! multiplies two matrices : this = A * B
  template <typename S, typename U> void product(const matrix<S> &A, const matrix<U> &B) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and B.c == c and A.c == B.r);
#endif
    QCM_ASSERT(&A != this and &B != this);
    for (size_t j = 0; j < c; j++) {
      size_t jj = j * B.r;
      for (size_t i = 0; i < r; ++i) {
        T z = 0.0;
        for (size_t k = 0; k < A.c; ++k) z += A.v[i + k * r] * B.v[k + jj];
        v[i + j * r] = z;
      }
    }
  }


  //! multiplies two matrices : this = A * B with a bound on a the intermediate index 
  template <typename S, typename U> void product(const matrix<S> &A, const matrix<U> &B, size_t M) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and B.c == c and A.c == B.r);
#endif
    if(M>A.c) M = A.c;
    QCM_ASSERT(&A != this and &B != this);
    for (size_t j = 0; j < c; j++) {
      size_t jj = j * B.r;
      for (size_t i = 0; i < r; ++i) {
        T z = 0.0;
        for (size_t k = 0; k < M; ++k) z += A.v[i + k * r] * B.v[k + jj];
        v[i + j * r] = z;
      }
    }
  }





  //! multiplies two matrices : this = A * transpose(B)
  template <typename S, typename U> void product_transpose(const matrix<S> &A, const matrix<U> &B) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and B.r == c and A.c == B.c);
#endif
    QCM_ASSERT(&A != this and &B != this);
    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z = 0.0;
        for (size_t k = 0; k < A.c; ++k) z += A.v[i + k * r] * B.v[j + k * r];
        v[i + j * r] = z;
      }
    }
  }



  //! multiplies two matrices : this = A * hermitian(B)
  template <typename S, typename U> void product_hermitian_conjugate(const matrix<S> &A, const matrix<U> &B) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and B.r == c and A.c == B.c);
#endif
    QCM_ASSERT(&A != this and &B != this);
    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z = 0.0;
        for (size_t k = 0; k < A.c; ++k) z += A.v[i + k * r] * conjugate(B.v[j + k * r]);
        v[i + j * r] = z;
      }
    }
  }



  //! adds a number times the identity matrix
  void add(T d) {
    for (size_t i = 0; i < r; i++) v[i + i * r] += d;
  }

  //! adds a number times another matrix
  template <typename S, typename B> void add(const matrix<S> &A, B d) {
    for (size_t i = 0; i < v.size(); i++) v[i] += d * A.v[i];
  }

  //! performs a similarity transformation : this = hermitian(A)  * this * A
  template <typename S> void simil(const matrix<S> &A) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == c and r == c);
#endif
    T *w;
    w = new T[r * c];
    memcpy(w, v.data(), r * c * sizeof(T));

    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z(0);
        for (size_t k = 0; k < c; ++k) {
          T zz(0);
          for (size_t l = 0; l < c; ++l) zz += A(l, j) * w[k + r * l];
          z += zz * conjugate(A(k, i));
        }
        v[i + j * r] = z;
      }
    }
    delete[] w;
  }

  //! performs a similarity transformation : this = A * this * hermitian(A)
  template <typename S> void simil_inv(const matrix<S> &A) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.c == r and r == c);
#endif
    T *w;
    w = new T[r * c];
    memcpy(w, v.data(), r * c * sizeof(T));

    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z(0);
        for (size_t k = 0; k < c; ++k) {
          T zz(0);
          for (size_t l = 0; l < c; ++l) zz += conjugate(A(j, l)) * w[k + r * l];
          z += zz * A(i, k);
        }
        v[i + j * r] = z;
      }
    }
    delete[] w;
  }

  //! performs a similarity transformation : this = transpose(A)  * this * A
  template <typename S> void simil_transpose(const matrix<S> &A) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and A.c == c);
#endif
    T *w;
    w = new T[r * c];
    memcpy(w, v.v, r * c * sizeof(T));

    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z(0);
        for (size_t k = 0; k < c; ++k) {
          T zz(0);
          for (size_t l = 0; l < c; ++l) zz += A(l, j) * w[k + r * l];
          z += zz * A(k, i);
        }
        v[i + j * r] = z;
      }
    }
    delete[] w;
  }

  //! performs a similarity transformation : this = hermitian(A) * this * conjugate(A)
  template <typename S> void simil_conjugate(const matrix<S> &A) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == r and A.c == c);
#endif
    T *w;
    w = new T[r * c];
    memcpy(w, v.v, r * c * sizeof(T));

    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z(0);
        for (size_t k = 0; k < c; ++k) {
          T zz(0);
          for (size_t l = 0; l < c; ++l) zz += conjugate(A(l, j)) * w[k + r * l];
          z += zz * conjugate(A(k, i));
        }
        v[i + j * r] = z;
      }
    }
    delete[] w;
  }

  //! performs a similarity transformation : this = hermitian(A) * B * A
  template <typename S1, typename S2> void simil(const matrix<S1> &A, const matrix<S2> &B) {
#ifdef BOUND_CHECK
    QCM_ASSERT(A.r == B.r and B.c == B.r and A.c == r);
#endif
    for (size_t j = 0; j < c; j++) {
      for (size_t i = 0; i < r; ++i) {
        T z = 0.0;
        for (size_t k = 0; k < B.c; ++k) {
          T zz = 0.0;
          for (size_t l = 0; l < B.c; ++l) zz += A(l, j) * B(k, l);
          z += zz * conjugate(A(k, i));
        }
        v[i + j * r] += z;
      }
    }
  }


  //! Application on an array : y += this * x
  void apply_add(T *x, T *y) {
    for (size_t j = 0; j < c; ++j) {
      for (size_t i = 0; i < r; ++i) y[i] += v[i + j * r] * x[j];
    }
  }

  void apply_add(const vector<T> &x, vector<T> &y) const {
    for (size_t j = 0; j < c; ++j) {
      for (size_t i = 0; i < r; ++i) y[i] += v[i + j * r] * x[j];
    }
  }


  //! Application on an array : y = this * x
  void apply(T *x, T *y) {
    memset(y, 0, r * sizeof(*y));
    apply_add(x, y);
  }

  void apply(const vector<T> &x, vector<T> &y) {
    to_zero(y);
    apply_add(x, y);
  }


  //! Application on an array : y += conj(x) * this
  void left_apply_add(const vector<T> &x, vector<T> &y) const {
    for (size_t j = 0; j < c; ++j) {
      for (size_t i = 0; i < r; ++i) y[j] += v[i + j * r] * conjugate(x[i]);
    }
  }


  //! checks whether the matrix is diagonal
  bool is_diagonal(double accuracy = 1.0e-8) {
    QCM_ASSERT(r == c);
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < c; ++j) {
        if (abs(v[i + r * j]) > accuracy and i != j) return false;
      }
    }
    return true;
  }

  //! takes the complex conjugate
  inline void cconjugate() { conjugate(v); }


  //! quadratic form
  T qform(vector<T> &x, vector<T> &y) {
    vector<T> w(r);
    apply(y, w);
    return x * w;
  }

  bool printable() {
    size_t max_dim_print = global_int("max_dim_print");
    if (r > max_dim_print or c > max_dim_print) return false;
    return true;
  }

  //! returns true if the matrix is unitary or orthogonal
  bool is_unitary(double threshold) {
    if (r != c) return false;
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j <= i; ++j) {
        T z(0);
        for (size_t k = 0; k < r; ++k) { z += conjugate(v[k + i * r]) * v[k + j * r]; }
        if (i == j and abs(z - 1.0) > threshold) return false;
        if (i != j and abs(z) > threshold) return false;
      }
    }
    return true;
  }



  //! returns true if the matrix is hermitian or real symmetric
  bool is_hermitian(double threshold=1e-10) {
    if (r != c) return false;
    for (size_t i = 0; i < r; ++i) {
      if (imag(v[i + i * r]) > threshold) return false;
      for (size_t j = 0; j < i; ++j) {
        if (abs(v[i + j * r] - conjugate(v[j + i * r])) > threshold) return false;
      }
    }
    return true;
  }



  /**
   Performs the Gram-Schmidt procedure on an input vector x to produce the current matrix, with the vector x located
   in the column associated with its largest value
   */
  bool Gram_Schmidt(const vector<T> &x, size_t col) {
    QCM_ASSERT(r == c and x.size() == r);

    vector<vector<T>> A(r);
    A[0] = x;
    if (!normalize(A[0])) return false;
    insert_column(col, A[0]);

    size_t k = 1;
    for (size_t i = 0; i < r; i++) {
      if (i == col) continue;
      A[k].resize(r);
      A[k][i] = 1.0;
      for (size_t j = 0; j < k; j++) {
        T z = A[j][i]; // A[j]*A[k]
        mult_add(-z, A[j], A[k]);
      }
      if (!normalize(A[k])) return false;
      insert_column(i, A[k]);
      k++;
    }
    if (!is_unitary(1.0e-6)) return false;
    return true;
  }



  /**
   Builds the current matrix as $M=(1+A)(1-A)^(-1)$
   */
  void Cayley(const matrix<T> &A) {
    zero();
    QCM_ASSERT(r == c and A.r == A.c and r == A.r);
    matrix<T> X(A);
    X.add(-1.0);
    X.v *= -1.0;
    X.inverse();
    matrix<T> Y(A);
    Y.add(1.0);
    product(Y, X);
  }



  /**
   Builds the current matrix as $A=(1+M)^(-1)(M-1)$
   */
  void Cayley_inverse(const matrix<T> &M) {
    zero();
    QCM_ASSERT(r == c and M.r == M.c and r == M.r);
    matrix<T> X(M);
    X.add(1.0);
    X.inverse();
    matrix<T> Y(M);
    Y.add(-1.0);
    product(X, Y);
  }



  /**
   Replaces the current matrix by its antihermitian (antisymmetric) part
   */
  void antisymmetrize() {
    for (size_t i = 0; i < r; i++) {
      for (size_t j = 0; j < i; j++) {
        (*this)(i, j) = ((*this)(i, j) - conjugate((*this)(j, i))) * 0.5;
        (*this)(j, i) = -conjugate((*this)(i, j));
      }
      T z = (*this)(i, i);
      (*this)(i, i) = 0.5 * (z - conjugate(z));
    }
  }


  /**
   Replaces the current matrix by its hermitian (symmetric) part
   */
  void symmetrize() {
    for (size_t i = 0; i < r; i++) {
      for (size_t j = 0; j < i; j++) {
        (*this)(i, j) = ((*this)(i, j) + conjugate((*this)(j, i))) * 0.5;
        (*this)(j, i) = conjugate((*this)(i, j));
      }
      T z = (*this)(i, i);
      (*this)(i, i) = 0.5 * (z + conjugate(z));
    }
  }



  friend std::istream &operator>>(std::istream &flux, matrix<T> &A) {
    for (size_t i = 0; i < A.r; i++) {
      for (size_t j = 0; j < A.c; j++) flux >> A(i, j);
    }
    return flux;
  }



  friend std::ostream &operator<<(std::ostream &flux, const matrix<T> &A) {
    size_t max_dim_print = global_int("max_dim_print");
    if (A.r > max_dim_print or A.c > max_dim_print) return flux;
    for (size_t i = 0; i < A.r; ++i) {
      for (size_t j = 0; j < A.c; ++j) flux << chop(A(i, j)) << '\t';
      flux << '\n';
    }
    flux << "\n";
    return flux;
  }


  // below are declarations of routines that are type specific
  double norm();
  double norm2();
  void inverse();
  T determinant();
  void eigensystem(vector<double> &d, matrix<T> &U) const; // finds the eigenvalues and eigenvectors of real symmetric matrix
  void eigenvalues(vector<double> &d);                     // finds the eigenvalues of hermitian matrix
  bool is_real(double accuracy = 1e-6);
  bool is_orthogonal(double accuracy = 1e-6);
  bool cholesky(matrix<double> &A);
  bool triangular_inverse();

  
  //------------------------------------------------------------------------------
};

matrix<Complex> to_complex_matrix(const matrix<double> &x);
inline matrix<complex<double>> to_complex_matrix(const matrix<complex<double>> &v)
{
  return v;
}

matrix<Complex> hermitian_matrix_from_real_vector(size_t d, const vector<double> &x);
void hermitian_matrix_to_real_vector(const matrix<Complex> &M, double *x);
matrix<Complex> matrix_from_real_vector(size_t d, const vector<double> &x);
void matrix_to_real_vector(const matrix<Complex> &M, double *x);

#endif
