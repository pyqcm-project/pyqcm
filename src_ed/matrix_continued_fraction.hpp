#ifndef matrix_continued_fraction_h
#define matrix_continued_fraction_h

/*
 matrix_continued_fraction.hpp

 Matrix-valued continued fraction arising from the block Lanczos method.

 Scalar analogue: continued_fraction (see continued_fraction.hpp).

 A block Lanczos run with block size p and M steps produces:
   A[0..M-1]  -- p×p Hermitian diagonal blocks
   B[0..M-1]  -- p×p upper-triangular off-diagonal QR blocks

 These define a block-tridiagonal projected Hamiltonian T whose (0,0)
 block resolvent F_0(z) = [(zI - T)^{-1}]_{00} is evaluated via the
 backward recursion (matrix continued fraction):

   F_M = 0_{p×p}
   F_j = (zI - A_j - B_j^H F_{j+1} B_j)^{-1},   j = M-1 … 0

 where B_j (j < M-1) are the off-diagonal blocks stored in B; B[M-1] is
 the truncation residual and is not used in the recursion, in exact
 analogy with the last element of the b array in continued_fraction.

 If the starting block phi was not pre-orthonormalised before calling
 blockLanczos, an optional weight matrix W can be supplied so that the
 full projected Green function is  G(z) = W^H F_0(z) W.

 Usage:
   vector<matrix<HilbertField>> A, B;
   int M0 = 50;
   blockLanczos(H, phi, A, B, M0);
   matrix_continued_fraction mcf(A, B);          // orthonormal starting block
   matrix<Complex> G = mcf.evaluate(z);
*/

#include "continued_fraction.hpp"   // pulls in block_matrix.hpp → matrix.hpp

//! Matrix-valued Jacobi continued fraction.
/**
 Partial denominators A[j] and partial-numerator factors B[j] are p×p
 complex matrices, stored internally as matrix<Complex> regardless of
 the field used in the Lanczos computation.
*/
struct matrix_continued_fraction
{
    int p;                         //!< block size
    vector<matrix<Complex>> A;     //!< diagonal blocks (partial denominators)
    vector<matrix<Complex>> B;     //!< off-diagonal QR blocks (partial numerator factors)
    matrix<Complex>         W;     //!< initial weight (default: identity)

    //! Default constructor
    matrix_continued_fraction();

    //! Constructor from block Lanczos output with orthonormal starting block (W = I).
    /**
     @param _A  Diagonal blocks from blockLanczos (double or Complex).
     @param _B  Off-diagonal QR blocks from blockLanczos (double or Complex).
    */
    template<typename HilbertField>
    matrix_continued_fraction(const vector<matrix<HilbertField>> &_A,
                               const vector<matrix<HilbertField>> &_B)
    {
        p = (int)_A[0].r;
        A.resize(_A.size());
        B.resize(_B.size());
        for(size_t j = 0; j < _A.size(); ++j) A[j] = to_complex_matrix(_A[j]);
        for(size_t j = 0; j < _B.size(); ++j) B[j] = to_complex_matrix(_B[j]);
        W.set_size(p);
        W.identity();
    }

    //! Constructor analogous to continued_fraction(a, b, e0, norm, create).
    /**
     Applies an energy shift to the diagonal blocks (for electron/hole
     components of the Green function, matching the scalar continued_fraction
     constructor convention) and stores an initial weight matrix W.

     @param _A     Diagonal blocks from blockLanczos.
     @param _B     Off-diagonal QR blocks from blockLanczos.
     @param e0     Ground-state energy used to shift A[j].
     @param _W     p×p weight matrix arising from initial-block normalisation.
                   Pass the identity for an orthonormal starting block.
     @param create If true (creation/electron sector):  A[j] -= e0*I.
                   If false (annihilation/hole sector):  A[j]  = e0*I - A[j].
    */
    template<typename HilbertField>
    matrix_continued_fraction(const vector<matrix<HilbertField>> &_A,
                               const vector<matrix<HilbertField>> &_B,
                               double e0,
                               const matrix<HilbertField> &_W,
                               bool create)
    {
        p = (int)_A[0].r;
        A.resize(_A.size());
        B.resize(_B.size());
        for(size_t j = 0; j < _A.size(); ++j){
            A[j] = to_complex_matrix(_A[j]);
            if(create) A[j] -= e0;            // A[j] -= e0 * I
            else {
                for(size_t r = 0; r < A[j].v.size(); ++r) A[j].v[r] = -A[j].v[r];
                A[j] += e0;                   // A[j] = e0*I - A[j]
            }
        }
        for(size_t j = 0; j < _B.size(); ++j) B[j] = to_complex_matrix(_B[j]);
        W = to_complex_matrix(_W);
    }

    //! Constructor from pre-processed (already complex, already shifted) matrices and explicit weight.
    /**
     Used when the MCF coefficients have been constructed externally (e.g., after
     periodization in the lattice model), so no additional template conversion or
     energy shift is needed.

     @param _A  Diagonal blocks, already complex and energy-shifted.
     @param _B  Off-diagonal QR blocks, already complex.
     @param _W  Weight matrix (p×p); G(z) = W^H F_0(z) W.
    */
    matrix_continued_fraction(const vector<matrix<Complex>> &_A,
                               const vector<matrix<Complex>> &_B,
                               const matrix<Complex> &_W)
    {
        if(_A.empty()){ p = 0; return; }
        p = (int)_A[0].r;
        A = _A;
        B = _B;
        W = _W;
    }

    //! Evaluate the matrix continued fraction at complex frequency z.
    /**
     Returns G(z) = W^H F_0(z) W where
       F_M = 0,
       F_j = (zI - A_j - B_j^H F_{j+1} B_j)^{-1},  j = M-1 … 0.
     B[M-1] (the last element) is the truncation residual and is not used.
    */
    matrix<Complex> evaluate(Complex z) const;

    //! Number of levels (= A.size()).
    int floors() const { return (int)A.size(); }
};


std::ostream& operator<<(std::ostream &flux, const matrix_continued_fraction &F);
std::istream& operator>>(std::istream &flux,       matrix_continued_fraction &F);


#endif