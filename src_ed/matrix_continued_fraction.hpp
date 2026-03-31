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
   matrix_continued_fraction<HilbertField> mcf(A, B);   // orthonormal starting block
   matrix<Complex> G = mcf.evaluate(z);
*/

#include "continued_fraction.hpp"   // pulls in block_matrix.hpp → matrix.hpp
// hdf5_io.hpp is already pulled in by continued_fraction.hpp
#include "Q_matrix.hpp"

//! Matrix-valued Jacobi continued fraction.
/**
 Template parameter T is the field of the Lanczos matrices A and B
 (double for a real Hamiltonian, Complex for a complex one).
 The weight matrix W and the Green function G(z) are always complex.
*/
template<typename T>
struct matrix_continued_fraction
{
    int p;                        //!< block size
    vector<matrix<T>> A;          //!< diagonal blocks (partial denominators)
    vector<matrix<T>> B;          //!< off-diagonal QR blocks (partial numerator factors)
    matrix<Complex>   W;          //!< initial weight (default: identity, always complex)

    //! Default constructor
    matrix_continued_fraction() : p(0) {}

    //! Constructor from block Lanczos output with orthonormal starting block (W = I).
    /**
     @param _A  Diagonal blocks from blockLanczos.
     @param _B  Off-diagonal QR blocks from blockLanczos.
    */
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B)
    {
        p = (int)_A[0].r;
        A = _A;
        B = _B;
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
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B,
                               double e0,
                               const matrix<T> &_W,
                               bool create)
    {
        p = (int)_A[0].r;
        A = _A;
        B = _B;
        for(size_t j = 0; j < A.size(); ++j){
            if(create) A[j] -= e0;            // A[j] -= e0 * I
            else {
                for(size_t r = 0; r < A[j].v.size(); ++r) A[j].v[r] = -A[j].v[r];
                A[j] += e0;                   // A[j] = e0*I - A[j]
            }
        }
        W = to_complex_matrix(_W);
    }

    //! Constructor from pre-processed matrices and explicit complex weight.
    /**
     Used when the MCF coefficients have been constructed externally (e.g., after
     periodization in the lattice model), so no additional energy shift is needed.

     @param _A  Diagonal blocks, already energy-shifted.
     @param _B  Off-diagonal QR blocks.
     @param _W  Weight matrix (p×p, complex); G(z) = W^H F_0(z) W.
    */
    matrix_continued_fraction(const vector<matrix<T>> &_A,
                               const vector<matrix<T>> &_B,
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
    matrix<Complex> evaluate(Complex z) const
    {
        int M = (int)A.size();

        // F starts as the M-th floor: F_M = 0
        matrix<Complex> F(p, p);   // zero-initialised

        for(int j = M - 1; j >= 0; --j){

            // Σ = B[j]^H · F · B[j]  (only when j < M-1; otherwise Σ = 0)
            matrix<Complex> sigma(p, p);  // zero-initialised
            if(j < M - 1){
                matrix<Complex> Bj = to_complex_matrix(B[j]);
                // step 1: tmp = F · B[j]
                matrix<Complex> tmp(p);
                tmp.product(F, Bj);
                // step 2: sigma = B[j]^H · tmp
                matrix<Complex> Bjh(Bj);
                Bjh.hermitian_conjugate();
                sigma.product(Bjh, tmp);
            }

            // Build denominator D = z·I - A[j] - Σ
            matrix<Complex> D(p, p);    // zero
            D += z;                     // D = z·I
            D -= to_complex_matrix(A[j]);  // D = z·I - A[j]
            D -= sigma;                 // D = z·I - A[j] - Σ

            D.inverse();                // D  →  D^{-1}
            F = D;                      // F_{j} = D^{-1}
        }

        // Apply weight: G = W^H · F_0 · W
        // W may be non-square (p rows × q columns): result is q×q.
        int q = (int)W.c;
        matrix<Complex> tmp(p, q);
        tmp.product(F, W);              // tmp = F · W  (p × q)
        matrix<Complex> Wh(W);
        Wh.hermitian_conjugate();       // Wh  = W^H   (q × p)
        matrix<Complex> G(q, q);
        G.product(Wh, tmp);             // G   = W^H · F · W  (q × q)

        return G;
    }

    //! Apply the block-tridiagonal Lanczos matrix T to a vector and return T*x.
    /**
     The input vector x must have size M*p (M = A.size(), p = block size).
     T is block-tridiagonal with diagonal blocks A[j], sub-diagonal blocks B[j-1]
     (at block row j, column j-1), and super-diagonal blocks B[j]^H (at block row j,
     column j+1).  B[M-1] is the truncation residual and is NOT applied.

       (T x)_j = B[j-1] x_{j-1} + A[j] x_j + B[j]^H x_{j+1}

     with the boundary terms B[-1] = B[M-1] = 0.
    */
    vector<T> apply(const vector<T>& x) const {
        int M = (int)A.size();
        vector<T> y(M * p, T(0));
        for(int j = 0; j < M; ++j){
            // diagonal: A[j] * x_j
            for(int col = 0; col < p; ++col)
                for(int row = 0; row < p; ++row)
                    y[j*p + row] += A[j](row, col) * x[j*p + col];
            // sub-diagonal: B[j-1] * x_{j-1}
            if(j > 0)
                for(int col = 0; col < p; ++col)
                    for(int row = 0; row < p; ++row)
                        y[j*p + row] += B[j-1](row, col) * x[(j-1)*p + col];
            // super-diagonal: B[j]^H * x_{j+1}  (B[M-1] not used)
            if(j < M-1)
                for(int col = 0; col < p; ++col)
                    for(int row = 0; row < p; ++row)
                        y[j*p + row] += conjugate(B[j](col, row)) * x[(j+1)*p + col];
        }
        return y;
    }

    //! Returns a real-valued copy by taking the real part of every matrix.
    /**
     Converts this MCF to a matrix_continued_fraction<double> by extracting
     the real part of each A[j], B[j], and W.  Useful when the Hamiltonian is
     known to be real and the imaginary parts are negligible numerical noise.
    */
    matrix_continued_fraction<double> real() const {
        matrix_continued_fraction<double> result;
        result.p = p;
        result.A.resize(A.size());
        result.B.resize(B.size());
        for(size_t j = 0; j < A.size(); ++j) result.A[j] = to_real_matrix(A[j]);
        for(size_t j = 0; j < B.size(); ++j) result.B[j] = to_real_matrix(B[j]);
        result.W = to_complex_matrix(to_real_matrix(W));
        return result;
    }

    //! Number of levels (= A.size()).
    int floors() const { return (int)A.size(); }
};


// C++14-compatible helpers: convert a Complex scalar to field T.
inline double      mcf_to_T(Complex z, double)  { return z.real(); }
inline Complex     mcf_to_T(Complex z, Complex) { return z; }


//! Diagonal-matrix operator for block Lanczos.
/**
 Implements the Hamiltonian H = diag(e[0], ..., e[M-1]) used in Q_matrix_to_mcf.
 Satisfies the TYPE concept expected by blockLanczos via mult_add().
*/
template<typename T>
struct diagonal_hamiltonian {
    const vector<double>& evals;
    explicit diagonal_hamiltonian(const vector<double>& _e) : evals(_e) {}
    void mult_add(const vector<T>& x, vector<T>& y) const {
        for(size_t i = 0; i < evals.size(); ++i)
            y[i] += T(evals[i]) * x[i];
    }
};


//! Convert a Q_matrix into a matrix_continued_fraction via block Lanczos.
/**
 The Hamiltonian used is the diagonal matrix H = diag(Q.e[0], ..., Q.e[M-1]).
 The L starting vectors are the L rows of Q.v.  The resulting MCF is equivalent
 to the Lehmann sum G(z)(a,b) = sum_k Q.v(a,k) * conj(Q.v(b,k)) / (z - Q.e[k]).

 The weight matrix W is extracted from Q.v via modified Gram-Schmidt (as in
 combine_via_lanczos), so that G(z) = W^H F_0(z) W.

 M0 is set to Q.M (the Krylov space cannot exceed the eigenvalue count M),
 capped by the global parameter max_iter_BL.

 @param Q  Combined Q_matrix (electron + hole poles, already energy-shifted).
 @return   matrix_continued_fraction equivalent to Q.
*/
template<typename HilbertField>
matrix_continued_fraction<HilbertField> Q_matrix_to_mcf(const Q_matrix<HilbertField>& Q)
{
    const int L = (int)Q.L;
    const int M = (int)Q.M;
    if(M == 0 || L == 0) return matrix_continued_fraction<HilbertField>();

    // Build starting block: phi[a] = row a of Q.v  (length-M vectors)
    vector<vector<HilbertField>> phi(L, vector<HilbertField>(M));
    for(int a = 0; a < L; ++a)
        for(int k = 0; k < M; ++k)
            phi[a][k] = Q.v(a, k);

    // Extract the upper-triangular QR factor W from phi via modified Gram-Schmidt.
    // The working copy q is discarded; blockLanczos re-orthonormalises phi internally.
    matrix<HilbertField> W(L);
    {
        vector<vector<HilbertField>> q(phi);
        for(int l = 0; l < L; ++l){
            for(int k = 0; k < l; ++k){
                HilbertField z = q[k] * q[l];
                W(k, l) = z;
                mult_add(-z, q[k], q[l]);
            }
            double nrm = norm(q[l]);
            W(l, l) = HilbertField(nrm);
            q[l] *= 1.0 / nrm;
        }
    }

    diagonal_hamiltonian<HilbertField> H(Q.e);
    vector<matrix<HilbertField>> A, B;
    int M0 = M;
    if(global_bool("block_Lanczos_QR"))
        blockLanczos(H, phi, A, B, M0);
    else
        blockLanczosSVD(H, phi, A, B, M0);

    return matrix_continued_fraction<HilbertField>(A, B, to_complex_matrix(W));
}


//! Combine two MCFs into a single one whose evaluate() gives G_h(z) + G_e(z)^T directly.
/**
 Assembles the direct-sum block-tridiagonal T = T_e ⊕ T_h (same blocks as
 direct_sum()) but uses a non-square weight

   W_combined = [conj(W_e)]   (first  p rows: conjugated electron weight)
                [W_h       ]   (second p rows: hole weight)

 of size 2p × p.  Because W_combined has p columns, evaluate() returns a p×p
 Green function rather than 2p×2p:

   G(z) = W_combined^H F_combined(z) W_combined
        = W_e^T F_e(z) W_e* + W_h^H F_h(z) W_h

 For T = double (real Hamiltonian): F_e is complex-symmetric and W_e is real, so
 W_e^T F_e W_e* = G_e(z) = G_e(z)^T.  The result is exact.

 For T = Complex: exact when time-reversal symmetry ensures G_e = G_e^T (the
 common case).  For Hamiltonians that break time-reversal the approximation
 replaces G_e^T with W_e^T F_e W_e*; in that case use two separate evaluate()
 calls instead.
*/
template<typename T>
matrix_continued_fraction<T> combine_for_gf(
    const matrix_continued_fraction<T>& e,
    const matrix_continued_fraction<T>& h)
{
    const int pe = e.p, ph = h.p;
    QCM_ASSERT(pe == ph);
    const int p = pe + ph;
    const int Me = e.floors(), Mh = h.floors(), M = max(Me, Mh);

    // Direct-sum A and B blocks (identical to direct_sum())
    vector<matrix<T>> A(M), B(M);
    for(int j = 0; j < M; ++j){
        A[j] = matrix<T>(p, p);
        B[j] = matrix<T>(p, p);
        if(j < Me){
            e.A[j].move_sub_matrix(pe, pe, 0, 0, 0,  0,  A[j]);
            e.B[j].move_sub_matrix(pe, pe, 0, 0, 0,  0,  B[j]);
        }
        if(j < Mh){
            h.A[j].move_sub_matrix(ph, ph, 0, 0, pe, pe, A[j]);
            h.B[j].move_sub_matrix(ph, ph, 0, 0, pe, pe, B[j]);
        }
    }

    // Non-square weight W (2p × p): upper block = conj(W_e), lower block = W_h.
    matrix<Complex> W(p, pe);   // zero-initialised
    for(int k = 0; k < pe; ++k)
        for(int i = 0; i < pe; ++i)
            W(k, i) = conj(e.W(k, i));   // conj(W_e) in electron rows
    for(int k = 0; k < ph; ++k)
        for(int i = 0; i < ph; ++i)
            W(pe + k, i) = h.W(k, i);    // W_h in hole rows

    return matrix_continued_fraction<T>(A, B, W);
}


//! Operator wrapper for the direct-sum T_e ⊕ T_h* acting on the concatenated space.
/**
 The combined space has dimension (Me + Mh)*p with the e-sector occupying the
 first Me*p components and the h-sector occupying the last Mh*p components.
 Both sectors share the same block size p.

 The e-sector applies T_e normally.  The h-sector applies T_h* (element-wise
 complex conjugate of T_h), i.e. T_h* x = conj(T_h conj(x)).  This is needed
 so that the combined MCF (with conj(W_h) starting vectors for the h-sector)
 reproduces G_mcf_h^T rather than G_mcf_h.  For T=double, T_h* = T_h and
 conj is a no-op, so the real case is unaffected.

 Satisfies the TYPE concept expected by blockLanczos:
   void mult_add(const vector<T>& x, vector<T>& y)
*/
template<typename T>
struct combined_sector_operator {
    const matrix_continued_fraction<T>& e;
    const matrix_continued_fraction<T>& h;
    const int Me, Mh, p;

    combined_sector_operator(const matrix_continued_fraction<T>& _e,
                              const matrix_continued_fraction<T>& _h)
        : e(_e), h(_h), Me(_e.floors()), Mh(_h.floors()), p(_e.p)
    { QCM_ASSERT(_e.p == _h.p); }

    void mult_add(const vector<T>& x, vector<T>& y) const {
        // e-sector: y_e += T_e * x_e
        vector<T> xe(x.begin(), x.begin() + Me * p);
        vector<T> ye = e.apply(xe);
        for(int i = 0; i < Me * p; ++i) y[i] += ye[i];

        // h-sector: y_h += T_h* x_h = conj(T_h conj(x_h))
        vector<T> xh(x.begin() + Me * p, x.end());
        for(auto& v : xh) v = conjugate(v);          // conjugate x_h
        vector<T> yh = h.apply(xh);                  // T_h * conj(x_h)
        for(int i = 0; i < Mh * p; ++i) y[Me * p + i] += conjugate(yh[i]);
    }
};


//! Combine two MCFs via a new block Lanczos run on their direct-sum operator.
/**
 Applies blockLanczos to the direct-sum operator T = T_e ⊕ T_h on a space of
 dimension N = (Me + Mh)*p, using p starting vectors whose level-0 components
 are W_e (e-sector) and W_h (h-sector).  The resulting MCF has the same block
 size p as the individual electron and hole MCFs.

 The Green function of the result satisfies:
   G_combined(z) = W_new^H F_new(z) W_new = G⁺(z) + (G⁻)ᵀ(z)

 matching the convention of the default (non-combined) path.  This is achieved
 by running blockLanczos on T_e ⊕ T_h* (T_h* = conj of T_h) and using
 conj(W_h) as h-sector starting vectors, so that the h contribution becomes
   conj(W_h)^H F_{T_h*} conj(W_h) = G_mcf_h^T = (G⁻)ᵀ
 For T=double (real Hamiltonian), T_h* = T_h and the modification is a no-op.

 M0 must be at least Me + Mh to capture the full Krylov space of the
 combined operator (each sector independently contributes Me and Mh steps).

 @param e   Electron MCF (block size p, Me floors).
 @param h   Hole MCF (block size p, Mh floors).
 @param M0  Maximum number of Lanczos steps; updated to actual count on return.
*/
template<typename T>
matrix_continued_fraction<T> combine_via_lanczos(
    const matrix_continued_fraction<T>& e,
    const matrix_continued_fraction<T>& h,
    int M0)
{
    const int p  = e.p;
    QCM_ASSERT(e.p == h.p);
    const int Me = e.floors(), Mh = h.floors();
    const int N  = (Me + Mh) * p;  // total dimension of the combined space

    // Build starting block: p vectors of length N.
    // phi[i]: W_e[:,i]       at e-sector level 0 (positions 0..p-1)
    // phi[i]: conj(W_h[:,i]) at h-sector level 0 (positions Me*p .. Me*p+p-1)
    // The conjugate on the h-sector, combined with the T_h* operator, ensures
    // the h contribution to evaluate() gives (G⁻)ᵀ instead of G⁻.
    vector<vector<T>> phi(p, vector<T>(N, T(0)));
    for(int i = 0; i < p; ++i){
        for(int k = 0; k < p; ++k){
            phi[i][k]          = mcf_to_T(e.W(k, i),       T(0));  // W_e
            phi[i][Me * p + k] = mcf_to_T(conj(h.W(k, i)), T(0));  // conj(W_h)
        }
    }

    // Extract the upper-triangular QR factor W_new from phi via modified
    // Gram-Schmidt (mirrors the procedure in model_instance::build_mcf).
    matrix<T> W_new(p);
    {
        vector<vector<T>> q(phi);   // working copies
        for(int l = 0; l < p; ++l){
            for(int k = 0; k < l; ++k){
                T z = q[k] * q[l];   // inner product <q[k]|q[l]>
                W_new(k, l) = z;
                mult_add(-z, q[k], q[l]);
            }
            double nrm = norm(q[l]);
            W_new(l, l) = T(nrm);
            q[l] *= 1.0 / nrm;
        }
    }

    // Run block Lanczos on T_e ⊕ T_h with the p starting vectors.
    // block_Lanczos_QR=true (default) uses QR; false uses polar decomposition (Hermitian B).
    combined_sector_operator<T> op(e, h);
    vector<matrix<T>> A_new, B_new;
    if(global_bool("block_Lanczos_QR"))
        blockLanczos(op, phi, A_new, B_new, M0);
    else
        blockLanczosSVD(op, phi, A_new, B_new, M0);

    return matrix_continued_fraction<T>(A_new, B_new, to_complex_matrix(W_new));
}


template<typename T>
std::ostream& operator<<(std::ostream& os, const matrix_continued_fraction<T>& F)
{
    int M = F.floors();
    os << "matrix_continued_fraction: p=" << F.p << "  floors=" << M << '\n';
    for(int j = 0; j < M; ++j){
        os << "A[" << j << "]:\n" << F.A[j];
    }
    for(int j = 0; j < (int)F.B.size(); ++j){
        os << "B[" << j << "]:\n" << F.B[j];
    }
    os << "W:\n" << F.W;
    return os;
}


template<typename T>
void h5_write_mcf(H5::Group& grp, const matrix_continued_fraction<T>& F)
{
    int M = F.floors();
    h5_write_attr(grp, "p",      F.p);
    h5_write_attr(grp, "floors", M);
    for(int j = 0; j < M; ++j){
        H5::Group ag = grp.createGroup("A_" + to_string(j));
        h5_write_mat(ag, "data", F.A[j]);
    }
    for(int j = 0; j < (int)F.B.size(); ++j){
        H5::Group bg = grp.createGroup("B_" + to_string(j));
        h5_write_mat(bg, "data", F.B[j]);
    }
    H5::Group wg = grp.createGroup("W");
    h5_write_mat(wg, "data", F.W);
}


template<typename T>
void h5_read_mcf(H5::Group& grp, matrix_continued_fraction<T>& F)
{
    F.p   = h5_read_attr_int(grp, "p");
    int M = h5_read_attr_int(grp, "floors");
    F.A.resize(M);
    F.B.resize(M);
    for(int j = 0; j < M; ++j){
        H5::Group ag = grp.openGroup("A_" + to_string(j));
        h5_read_mat(ag, "data", F.A[j]);
    }
    for(int j = 0; j < M; ++j){
        H5::Group bg = grp.openGroup("B_" + to_string(j));
        h5_read_mat(bg, "data", F.B[j]);
    }
    H5::Group wg = grp.openGroup("W");
    h5_read_mat(wg, "data", F.W);
}


#endif
