# Test file
# Goal : lehmann_matrix.to_continued_fraction() must place its very first
#        block B[0] in the *same basis* as the original amplitudes Q, since
#        B[0] is the only floor tied to that basis (Q^dagger = V0 B[0]); any
#        other unitary choice is an equally valid factorization of the
#        zeroth moment Q Q^dagger, but silently rotates A[0] and everything
#        recovered by dropping the first floor (continued_fraction_matrix(
#        A[1:], B[1:]), used e.g. to isolate a self-energy from a cluster
#        Green function) out of the physical (site) basis.
#
# This is exercised for real by periodization_CDMFT_PY.py: it builds Ginv by
# dropping the cluster Green function's first floor, and expects Ginv's pole
# at a bath energy to carry the *same* residue matrix as the (exactly, and
# independently, computed) hybridization function there. Before the fix, the
# zeroth moment Q Q^dagger of a physical Green function is close to a
# multiple of the identity (the usual completeness sum rule), which is
# exactly the case where an SVD/QR factor has an unconstrained unitary gauge
# -- B[0] came out as some arbitrary rotation instead of (approximately) the
# identity, silently rotating Ginv's residues away from the physical basis
# and producing spurious, sometimes negative, residues once subtracted from
# the (correctly-based) hybridization function.
#
# Reproduced here without any ED solver: G(z) = (zI - A0 - Y(z))^{-1} is
# built directly from a random physical A0 and Y = Gamma + Sigma_c, where
# Gamma only has weight in a 2-dimensional ("bath-coupled") subspace of the
# L = 4 orbital space -- the rest are "dark" orbitals, just as sites 3 and 4
# are in periodization_CDMFT_PY.py's model. G's Lehmann representation is
# then obtained exactly (continued_fraction_matrix.to_lehmann()), and fed
# back through lehmann_matrix.to_continued_fraction() -- the round trip that
# needs to preserve the physical basis.
# --------------------------------------------------------------------------------
import numpy as np
from pyqcm.green_structure import lehmann_matrix, continued_fraction_matrix

rng = np.random.default_rng(3)
L = 4

def max_diff(f, g):
    return np.max(np.abs(np.asarray(f) - np.asarray(g)))

# --------------------------------------------------------------------------------
# A0: random physical (Hermitian) one-body matrix

H = rng.standard_normal((L, L)) + 1j * rng.standard_normal((L, L))
A0 = 0.5 * (H + H.conj().T)

# --------------------------------------------------------------------------------
# Gamma: a "hybridization"-like function with weight confined to a
# 2-dimensional subspace of the 4-dimensional orbital space (orbitals 2, 3
# are "dark", exactly as sites 3, 4 are in the real model)

Wg = np.array([1.3, 1.3])
Qg_2d = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
Qg = np.zeros((L, 2), dtype=complex)
Qg[:2, :] = Qg_2d
Gamma = lehmann_matrix(Wg, Qg)

# Sigma_c: an independent self-energy-like function, full support, poles that
# do not collide with Gamma's

Ws = np.sort(rng.uniform(-3, 3, 5))
Qs = rng.standard_normal((L, 5)) + 1j * rng.standard_normal((L, 5))
Sigma_c = lehmann_matrix(Ws, Qs)

Y = Gamma.add_to(Sigma_c)
Y_cf = Y.to_continued_fraction()

# --------------------------------------------------------------------------------
# build G(z) = (zI - A0 - Y(z))^{-1} exactly, as a continued_fraction_matrix,
# then read off its (exact) Lehmann representation

A_full = np.concatenate([A0[None, :, :], Y_cf.A], axis=0)
B_full = np.concatenate([np.eye(L)[None, :, :], Y_cf.B], axis=0)
G_cf_true = continued_fraction_matrix(A_full, B_full)
G_lehmann = G_cf_true.to_lehmann()

z = np.array([0.3 + 0.5j, -1.2 + 0.8j, 2.1 - 0.4j, 5.0 + 0.2j])

err = max_diff(G_lehmann.evaluate(z), G_cf_true.evaluate(z))
print("G_lehmann vs G_cf_true (sanity check) : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# round trip through the fixed lehmann_matrix.to_continued_fraction(): B[0]
# must land back (approximately) on the identity, so A[0] stays in the
# physical basis and matches A0 exactly, not up to an arbitrary rotation

G2_cf = G_lehmann.to_continued_fraction()

print("\nG2_cf.B[0] (should be close to the identity):\n", G2_cf.B[0])
err_B0 = max_diff(G2_cf.B[0], np.eye(L))
print("max|B[0] - I| =", err_B0)
assert err_B0 < 1e-8

err_A0 = max_diff(G2_cf.A[0], A0)
print("max|A[0] - A0| =", err_A0)
assert err_A0 < 1e-8

# --------------------------------------------------------------------------------
# dropping the first floor must reproduce Y(z) directly (no residual
# rotation), the trick periodization_CDMFT_PY.py relies on to isolate a
# self-energy from a cluster Green function

Ginv = continued_fraction_matrix(G2_cf.A[1:], G2_cf.B[1:])
err_Y = max_diff(Ginv.evaluate(z), Y.evaluate(z))
print("Ginv (drop floor 0) vs Y : max|diff| =", err_Y)
assert err_Y < 1e-8

# --------------------------------------------------------------------------------
# most direct check: Ginv's residue matrix at Gamma's pole must equal
# Gamma's own residue matrix exactly (not some rotated, or "leaked into the
# dark orbitals", version of it) -- this is precisely the quantity that came
# out wrong (off by ~0.2, full orbital-structure mismatch) in
# periodization_CDMFT_PY.py before the fix

Ginv_lehmann = Ginv.to_lehmann()
idx = np.where(np.abs(Ginv_lehmann.W - 1.3) < 1e-6)[0]
print("\nGinv poles matching Gamma's location:", Ginv_lehmann.W[idx])
assert idx.size == 2

Rg = Ginv_lehmann.Q[:, idx] @ Ginv_lehmann.Q[:, idx].conj().T
Rgamma = Gamma.Q @ Gamma.Q.conj().T
err_R = max_diff(Rg, Rgamma)
print("Ginv residue at Gamma's pole:\n", Rg.real)
print("Gamma's own residue        :\n", Rgamma.real)
print("max|diff| =", err_R)
assert err_R < 1e-8

print("\ntest_MCF_basis_gauge: all checks passed")
