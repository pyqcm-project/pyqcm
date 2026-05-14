# Test file
# Goal: verify that method='trf' (Trust Region Reflective with analytical Jacobian)
#       converges to the same solution as method='Nelder-Mead' on a simple CDMFT model,
#       and that qcm.CDMFT_residuals / qcm.CDMFT_gradient are callable and consistent.
# -------------------------------------------------------------------------------
import sys
import numpy as np
import pyqcm
from pyqcm import qcm
from pyqcm.cdmft import CDMFT, frequency_grid

# -------------------------------------------------------------------------------
# Minimal 2-orbital cluster + 4 bath sites, graphene-like geometry.
# -------------------------------------------------------------------------------

clus = pyqcm.cluster_model(2, n_bath=4, name='clus_trf')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('Graphene_trf', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))

model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))

model.set_target_sectors(['R0:N6:S0'])

# set_parameters() can only be called once — physics params never change.
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
""")

INIT_BATH = {'eb1_1': 1.0, 'eb2_1': -1.1, 'tb1_1': 0.5, 'tb2_1': 0.4}
varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']
grid = frequency_grid('legendre', (1, 10, 10, 10, 10))

CDMFT_KWARGS = dict(
    varia=varia,
    grid=grid,
    accur=(5e-5, 1e-3),
    convergence=('parameters', 'hybridization'),
    converge_with_stdev=False,
    miniter=1,
    maxiter=20,
    depth=1,
    iteration='broyden',
)


def reset_bath():
    """Reset bath parameters to initial values using set_parameter (plural is one-shot)."""
    for name, val in INIT_BATH.items():
        model.set_parameter(name, val)


# -------------------------------------------------------------------------------
# 1. Run reference with Nelder-Mead
# -------------------------------------------------------------------------------
ref = CDMFT(model, method='Nelder-Mead', **CDMFT_KWARGS)
x_ref = np.array([ref.I.parameters()[v] for v in varia])
dist_ref = ref.dist
print(f"[Nelder-Mead] dist={dist_ref:.4g}, x={x_ref}")

# -------------------------------------------------------------------------------
# 2. Run with method='trf'
# -------------------------------------------------------------------------------
reset_bath()
trf = CDMFT(model, method='trf', jac=True, lm_max_nfev=2000, **CDMFT_KWARGS)
x_trf = np.array([trf.I.parameters()[v] for v in varia])
dist_trf = trf.dist
print(f"[trf]         dist={dist_trf:.4g}, x={x_trf}")

# -------------------------------------------------------------------------------
# 3. TRF and Nelder-Mead should reach comparable distance values.
# -------------------------------------------------------------------------------
if dist_trf > max(dist_ref * 10, 1e-6):
    print(f"FAIL: TRF distance {dist_trf:.4g} much larger than Nelder-Mead {dist_ref:.4g}")
    sys.exit(1)

# -------------------------------------------------------------------------------
# 4. Check CDMFT_residuals and CDMFT_gradient shapes using the converged TRF state.
#    trf.I is the final instance created at convergence; we must initialize G_host
#    on it before calling CDMFT_residuals / CDMFT_gradient.
# -------------------------------------------------------------------------------
# Bath parameters for cluster 0 (nvar[0] = Hartree offset, nvar[1] = cluster 1 count)
hartree_offset = trf.nvar[0]
n_bath_params = trf.nvar[1]
x_bath = trf.x[hartree_offset : hartree_offset + n_bath_params]

# Initialize G_host on the final converged instance so CDMFT_residuals can run.
qcm.CDMFT_host(grid.wr, grid.cdmft_weight, trf.I.label)

r = qcm.CDMFT_residuals(x_bath, 0, trf.I.label)
J = qcm.CDMFT_gradient(x_bath, 0, trf.I.label)

Nfreq = len(grid.wr)
dim = 2  # 2 cluster sites
expected_rows = 2 * Nfreq * dim * dim

if len(r) != expected_rows:
    print(f"FAIL: residual length {len(r)} != expected {expected_rows}")
    sys.exit(1)

if J.shape != (expected_rows, n_bath_params):
    print(f"FAIL: Jacobian shape {J.shape} != expected ({expected_rows}, {n_bath_params})")
    sys.exit(1)

# -------------------------------------------------------------------------------
# 5. Jacobian consistency: J[:,0]·δ ≈ Δr for a small perturbation of parameter 0.
# -------------------------------------------------------------------------------
delta = 1e-5
x_pert = x_bath.copy()
x_pert[0] += delta
r_pert = np.array(qcm.CDMFT_residuals(x_pert, 0, trf.I.label))
r0 = np.array(r)
dr_fd = (r_pert - r0) / delta
dr_jac = J[:, 0]

denom = np.max(np.abs(dr_fd)) + 1e-14
rel_err = np.max(np.abs(dr_fd - dr_jac)) / denom
print(f"Jacobian column 0 finite-difference check: max rel error = {rel_err:.3g}")
if rel_err > 0.001:
    print(f"FAIL: Jacobian first-column relative error {rel_err:.3g} exceeds 0.1%")
    sys.exit(1)

# -------------------------------------------------------------------------------
# ValueError test: method='trf' without jac=True must raise.
# The ValueError from optimize() is wrapped in SolverError by CDMFT.__init__,
# so we check either the top-level exception or its __cause__.
# -------------------------------------------------------------------------------
reset_bath()
try:
    CDMFT(model, method='trf', **CDMFT_KWARGS)
    # If we reach here, no error was raised — that's a test failure.
    print("FAIL: CDMFT(method='trf') without jac=True should have raised ValueError")
    sys.exit(1)
except Exception as e:
    # The ValueError may be raised directly or wrapped in SolverError.
    cause = e.__cause__ if e.__cause__ is not None else e
    if isinstance(cause, ValueError) or isinstance(e, ValueError):
        msg = str(cause) if isinstance(cause, ValueError) else str(e)
        print(f"[trf-no-jac] ValueError raised correctly: {msg}")
    else:
        print(f"FAIL: expected ValueError (possibly wrapped), got {type(e).__name__}: {e}")
        sys.exit(1)

print("All tests passed.")
sys.exit(0)
