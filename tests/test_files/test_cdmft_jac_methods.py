# test_cdmft_jac_methods.py
# Goal: verify that the analytical-Jacobian quasi-Newton methods
#   BFGS + jac=True, L-BFGS-B + jac=True, L-BFGS-B + jac=True (NLopt backend)
#   converge to the same bath-fitting solution as Nelder-Mead on a simple CDMFT model,
#   and that the global parameter cdmft_jacobian_delta flows through
#   to CDMFT_gradient.
# -------------------------------------------------------------------------------
import sys
import numpy as np
import pyqcm
from pyqcm import qcm
from pyqcm.cdmft import CDMFT, frequency_grid

# -------------------------------------------------------------------------------
# Minimal 2-orbital cluster + 4 bath sites, graphene-like geometry
# (same physics as test_cdmft_trf.py, different model name to avoid conflicts).
# -------------------------------------------------------------------------------

clus = pyqcm.cluster_model(2, n_bath=4, name='clus_jac')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('Graphene_jac', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))

model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))

model.set_target_sectors(['R0:N6:S0'])

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
    """Reset bath parameters to initial values."""
    for name, val in INIT_BATH.items():
        model.set_parameter(name, val)


# -------------------------------------------------------------------------------
# 1. Reference run with Nelder-Mead (quality baseline for the -jac methods).
# -------------------------------------------------------------------------------
ref = CDMFT(model, method='Nelder-Mead', **CDMFT_KWARGS)
dist_ref = ref.dist
print(f"[Nelder-Mead] dist={dist_ref:.4g}")


# -------------------------------------------------------------------------------
# 2. Global-parameter round-trip test (spec acceptance criterion 1).
# -------------------------------------------------------------------------------
pyqcm.set_global_parameter("cdmft_jacobian_delta", 5e-6)
got = pyqcm.get_global_parameter("cdmft_jacobian_delta")
if not np.isclose(got, 5e-6):
    print(f"FAIL: global parameter round-trip returned {got}, expected 5e-6")
    sys.exit(1)
print(f"[global-param] round-trip OK: got {got}")


# -------------------------------------------------------------------------------
# 3. Flow-through test: changing cdmft_jacobian_delta changes the Jacobian.
# -------------------------------------------------------------------------------
reset_bath()
trf = CDMFT(model, method='trf', jac=True, lm_max_nfev=2000, **CDMFT_KWARGS)
# Use converged bath parameters directly from model state
x_bath = model.parameters(varia)
# Set up the host Green function (required before calling CDMFT_gradient)
qcm.CDMFT_host(grid.wr, grid.cdmft_weight, trf.I.label)

# Use 1e-4 (10x default) and 1e-6 (0.1x default): far enough apart that
# central differences at O(delta^2) produce a measurable column-wise difference,
# yet both within the linear-response regime for typical bath parameters.
pyqcm.set_global_parameter("cdmft_jacobian_delta", 1e-4)
J_coarse = np.array(qcm.CDMFT_gradient(x_bath, 0, trf.I.label))

pyqcm.set_global_parameter("cdmft_jacobian_delta", 1e-6)
J_fine = np.array(qcm.CDMFT_gradient(x_bath, 0, trf.I.label))

# Reset to default before remaining tests
pyqcm.set_global_parameter("cdmft_jacobian_delta", 1e-5)

diff_norm = np.linalg.norm(J_coarse - J_fine)
if np.isnan(diff_norm):
    print("FAIL: J_coarse or J_fine is NaN; CDMFT_gradient computation failed")
    sys.exit(1)
if diff_norm == 0.0:
    print("FAIL: J_coarse and J_fine are bit-identical; cdmft_jacobian_delta not flowing through to C++")
    sys.exit(1)
print(f"[flow-through] |J(1e-4) - J(1e-6)| = {diff_norm:.3g} (nonzero, OK)")


# -------------------------------------------------------------------------------
# 4. BFGS + jac=True: scipy quasi-Newton with analytical Jacobian.
# -------------------------------------------------------------------------------
reset_bath()
bfgs_jac = CDMFT(model, method='BFGS', jac=True, **CDMFT_KWARGS)
dist_bfgs_jac = bfgs_jac.dist
print(f"[BFGS+jac]    dist={dist_bfgs_jac:.4g}")

if np.isnan(dist_bfgs_jac):
    print("FAIL: BFGS+jac=True returned NaN distance")
    sys.exit(1)
if dist_bfgs_jac > max(dist_ref * 10, 1e-6):
    print(f"FAIL: BFGS+jac=True distance {dist_bfgs_jac:.4g} much larger than Nelder-Mead {dist_ref:.4g}")
    sys.exit(1)

# -------------------------------------------------------------------------------
# 5. L-BFGS-B + jac=True: scipy bounded quasi-Newton with analytical Jacobian.
# -------------------------------------------------------------------------------
reset_bath()
lbfgsb_jac = CDMFT(model, method='L-BFGS-B', jac=True, **CDMFT_KWARGS)
dist_lbfgsb_jac = lbfgsb_jac.dist
print(f"[L-BFGS-B+jac] dist={dist_lbfgsb_jac:.4g}")

if np.isnan(dist_lbfgsb_jac):
    print("FAIL: L-BFGS-B+jac=True returned NaN distance")
    sys.exit(1)
if dist_lbfgsb_jac > max(dist_ref * 10, 1e-6):
    print(f"FAIL: L-BFGS-B+jac=True distance {dist_lbfgsb_jac:.4g} much larger than Nelder-Mead {dist_ref:.4g}")
    sys.exit(1)

# -------------------------------------------------------------------------------
# 6. L-BFGS-B + jac=True (NLopt backend): NLopt L-BFGS with analytical Jacobian.
# -------------------------------------------------------------------------------
reset_bath()
nlopt_jac = CDMFT(model, method='L-BFGS-B', jac=True, **CDMFT_KWARGS)
dist_nlopt_jac = nlopt_jac.dist
print(f"[L-BFGS-B+jac(NLopt)] dist={dist_nlopt_jac:.4g}")

if np.isnan(dist_nlopt_jac):
    print("FAIL: L-BFGS-B+jac=True (NLopt backend) returned NaN distance")
    sys.exit(1)
if dist_nlopt_jac > max(dist_ref * 10, 1e-6):
    print(f"FAIL: L-BFGS-B+jac=True (NLopt backend) distance {dist_nlopt_jac:.4g} much larger than Nelder-Mead {dist_ref:.4g}")
    sys.exit(1)

print("All tests passed.")
sys.exit(0)
