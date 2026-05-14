import numpy as np

import pyqcm
from pyqcm.cdmft import CDMFT

pyqcm.set_global_parameter("cdmft_jacobian_delta", 1e-8)

# ----------------------------------------------------------------
# Minimal 2-orbital cluster + 4 bath sites, graphene-like geometry
# ----------------------------------------------------------------

clus_model = pyqcm.cluster_model(2, n_bath=4, name="clus_trf")
clus_model.new_operator(
    "tb1", "one-body", [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)]
)
clus_model.new_operator(
    "tb2", "one-body", [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)]
)
clus_model.new_operator(
    "eb1", "one-body", [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)]
)
clus_model.new_operator(
    "eb2", "one-body", [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)]
)

clus = pyqcm.cluster(clus_model, ((0, 0, 0), (1, 0, 0)))

model = pyqcm.lattice_model(
    "Graphene_trf", clus, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0))
)
model.set_basis([(1, 0, 0), [-0.5, np.sqrt(3) / 2, 0]])

model.interaction_operator("U", orbitals=(1, 1))
model.interaction_operator("U", orbitals=(2, 2))

model.hopping_operator("t", (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator("t", (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator("t", (-1, -1, 0), -1, orbitals=(1, 2))

model.set_target_sectors(["R0:N6:S0"])
model.set_parameters("""
    U = 4
    mu = 0.5*U
    t = 1
    eb1_1 = 1.0
    eb2_1 = -1.0
    tb1_1 = 0.5
    tb2_1 = 0.5
""")

varia = ["eb1_1", "eb2_1", "tb1_1", "tb2_1"]

# ---------------------
# Run with method='trf'
# ---------------------

X = CDMFT(
    model,
    method="trf",
    jac=True,
    varia=varia,
    iteration="fixed_point",
)
