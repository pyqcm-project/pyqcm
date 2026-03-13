"""Test SB-CDMFT for the 2x2 plaquette hybridized with four 4-orbital
subbaths each belonging to an irrep of C2v point group.
"""

import pyqcm
from pyqcm.cdmft import CDMFT

# Number of orbitals of the models
n_sites, n_baths = 4, 4
no = n_sites + n_baths

# The 4 coupling sequences correspond to the 4 irreps of C2v
irreps = {
    "A1": [1, 1, 1, 1],
    "A2": [1, 1, -1, -1],
    "B1": [1, -1, 1, -1],
    "B2": [1, -1, -1, 1],
}
cluster_models = {}

for sb_name, irrep in irreps.items():
    cluster_model = pyqcm.cluster_model(
        n_sites, n_baths, name="2x2_4b_{:s}".format(sb_name)
    )
    for i in range(1, n_baths + 1):
        lab = i + n_sites
        cluster_model.new_operator(
            "eb{:d}".format(i), "one-body", [(lab, lab, 1), (lab + no, lab + no, 1)]
        )
        elem = [(j + 1, lab, irrep[j]) for j in range(n_sites)] + [
            (j + 1 + no, lab + no, irrep[j]) for j in range(n_sites)
        ]
        cluster_model.new_operator("tb{:d}".format(i), "one-body", elem)
    cluster_models[sb_name] = cluster_model

cluster_model_A1, cluster_model_A2, cluster_model_B1, cluster_model_B2 = (
    cluster_models["A1"],
    cluster_models["A2"],
    cluster_models["B1"],
    cluster_models["B2"],
)

varia = []
for k in range(1, n_baths + 1):
    varia += ["eb{:d}_{:d}".format(i, k) for i in range(1, n_baths + 1)]
    varia += ["tb{:d}_{:d}".format(i, k) for i in range(1, n_baths + 1)]

clus = pyqcm.cluster(
    (cluster_model_A1, cluster_model_A2, cluster_model_B1, cluster_model_B2),
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
)
model = pyqcm.lattice_model("2x2_4b_4sb", clus, [[2, 0, 0], [0, 2, 0]])

model.varia = varia
model.interaction_operator("U")
model.hopping_operator("t", (1, 0, 0), -1)
model.hopping_operator("t", (0, 1, 0), -1)

model.set_target_sectors(["R0:N8:S0", "R0:N8:S0", "R0:N8:S0", "R0:N8:S0"])

lattice_params = """
    U = 8
    t = 1
    mu = 4
"""

bath_params = """
    eb1_1 = 1.0
    eb2_1 = -1.0
    eb3_1 = 1.0
    eb4_1 = -1.0
    tb1_1 = 0.5
    tb2_1 = 0.5
    tb3_1 = 0.5
    tb4_1 = 0.5
    eb1_2 = 1.0
    eb2_2 = -1.0
    eb3_2 = 1.0
    eb4_2 = -1.0
    tb1_2 = 0.5
    tb2_2 = 0.5
    tb3_2 = 0.5
    tb4_2 = 0.5
    eb1_3 = 1.0
    eb2_3 = -1.0
    eb3_3 = 1.0
    eb4_3 = -1.0
    tb1_3 = 0.5
    tb2_3 = 0.5
    tb3_3 = 0.5
    tb4_3 = 0.5
    eb1_4 = 1.0
    eb2_4 = -1.0
    eb3_4 = 1.0
    eb4_4 = -1.0
    tb1_4 = 0.5
    tb2_4 = 0.5
    tb3_4 = 0.5
    tb4_4 = 0.5
"""

model.set_parameters(lattice_params + bath_params)

X = CDMFT(
    model,
    varia,
    accur_bath=1e-6,
    accur_dist=1e-12,
    method="BOBYQA",
    iteration="fixed_point",
)

X.I.cluster_spectral_function()
X.I.spectral_function()
