"""Test SB-CDMFT for the 1x4 chain hybridized with two 6-orbital
subbaths each belonging to an irrep of C2 point group.
"""

import pyqcm
from pyqcm.cdmft import CDMFT

# Number of orbitals of the models
n_sites, n_baths = 4, 6
no = n_sites + n_baths

# The 2 coupling sequences correspond to the 2 irreps of C2
irreps = {
    "A": [1, 1, 1, 1],
    "B": [1, 1, -1, -1],
}
cluster_models = {}

for sb_name, irrep in irreps.items():
    cluster_model = pyqcm.cluster_model(
        n_sites, n_baths, name="1x4_6b_{:s}".format(sb_name)
    )
    for i in range(1, n_baths + 1):
        lab = i + n_sites
        cluster_model.new_operator(
            "eb{:d}".format(i), "one-body", [(lab, lab, 1), (lab + no, lab + no, 1)]
        )
        elem = [
            (1, lab, irrep[0]),
            (n_sites, lab, irrep[n_sites - 1]),
            (1 + no, lab + no, irrep[0]),
            (n_sites + no, lab + no, irrep[n_sites - 1]),
        ]
        cluster_model.new_operator("tb{:d}".format(i), "one-body", elem)
    cluster_models[sb_name] = cluster_model

cluster_model_A, cluster_model_B = cluster_models["A"], cluster_models["B"]

varia = []
for k in range(1, len(irreps) + 1):
    varia += ["eb{:d}_{:d}".format(i, k) for i in range(1, n_baths + 1)]
    varia += ["tb{:d}_{:d}".format(i, k) for i in range(1, n_baths + 1)]

clus = pyqcm.cluster(
    (cluster_model_A, cluster_model_B),
    [[i, 0, 0] for i in range(n_sites)],
)
model = pyqcm.lattice_model("1x4_6b_2sb", clus, [[n_sites, 0, 0]])

model.varia = varia
model.interaction_operator("U")
model.hopping_operator("t", (1, 0, 0), -1)

model.set_target_sectors(["R0:N10:S0", "R0:N10:S0"])

lattice_params = """
    U = 4
    t = 1
    mu = 2
"""

bath_params = """
    eb1_1 = 1.0
    eb2_1 = -1.0
    eb3_1 = 1.0
    eb4_1 = -1.0
    eb5_1 = 1.0
    eb6_1 = -1.0
    tb1_1 = 0.5
    tb2_1 = 0.5
    tb3_1 = 0.5
    tb4_1 = 0.5
    tb5_1 = 0.5
    tb6_1 = 0.5
    eb1_2 = 1.0
    eb2_2 = -1.0
    eb3_2 = 1.0
    eb4_2 = -1.0
    eb5_2 = 1.0
    eb6_2 = -1.0
    tb1_2 = 0.5
    tb2_2 = 0.5
    tb3_2 = 0.5
    tb4_2 = 0.5
    tb5_2 = 0.5
    tb6_2 = 0.5
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
