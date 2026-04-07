# Test file
# Goal : to test CDMFT in a 1-cluster model with two sub-baths
#--------------------------------------------------------------------------------
"""Test SB-CDMFT for the 1x4 chain hybridized with two 4-orbital
subbaths each belonging to an irrep of C2 point group.
"""

import pyqcm
from pyqcm.cdmft import CDMFT

# Number of orbitals of the models
ns, nb = 4, 4
no = ns + nb

CMA = pyqcm.cluster_model(4, 4, 'CMA', generators=([4,3,2,1,0,0,0,0],), bath_irrep=True)
CMB = pyqcm.cluster_model(4, 4, 'CMB', generators=([4,3,2,1,1,1,1,1],), bath_irrep=True)

for i in range(1, nb + 1):
    lab = i + ns
    CMA.new_operator("eb{:d}".format(i), "one-body", [(lab, lab, 1), (lab + no, lab + no, 1)])
    CMB.new_operator("eb{:d}".format(i), "one-body", [(lab, lab, 1), (lab + no, lab + no, 1)])
    elemA = [(1, lab, 1), (ns, lab, 1), (1 + no, lab + no,1),(ns + no, lab + no, 1)]
    elemB = [(1, lab, 1), (ns, lab,-1), (1 + no, lab + no,1),(ns + no, lab + no,-1)]
    CMA.new_operator("tb{:d}".format(i), "one-body", elemA)
    CMB.new_operator("tb{:d}".format(i), "one-body", elemB)

varia = []
for k in range(1,3):
    varia += ["eb{:d}_{:d}".format(i, k) for i in range(1, nb + 1)]
    varia += ["tb{:d}_{:d}".format(i, k) for i in range(1, nb + 1)]

clus = pyqcm.cluster((CMA, CMB), [[i, 0, 0] for i in range(ns)])
model = pyqcm.lattice_model("1x4_4b_2sb", clus, [[ns, 0, 0]])

model.varia = varia
model.interaction_operator("U")
model.hopping_operator("t", (1, 0, 0), -1)

model.set_target_sectors(["R0:N8:S0", "R0:N8:S0"])

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
"""

model.set_parameters(lattice_params + bath_params)

X = CDMFT(
    model,
    varia,
    accur_bath=1e-6,
    accur_dist=1e-12,
    method="PRAXIS",
    iteration="fixed_point",
)

