# Test file
# Goal : round-trip the cluster Green function through the TRIQS converter layer
#        (pyqcm.TRIQS_converter) and check it against direct pyqcm evaluation.
#
# This test is skipped (exit 0) if TRIQS is not installed, so it is safe to keep
# in the suite on machines without TRIQS.
# --------------------------------------------------------------------------------
import sys
import numpy as np

try:
    from triqs.gf import MeshImFreq, MeshReFreq
except ImportError:
    print("TRIQS not installed -- skipping test_TRIQS_converter")
    sys.exit(0)

import pyqcm
from pyqcm import TRIQS_converter as tc

# --------------------------------------------------------------------------------
# A minimal 2-site Hubbard cluster, no bath, normal mixing (mixing = 0)

clus = pyqcm.cluster_model(2, name="clus")
clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)))
model = pyqcm.lattice_model("1D_2site", [clus0], ((2, 0, 0),))
model.interaction_operator("U")
model.hopping_operator("t", (1, 0, 0), -1)

model.set_target_sectors(["N2:S0"])
model.set_parameters("""
    U = 4
    mu = 2
    t = 1
""")

I = pyqcm.model_instance(model)
I.Green_function_solve()

assert model.mixing == 0, "this test assumes normal mixing"
d = int(model.dimGFC[0])

# --------------------------------------------------------------------------------
# 1) Matsubara mesh: exact Lehmann build vs sampled build vs direct evaluation

beta = 50.0
mesh = MeshImFreq(beta=beta, S="Fermion", n_iw=200)

G_lehmann = tc.green_function(I, mesh, clus=0, from_lehmann=True)
G_sampled = tc.green_function(I, mesh, clus=0, from_lehmann=False)
Sigma = tc.self_energy(I, mesh, clus=0)
Hyb = tc.hybridization(I, mesh, clus=0)

# block structure: normal mixing -> 'up' and 'down', identical
names = sorted(name for name, _ in G_lehmann)
assert names == ["down", "up"], names
for name, g in G_lehmann:
    assert g.data.shape[1:] == (d, d), (name, g.data.shape)

err_lehmann = 0.0
err_sampled = 0.0
err_spinsym = 0.0
err_dyson = 0.0
for w in mesh:
    z = complex(w)
    Gdir = np.asarray(I.cluster_Green_function(z, clus=0))
    err_lehmann = max(err_lehmann, np.max(np.abs(G_lehmann["up"][w] - Gdir)))
    err_sampled = max(err_sampled, np.max(np.abs(G_sampled["up"][w] - Gdir)))
    # up and down must coincide for normal mixing
    err_spinsym = max(err_spinsym, np.max(np.abs(G_lehmann["up"][w] - G_lehmann["down"][w])))
    # Dyson consistency: G^{-1} = z + mu - t(k=0 cluster) - Sigma  is not assumed here;
    # instead just check Sigma was sampled to match the direct call
    Sdir = np.asarray(I.cluster_self_energy(z, clus=0))
    err_dyson = max(err_dyson, np.max(np.abs(Sigma["up"][w] - Sdir)))

print("max |G_lehmann - G_direct|  =", err_lehmann)
print("max |G_sampled - G_direct|  =", err_sampled)
print("max |G_up - G_down|         =", err_spinsym)
print("max |Sigma - Sigma_direct|  =", err_dyson)

assert err_lehmann < 1e-9, "Lehmann reconstruction disagrees with direct evaluation"
assert err_sampled < 1e-12, "sampled reconstruction disagrees with direct evaluation"
assert err_spinsym < 1e-12, "spin up/down blocks differ for normal mixing"
assert err_dyson < 1e-12, "self-energy block disagrees with direct evaluation"

# --------------------------------------------------------------------------------
# 2) Real-frequency mesh with broadening

rmesh = MeshReFreq(window=(-6.0, 6.0), n_w=400)
eta = 0.05
Gr = tc.green_function(I, rmesh, clus=0, eta=eta, from_lehmann=True)
err_real = 0.0
for w in rmesh:
    z = complex(w).real + 1j * eta
    Gdir = np.asarray(I.cluster_Green_function(z, clus=0))
    err_real = max(err_real, np.max(np.abs(Gr["up"][w] - Gdir)))
print("max |G_real - G_direct|     =", err_real)
assert err_real < 1e-9, "real-axis Lehmann reconstruction disagrees with direct evaluation"

# --------------------------------------------------------------------------------
# 3) TRIQS -> pyqcm extraction helpers round-trip

z_arr = tc.mesh_to_z(mesh)
assert z_arr.shape[0] == len([_ for _ in mesh])
d_dict = tc.blockgf_to_dict(G_lehmann)
assert set(d_dict.keys()) == {"up", "down"}
z_blk, vals = d_dict["up"]
assert vals.shape[1:] == (d, d)
# the extracted values must equal the Gf data
assert np.max(np.abs(vals - G_lehmann["up"].data)) < 1e-14

print("\ntest_TRIQS_converter: all checks passed")
