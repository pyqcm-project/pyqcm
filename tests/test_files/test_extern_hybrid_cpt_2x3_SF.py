# Test file
# Goal : writing an external hybridization file in hdf5 format, to be read back immediately and compared
# First, the 3-band model is defined, with a cluster of 4 Cu sites (with interaction) and another, noninteracting cluster of 8 oxygen sites
# Second, the file is read by a second model containing only Cu sites.
# This version of the test is CPT only (the Cu cluster has no bath attached to it)
#--------------------------------------------------------------------------------

import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT
import h5py

#--------------------------------------------------------------------------------
# Definition of the model
def Emery_model(oxygen=False, hybrid_file = None):
    """
    param bool oxygen : True if the model contains oxygen atoms
    param hybrid_file : name of the HD5 file to be written or read
    """
    Cu_sites = ((0, 0, 0), (2, 0, 0), (4, 0, 0), (0, 2, 0), (2, 2, 0), (4, 2, 0))
    Cu = pyqcm.cluster_model(6, name='Cu6')
    clus_Cu = pyqcm.cluster(Cu, Cu_sites)
    if oxygen:
        O_sites  = ((1, 0, 0), (3, 0, 0), (5, 0, 0), (0, 1, 0), (2, 1, 0), (4, 1, 0), (1, 2, 0), (3, 2, 0), (5, 2, 0), (0, 3, 0), (2, 3, 0), (4, 3, 0))
        O = pyqcm.cluster_model(12, name='O12')
        clus_O = pyqcm.cluster(O, O_sites)
        model = pyqcm.lattice_model('Cu6_O12', (clus_Cu, clus_O), ((6,0,0), (0,4,0)), ((2,0,0), (0,2,0)))
        model.hopping_operator('tpd', (1, 0, 0), 1, orbitals=(1,2))
        model.hopping_operator('tpd', (1, 0, 0), 1, orbitals=(2,1))
        model.hopping_operator('tpd', (0, 1, 0), 1, orbitals=(1,3))
        model.hopping_operator('tpd', (0, 1, 0), 1, orbitals=(3,1))
        model.hopping_operator('e', (0, 0, 0), 1,orbitals=(2,2),tau=0,sigma=0)
        model.hopping_operator('e', (0, 0, 0), 1,orbitals=(3,3),tau=0,sigma=0)
        model.hopping_operator('tpp', ( 1, 1, 0), 1, orbitals=(2,3))
        model.hopping_operator('tpp', ( 1, 1, 0), 1, orbitals=(3,2))
        model.hopping_operator('tpp', (-1, 1, 0), 1, orbitals=(2,3))
        model.hopping_operator('tpp', (-1, 1, 0), 1, orbitals=(3,2))
    else:
        model = pyqcm.lattice_model('Cu6', (clus_Cu,), ((6,0,0), (0,4,0)), ((2,0,0), (0,2,0)), hybrid_file=hybrid_file)
    model.interaction_operator('U', orbitals=(1,1))
    model.hopping_operator('tc', (2, 0, 0), -1, orbitals=(1,1)) 
    model.hopping_operator('tc', (0, 2, 0), -1, orbitals=(1,1)) 
    model.anomalous_operator('D', (2, 0, 0), 1, orbitals=(1,1)) 
    model.anomalous_operator('D', (0, 2, 0),-1, orbitals=(1,1)) 
    model.hopping_operator('ed', (0, 0, 0), 1,orbitals=(1,1),tau=0,sigma=0)
    model.set_basis(((0.5,0,0), (0,0.5,0), (0,0,0.5)))
    return model

#--------------------------------------------------------------------------------
# 1st part : writing the external hybridization from the model with oxygens

model = Emery_model(oxygen = True)
model.set_target_sectors(['N6:S0', 'N12:S0'])

band_params = """
U = 14
e = 2.5
mu = 10.5
tpd = 2.1
tpp = 1
tc = 0.1
ed = 1e-9
# D = 1e-9
"""
dim = 6
model.set_parameters(band_params)

# defining a frequency grid along the reals axis
wr = np.arange(-15,5,0.05); weight = np.ones_like(wr)*0.05; eta = 0.2
nw = wr.shape[0]

# defining a wavevector grid
nk_side = 8 # number of k points on the side of the internal averaging grid

# defining a wavevector path along a triangular path Gamma-(pi,0)-(pi,pi)-Gamma
# (components in the [0,1] interval, superdual basis, like the original grid)
k, tick_pos, tick_str = pyqcm.wavevector_path(n=32, shape='triangle')
nk = k.shape[0]

pyqcm.set_global_parameter('kgrid_side', nk_side) # necessary if averages are computed on a grid
pyqcm.qcm.frequency_grid(wr, weight)
I = pyqcm.model_instance(model)

# computing averages (for future comparison)
A = I.averages()
n_Cu = 3*A['ed']
tc_ave = 3*A['tc']
print('n_Cu = ', n_Cu)  # *3 because 3 atoms per unit cell in the Emery model
print('<tc> = ', tc_ave)

# I.cluster_averages(pr=True)
I.GS_consistency(check_ground_state=True)

# Now computing the external hybridization file and writing in HF5 file
dim_red = dim  # dimension of the Cu part of the Green function
dim_tot = model.dimGF  # total dimension of the Green function
dim_O = dim_tot - dim_red # dimension of the oxygen part of the Green function

hybrid = np.zeros((nw, nk, dim_red, dim_red), dtype=complex) # allocating the hybridization function
id = np.identity(dim_O)

for iw in range(nw):
    for ik in range(nk):
        # IMPORTANT : the k-grid that is stored in the file is defined in the [0,1] interval in the superdual basis (\tilde k)
        # The k-value used in the V_matrix() function is in the physical basis, in multiples of 2pi.
        # Hence one must divide by 3 along x and by 2 along y
        # because the unit cell has length 2, it is important to use set_basis() correctly to compendate for that
        K = np.copy(k[ik])
        K[0] *= 0.3333333333
        K[1] *= 0.5
        z = wr[iw] + 1j*eta  # complex frequency on the real axis (must match the reader's convention)
        Vc = I.V_matrix(z, K)
        V = Vc[0:dim_red, dim_red:]
        H2 = I.cluster_hopping_matrix(clus=1) + Vc[dim_red:, dim_red:]
        G2 = np.linalg.inv(z*id-H2)
        hybrid[iw, ik, :, :] = V@G2@np.conjugate(V.T)  # V.(z - H2)^{-1}.V^T

iw = 10
ik = 2
K = np.copy(k[ik])
print('K (grid) = ', K)
K[0] *= 0.3333333333
K[1] *= 0.5
print('K (phys) = ', K)
G = I.CPT_Green_function(wr[iw] + 1j*eta,K)
# print('gamma\n', hybrid[iw, ik, :, :])
print('Gcpt\n', G[0:dim, 0:dim])

# writing the result in a HDF5 file
with h5py.File('hybrid.h5', "w") as f:
    f.create_dataset("eta", data = eta, dtype=float)
    f.create_dataset("w", data = wr, dtype=wr.dtype)
    f.create_dataset("weight", data = weight, dtype=weight.dtype)
    f.create_dataset("k", data = k, dtype=k.dtype)
    f.create_dataset("hybrid_real", data=hybrid.real, dtype=float)
    f.create_dataset("hybrid_imag", data=hybrid.imag, dtype=float)
    f.create_dataset("mixing", data=model.mixing, dtype=int)


#--------------------------------------------------------------------------------
# 2nd part : reading the external hybridization for the model with Cu atoms only

pyqcm.reset_model()

model = Emery_model(hybrid_file = 'hybrid.h5')

model.set_target_sectors('N6:S0')
# model.set_target_sectors('S0')
band_params = """
U = 14
mu = 10.5
tc = 0.1
ed = 1e-9
# D = 1e-9
"""

model.set_parameters(band_params)

I = pyqcm.model_instance(model)
# The file is on the real frequency axis (eta != 0): lattice averages, which require the imaginary axis,
# cannot be computed from it and must be refused
try:
    I.averages()
except RuntimeError as ex:
    print('averages refused as expected: ', ex)
else:
    raise AssertionError('averages should be refused with a real-axis external hybridization')

# I.cluster_averages(pr=True)
print('Gcpt\n', I.CPT_Green_function_grid(iw,ik))

I.spectral_function(file='test_extern_hybrid_cpt_2x3_SF.pdf')


