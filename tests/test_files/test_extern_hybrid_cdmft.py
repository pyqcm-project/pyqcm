# Test file
# Goal : writing an external hybridization file in hdf5 format, to be read back immediately and compared
# First, the 3-band model is defined, with a cluster of 4 Cu sites (with interaction) and another, noninteracting cluster of 8 oxygen sites
# Second, the file is read by a second model containing only Cu sites.
# This version of the test is for CDMFT only (the Cu cluster has a bath attached to it)
#--------------------------------------------------------------------------------

import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT, frequency_grid
import h5py

#--------------------------------------------------------------------------------
# Definition of the model
def Emery_model(oxygen=False, hybrid_file = None):
    """
    param bool oxygen : True if the model contains oxygen atoms
    param hybrid_file : name of the HD5 file to be written or read
    """

    # defining the Cu cluster model, with bath
    ns = 4
    nb = 6 # number of bath sites
    no = ns + nb
    Cu_CM = pyqcm.cluster_model(ns, nb, name='Cu4_6b_C2v', generators=((2,1,4,3,0,0,0,2,2,2), (3,4,1,2,0,0,2,0,2,2)), bath_irrep=True)

    # bath energies
    for i in range(1,nb+1):
        name = 'eb'+str(i)
        lab = i+ns
        Cu_CM.new_operator(name, 'one-body',(
            (lab,lab,1.0),
            (lab + no,lab + no,1.0)
        ))

    # bath hybridizations
    def new_hyb(x,seq):
        global var_H, var_SC
        elem = []
        for i in range(ns):
            elem.append((i+1,x+ns,seq[i]))
        for i in range(ns):
            elem.append((i+1+no,x+ns+no,seq[i]))
        name = 'tb'+str(x)
        Cu_CM.new_operator(name, 'one-body', elem)

    new_hyb(1,(1, 1, 1, 1))
    new_hyb(2,(1, 1, 1, 1))
    new_hyb(3,(1, 1,-1,-1))
    new_hyb(4,(1,-1, 1,-1))
    new_hyb(5,(1,-1,-1, 1))
    new_hyb(6,(1,-1,-1, 1))

    Cu_sites = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0))
    Cu_cluster = pyqcm.cluster(Cu_CM, Cu_sites)

    if oxygen:
        O_sites  = ((1, 0, 0),(3, 0, 0), (0, 1, 0), (2, 1, 0), (1, 2, 0),(3, 2, 0), (0, 3, 0), (2, 3, 0))
        O = pyqcm.cluster_model(8, name='O8')
        O_cluster = pyqcm.cluster(O, O_sites)
        model = pyqcm.lattice_model('Cu4_O8_6b', (Cu_cluster, O_cluster), ((4,0,0), (0,4,0)), ((2,0,0), (0,2,0)))
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
        model = pyqcm.lattice_model('Cu4_6b', (Cu_cluster,), ((4,0,0), (0,4,0)), ((2,0,0), (0,2,0)), hybrid_file=hybrid_file)
    model.interaction_operator('U', orbitals=(1,1))
    model.hopping_operator('tc', (2, 0, 0), -1, orbitals=(1,1)) 
    model.hopping_operator('tc', (0, 2, 0), -1, orbitals=(1,1)) 
    model.hopping_operator('ed', (0, 0, 0), 1,orbitals=(1,1),tau=0,sigma=0)
    model.set_basis(((0.5,0,0), (0,0.5,0), (0,0,0.5)))
    return model

#--------------------------------------------------------------------------------
# 1st part : writing the external hybridization from the model with oxygens

model = Emery_model(oxygen = True)

model.set_target_sectors(['N10:S0', 'N8:S0'])

band_params = """
U = 14
e = 2.5
mu = 10.5
tpd = 2.1
tpp = 1
tc = 0.1
ed = 1e-9
"""

bath_params="""
eb1_1 = 1
eb2_1 = -1
eb3_1 = 1
eb4_1 = -1
eb5_1 = 1
eb6_1 = -1
tb1_1 = 1
tb2_1 = 1
tb3_1 = 1
tb4_1 = 1
tb5_1 = 1
tb6_1 = 1
"""

model.set_parameters(band_params+bath_params)

# defining a frequency grid along the Matsubara axis
wr, weight = pyqcm.legendre_frequency_grid(1, 5, 10, 10, 10)
nw = wr.shape[0]

# defining a wavevector grid
nk_side = 16 # number of k points on the side of the grid
nk = nk_side*nk_side
k1D = np.linspace(0, 1, nk_side, endpoint=False)
kx,ky = np.meshgrid(k1D, k1D, copy=True, sparse=False, indexing='xy')
kx = kx.reshape(nk)
ky = ky.reshape(nk)
k = np.stack((kx,ky,np.zeros(nk))).T  # *2 en raison de la base physique (0.5, 0,.5)

pyqcm.set_global_parameter('kgrid_side', nk_side) # necessary if averages on computed on a grid
pyqcm.qcm.frequency_grid(wr, weight)
I = pyqcm.model_instance(model)

# computing averages (for future comparison)
A = I.averages()
n_Cu = 3*A['ed']
tc_ave = 3*A['tc']
print('n_Cu = ', n_Cu)  # *3 because 3 atoms per unit cell in the Emery model
print('<tc> = ', tc_ave)


# Now computing the external hybridization file and writing in HF5 file
dim_red = 4  # dimension of the Cu part of the Green function
dim_tot = model.dimGF  # total dimension of the Green function
dim_O = dim_tot - dim_red # dimension of the oxygen part of the Green function

hybrid = np.zeros((nw, nk, dim_red, dim_red), dtype=complex) # allocating the hybridization function
id = np.identity(dim_O)

for iw in range(nw):
    for ik in range(nk):
        K = k[ik]/2  # /2 en raison du rapport de longueur super-amas / amas
        Vc = I.V_matrix(wr[iw]*1j, K)
        V = Vc[0:dim_red, dim_red:]
        H2 = I.cluster_hopping_matrix(clus=1) + Vc[dim_red:, dim_red:]
        G2 = np.linalg.inv(wr[iw]*1j*id-H2)
        hybrid[iw, ik, :, :] = V@G2@np.conjugate(V.T)  # V.(w - H2)^{-1}.V^T

# writing the result in a HDF5 file
with h5py.File('hybrid.h5', "w") as f:
    f.create_dataset("w", data = wr, dtype=wr.dtype)
    f.create_dataset("weight", data = weight, dtype=weight.dtype)
    f.create_dataset("k", data = k, dtype=k.dtype)
    f.create_dataset("hybrid_real", data=hybrid.real, dtype=float)
    f.create_dataset("hybrid_imag", data=hybrid.imag, dtype=float)
    f.attrs['mixing'] = model.mixing


#--------------------------------------------------------------------------------
# 2nd part : reading the external hybridization for the model with Cu atoms only

pyqcm.reset_model()

model = Emery_model(hybrid_file = 'hybrid.h5')

model.set_target_sectors('N10:S0/N12:S0/N8:S0')
band_params = """
U = 14
mu = 10.5
tc = 0.1
ed = 1e-9
"""

model.set_parameters(band_params+bath_params)

varia = [
'eb1_1',
'eb2_1',
'eb3_1',
'eb4_1',
'eb5_1',
'eb6_1',
'tb1_1',
'tb2_1',
'tb3_1',
'tb4_1',
'tb5_1',
'tb6_1'
]


sol = CDMFT(model, varia=varia, grid=frequency_grid('legendre', (1, 5, 10, 10, 10)), accur=1e-3, convergence='self-energy', maxiter=64, depth=1, iteration='fixed_point')
A = sol.I.averages(pr=True)
I = pyqcm.model_instance(model)
A = sol.I.averages(pr=True)
print('n_Cu = ', A['ed'])
n_Cu_read = A['ed']
tc_ave_read = A['tc']




