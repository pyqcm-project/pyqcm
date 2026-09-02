# ==================================================================================================
# Validation of the "cellularization" of an external hybridization function.
#
# Physical setup: the three-band Emery model of the CuO2 plane. Each unit cell holds one Cu orbital,
# which carries the interaction U, and two oxygen orbitals (Ox, Oy), which are non interacting and
# only provide hybridization paths between the Cu.
#
# Question answered by this script: the oxygens can be eliminated exactly and replaced by a
# frequency- and wavevector-dependent hybridization function Gamma(K,w) attached to a Cu-only
# lattice model. Can a Gamma computed on a SMALL cluster be refactored ("cellularized") onto a
# BIGGER cluster, and does the result still reproduce the original model exactly?
#
# The script proceeds in five stages:
#
#   LM0   4 Cu + 8 O, 2x2 supercell        -> the exact CPT reference
#   LM1   1 Cu + 2 O, 1x1 supercell        -> the model Gamma is extracted from
#           -> hybrid.h5   : Gamma^(1x1)(k,w), tabulated over the full Cu Brillouin zone
#           -> hybridc.h5  : Gamma^(2x2)(K,w), obtained by cellularizing the above
#   LM2   4 Cu only + hybridc.h5           -> must reproduce LM0's CPT Green function exactly
#
#   Emery_2x2+O   4 Cu + 6 bath + 8 O            -> CDMFT reference
#   Emery_2x2     4 Cu + 6 bath + hybridc.h5     -> must reproduce the CDMFT reference
#
# Why this works at finite U (the subtle point). Gamma is extracted below as V.(G^-1)_OO^-1.V^dag
# from the INVERSE CPT Green function of LM1. Since U lives on the Cu orbital only, the self-energy
# has a single non-zero element, in the Cu-Cu block; the O-O and Cu-O blocks of G_cpt^-1 are
# therefore exactly non interacting. The extracted Gamma is thus a pure one-body object, independent
# of U and of the cluster it was obtained from -- which is precisely what makes it transferable to a
# plaquette model carrying a completely different self-energy.
#
# Coordinates: every site and link coordinate below is expressed in the working basis defined by
# set_basis(((0.5,0,0),(0,0.5,0),(0,0,1))), i.e. in units of HALF a Cu-Cu distance. Cu therefore sit
# at even coordinates and O at odd ones (the oxygens occupy the Cu-Cu midpoints).
# ==================================================================================================

import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT, frequency_grid

# pyqcm.set_global_parameter('temperature', 0.01)
pyqcm.warnings = True  # we want to see the ground-state consistency warnings

# the 2x2 Cu plaquette, and the 8 oxygens that surround it (working basis: see header)
Cu_sites = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0))
O_sites  = ((-1, 0, 0),(1, 0, 0), (0,-1, 0), (0, 1, 0), (1, 2, 0),(3, 2, 0), (2, 1, 0), (2, 3, 0))

# Cu parameters. U sits on the Cu orbital only, which is what makes the elimination of the oxygens
# exact. tc is a direct Cu-Cu hopping, deliberately kept tiny.
band_params_Cu="""
U = 12  # 12
mu = 6   # 10
ed = 0.0
tc = 0.01
"""

# Oxygen parameters: on-site energy and the Cu-O / O-O hoppings. These are exactly the terms that
# get folded into Gamma(K,w) when the oxygens are integrated out.
band_params_O = """
tpd = 2.1
tpp = 1
tppp = 0.2
e = 2.5
"""

# == mutable parameters =====================================================
nk_side = 16 # number of k points on the side of the grid
#============================================================================

# frequency grid ------------------------------------------------------------

# A SINGLE frequency grid object is used throughout the script: it defines the frequencies at
# which the external hybridization is tabulated in the h5 files AND the frequencies of the CDMFT
# distance function. The two MUST coincide: with an external hybridization, CDMFT_host() builds
# the host at the frequencies stored in the h5 file, whereas the distance function and the
# bath hybridization are evaluated on the grid passed to CDMFT(). Using two different grids
# would minimize a distance between quantities taken at different frequencies; CDMFT_host()
# now rejects that, but keeping a single grid object here makes it impossible in the first place.
cdmft_grid = frequency_grid('regular', specs=(10, 50, 10))
wgrid = (cdmft_grid.wr, cdmft_grid.weight)

# Two possible frequency contours. The Matsubara branch (real_axis = False) is the one used for
# CDMFT; the real-axis branch is only useful to inspect spectral functions, since the CDMFT
# distance function is defined on the imaginary axis. eta is the imaginary offset of the contour
# and is stored in the h5 files so that the C++ side knows which contour Gamma was tabulated on.
real_axis = False
if real_axis:
    wr = np.arange(-5,5,0.03); weight = np.ones_like(wr)*0.05; 
    eta = 0.1
    wc = wr + eta*1j
else:
    wr, weight = wgrid
    eta = 0.0
    wc = wr*1j
 
nw = wr.shape[0]

# momentum grid ------------------------------------------------------------

# default wavevector grid (used by averages() and by the CDMFT host integration) and default
# frequency grid for the discrete frequency integrals: both must be set before any average is taken.
pyqcm.set_wavevector_grid(nk_side,nk_side,1)
pyqcm.discrete_integration_grid(*wgrid)

# we construct the momentum grid for the cluster system
nkc = nk_side*nk_side  # number of k points in the reduced Brillouin zone
nk = 4*nkc  # number of k points in the original Brillouin zone
k1D = np.linspace(0, 0.5, nk_side, endpoint=False)
kx,ky = np.meshgrid(k1D, k1D, copy=True, sparse=False, indexing='xy')
kx = kx.reshape(nkc)
ky = ky.reshape(nkc)
# kc spans the REDUCED (2x2) Brillouin zone, i.e. [0,0.5)^2, in the units accepted by
# CPT_Green_function_inverse() (physical reciprocal basis: the Cu zone is [0,1)^2).
# NB: the h5 file expects the *superlattice* reciprocal coordinates instead, i.e. 2*kc for the 2x2 model.
kc = np.stack((kx,ky,np.zeros(nkc))).T

# kc, tick_pos, tick_str = pyqcm.wavevector_path(n=32, shape='triangle')
# nkc = kc.shape[0]
# nk = 4*nkc  # number of k points in the original Brillouin zone

# we construct the momentum grid for the single site system from the cluster one
# The 1x1 model has no superlattice enlargement, so its reduced zone IS the full Cu zone [0,1)^2.
# We tile that zone as the union of the four translates kc + Q, Q being the 4 reciprocal
# superlattice vectors of the 2x2 supercell. The ordering matters: cellularize2x2() below relies on
# block j of this array holding kc + Q[j].
k = np.empty((nk, 3))
k[0:nkc, :] = kc
b = 0.5  # (pi,0) in these units: the Cu Brillouin zone is [0,1)^2
k[nkc:2*nkc, :] = kc + b*np.array((1, 0, 0))       # (pi,0)
k[2*nkc:3*nkc, :] = kc + b*np.array((0, 1, 0))     # (0,pi)
k[3*nkc:, :] = kc + b*np.array((1, 1, 0))          # (pi,pi)
# k *= 0.5;  # pour ramener au domaine standard # NON!

# indices of the frequency and of the wavevector used for the end-of-script comparison.
# NB: do NOT use k_test_label = 0 : at K=0 the K-dependent phase of the cellularization
# formula is 1, so the Gamma point cannot detect an error in that phase.
iw_test = 5
k_test_label = 37


#============================================================================
# Defining operators
#
# Orbital legend, identical in every model built below (one unit cell = 3 orbitals):
#     orbital 1 = Cu           at (0,0,0) in the cell
#     orbital 2 = Ox           at (1,0,0)      "
#     orbital 3 = Oy           at (0,1,0)      "
# tau=0 turns a (0,0,0) "hopping" into a pure on-site energy (a density operator) rather than a bond.

def model_operators(model):
    model.interaction_operator('U', orbitals=(1,1))                             # Hubbard U on Cu
    model.hopping_operator('ed', (0, 0, 0), 1,orbitals=(1,1),tau=0,sigma=0)      # Cu on-site energy
    model.hopping_operator('tc', (2, 0, 0), 1, orbitals=(1,1))                   # direct Cu-Cu hopping, x
    model.hopping_operator('tc', (0, 2, 0), 1, orbitals=(1,1))                   #        "            , y
    model.hopping_operator('e', (0, 0, 0), 1,orbitals=(2,2),tau=0,sigma=0)       # O on-site energy, Ox
    model.hopping_operator('e', (0, 0, 0), 1,orbitals=(3,3),tau=0,sigma=0)       #        "        , Oy
    model.hopping_operator('tpd', (1, 0, 0), 1, orbitals=(1,2))                  # Cu-Ox
    model.hopping_operator('tpd', (1, 0, 0), 1, orbitals=(2,1))
    model.hopping_operator('tpd', (0, 1, 0), 1, orbitals=(1,3))                  # Cu-Oy
    model.hopping_operator('tpd', (0, 1, 0), 1, orbitals=(3,1))
    model.hopping_operator('tpp', ( 1, 1, 0), 1, orbitals=(2,3))                 # Ox-Oy, both diagonals
    model.hopping_operator('tpp', ( 1, 1, 0), 1, orbitals=(3,2))
    model.hopping_operator('tpp', (-1, 1, 0), 1, orbitals=(2,3))
    model.hopping_operator('tpp', (-1, 1, 0), 1, orbitals=(3,2))
    model.hopping_operator('tppp', (2, 0, 0),-1, orbitals=(2,2))                 # O-O, 2nd neighbours
    model.hopping_operator('tppp', (0, 2, 0),-1, orbitals=(3,3))

# Same model, with the oxygens removed: this is the operator set of every Cu-only lattice model,
# in which the whole oxygen sector is carried instead by the external hybridization Gamma(K,w).
def model_operators_reduced(model):
    model.interaction_operator('U', orbitals=(1,1)) 
    model.hopping_operator('ed', (0, 0, 0), 1, orbitals=(1,1), tau=0, sigma=0)
    model.hopping_operator('tc', (2, 0, 0), 1, orbitals=(1,1))
    model.hopping_operator('tc', (0, 2, 0), 1, orbitals=(1,1))

#============================================================================
# Defining the reference 4-site model, with a 4-site Cu cluster and an 8-site oxygen cluster
#
# LM0 is the yardstick: everything that follows must reproduce its Cu-Cu Green function. Its
# supercell is 2x2 unit cells (superlattice (4,0,0),(0,4,0) in the working basis) and is tiled by
# two clusters: the 4 Cu, which carry U, and the 8 surrounding O, which do not.

CM_Cu = pyqcm.cluster_model(4, 0, name='CM_Cu')
clus_Cu = pyqcm.cluster(CM_Cu, Cu_sites)
CM_O = pyqcm.cluster_model(8, 0, name='CM_O')
clus_O = pyqcm.cluster(CM_O, O_sites)
model = pyqcm.lattice_model('LM0', (clus_Cu,clus_O), ((4,0,0),(0,4,0)), ((2,0,0),(0,2,0)))
model.set_basis(((0.5,0,0),(0,0.5,0),(0,0,1)))
model_operators(model)

# one sector list per cluster. The Cu cluster is scanned around half filling (N4) plus the doped
# sectors; the O cluster carries no interaction, so pyqcm solves it as an uncorrelated one-body
# problem and its sector specification is nominal.
model.set_target_sectors(['N4:S0/N6:S0/N2:S0', 'N8:S0'])

model.set_parameters(band_params_Cu+band_params_O)
I = model.model_instance()
I.GS_consistency()
# The factor of 3 converts the per-orbital normalization of averages() into a per-Cu density:
# this model has 3 orbitals (Cu, Ox, Oy) per unit cell. Cu-only models below need no such factor.
print('\naverages on the reference, 4*Cu+8*O model (corrected for factor of 3):\n')
ave = I.averages(pr=False)
ave = I.averages()
print('<ed> = ', 3*ave['ed'])
print('<tc> = ', 3*ave['tc'])

# the Cu-Cu block of the CPT Green function of the reference model: this is what the
# 2x2 + cellularized-hybridization model must reproduce, self-energy included.
Gcpt_LM0 = I.CPT_Green_function(wc[iw_test], kc[k_test_label])[0:4, 0:4].copy()

# I.spectral_function(w=15,orb=1)
# exit()

#============================================================================
# Defining the "single unit cell" model (one Cu site, 2 O sites)
#
# LM1 is the model Gamma is extracted from: a single unit cell, hence no superlattice enlargement
# (superlattice = unit cell = (2,0,0),(0,2,0)) and a reduced Brillouin zone equal to the full Cu
# zone [0,1)^2 -- which is exactly the array k built above.

pyqcm.reset_model()

CM1 = pyqcm.cluster_model(3, 0, name='CM1')
clus1 = pyqcm.cluster(CM1, ((0,0,0), (1,0,0), (0,1,0)))   # Cu, Ox, Oy
model = pyqcm.lattice_model('LM1', clus1, ((2,0,0),(0,2,0)), ((2,0,0),(0,2,0)))
model.set_basis(((0.5,0,0),(0,0.5,0),(0,0,1)))
model_operators(model)

# 3 sites = 6 spin-orbitals. Only even N is compatible with S0; the oxygens being deep, the
# relevant filling sits above N3, hence N4 and N6 in the list.
model.set_target_sectors(['N4:S0/N6:S0/N2:S0/N0:S0'])

model.set_parameters(band_params_Cu+band_params_O)
I = model.model_instance()
I.ground_state(pr=True)
I.GS_consistency()
print('\naverages on the single-unit-cell model (corrected for factor of 3):\n')

pyqcm.set_wavevector_grid(2*nk_side,2*nk_side,1) # for integration with the single-unit-cell model. 4 x the number of k-points

ave = I.averages()
print('<ed> = ', 3*ave['ed'])
print('<tc> = ', 3*ave['tc'])

# exit()
#============================================================================
# Computing and writing the external hybridization function from the solution
#
# Gamma^(1x1)(k,w) = G^-1_{Cu,O} . [G^-1_{O,O}]^-1 . G^-1_{O,Cu}, all blocks taken from the INVERSE
# CPT Green function of LM1. Because U acts on the Cu orbital only, the self-energy contributes to
# the Cu-Cu element alone: the three blocks used here are strictly non interacting, so the Gamma
# obtained is a one-body object valid for ANY value of U (see header).

import h5py


# the BZ of LM1 is 4x that of the 2x2 models, hence the doubled grid side, so that the k-space
# resolution matches. qcm.frequency_grid() sets the GLOBAL integration grid used by averages().
pyqcm.set_global_parameter('kgrid_side', 2*nk_side)
pyqcm.qcm.frequency_grid(wr, weight)

dim_red = 1  # 1 site
dim_tot = model.dimGF
dim_O = dim_tot - dim_red

hybrid = np.zeros((nw, nk, dim_red, dim_red), dtype=complex)
id = np.identity(dim_O)   # only used by the alternative formulation commented out below

for iw in range(nw):
    for ik in range(nk):
        iGcpt = I.CPT_Green_function_inverse(wc[iw], k[ik])
        V = iGcpt[0:dim_red, dim_red:]
        G22 = np.linalg.inv(iGcpt[dim_red:, dim_red:])
        hybrid[iw, ik, :, :] = V @ G22 @ iGcpt[dim_red:, 0:dim_red]

        # Equivalent formulation, from the bare V matrix rather than from G_cpt^-1. It gives the
        # same Gamma for the same reason as above (no self-energy in the oxygen blocks), and is
        # kept here as a cross-check.
        # if iw == nw//2: print('\n\nw = ', wc[iw], '\tK = ', k[ik], '\niGcpt = ', iGcpt.real, '\nhyb = ', hybrid[iw, ik, :, :].real)
        # Vc = I.V_matrix(wc[iw], k[ik])
        # V = Vc[0:dim_red, dim_red:]
        # H2 = I.cluster_hopping_matrix(clus=0)[dim_red:, dim_red:] + Vc[dim_red:, dim_red:]
        # G2 = np.linalg.inv(wc[iw]*id-H2)
        # hybrid[iw, ik, :, :] = V@G2@np.conjugate(V.T)

# The cellularization identity  G^(2x2)_00(K) = (1/4) sum_Q G^(1x1)(K+Q)  holds for the
# *hybridization function*, which is a one-body object. It carries over to the Green function
# only if the two models share the same self-energy, i.e. only at U = 0: here the 1x1 model is
# solved on a 3-site Cu+2O cluster while the plaquette model is solved on a 4-site Cu cluster,
# so these two Green functions are genuinely different objects as soon as U != 0.
# The reference that remains exact at any U is Gcpt_LM0 above (same 4-Cu cluster, same Sigma).
Gcpt_ref = sum(I.CPT_Green_function(wc[iw_test], k[j*nkc + k_test_label])[0,0] for j in range(4))/4
print('(1/4) Sum_Q Gcpt^(1x1)(K+Q) @ ', wc[iw_test], ' (equals the plaquette result only at U=0) :\n', Gcpt_ref)

# Layout of an external-hybridization file, as read by lattice_hybrid.cpp:
#   eta          imaginary offset of the frequency contour (0 on the Matsubara axis)
#   w, weight    the frequency grid and its integration weights          -> shape (nw,)
#   k            wavevectors in RECIPROCAL SUPERLATTICE units, range [0,1)
#                (for LM1 superlattice = unit cell, so these coincide with the physical ones)
#   hybrid_*     real and imaginary parts of Gamma                       -> shape (nw, nk, d, d)
#   mixing       the mixing state of the model that produced the file; must match the model reading it
with h5py.File("hybrid.h5", "w") as f:
    f.create_dataset("eta", data = eta, dtype=float)
    f.create_dataset("w", data = wr, dtype=wr.dtype)
    f.create_dataset("weight", data = weight, dtype=weight.dtype)
    f.create_dataset("k", data = k, dtype=k.dtype)
    f.create_dataset("hybrid_real", data=hybrid.real, dtype=float)
    f.create_dataset("hybrid_imag", data=hybrid.imag, dtype=float)
    f.create_dataset("mixing", data=model.mixing, dtype=int)


#============================================================================
# Defining the "single unit cell" model (one Cu site, no oxygens) 
# to check whether the hybridization function written in hybrid.h5 is correct (no cellularization needed)
# CHECK DONE (2026-08-21) : hybridization function correct (same averages, with the same grid and taking factor of 3 into account)

#    pyqcm.reset_model()

#    CM1 = pyqcm.cluster_model(1, 0, name='Cu1')
#    clus1 = pyqcm.cluster(CM1, ((0,0,0),))
#    model = pyqcm.lattice_model('LM1sep', (clus1,), ((2,0,0),(0,2,0)), ((2,0,0),(0,2,0)), hybrid_file = 'hybrid.h5')
#    model.set_basis(((0.5,0,0),(0,0.5,0),(0,0,1)))
#    model_operators_reduced(model) 

#    model.set_target_sectors(['N0:S0'])
#    
#    model.set_parameters(band_params_Cu)
#    I = model.model_instance()
#    print('\naverages on the single unit cell model:\n')
#    I.averages(pr=True)
#    
#    print('Gcpt:\n', I.CPT_Green_function_grid(5,12))
#    I.averages(pr=True)
#    # I.spectral_function()
#    
#    exit()


#============================================================================
# Refactoring the external hybridization (cellularization)

def cellularize2x2(input, output):
    """
    Refactors a hd5 file (input) containing an external hyrbidization file into another hd5 file (output) appropriate for a 2x2 cluster

    param str input: name of the input hd5 file
    param str input: name of the output hd5 file
    """

    dim_redc = 4  # 4 sites in a 2x2 cluster

    # reading the HDF file
    with h5py.File(input, "r") as f:
        hybrid = f["hybrid_real"][:] + 1j*f["hybrid_imag"][:]
        nw = hybrid.shape[0]
        nk = hybrid.shape[1]
        assert(nk%dim_redc == 0)   # the input grid must be the 4 blocks kc + Q built above
        nkc = nk//dim_redc

    hybridc = np.zeros((nw, nkc, dim_redc, dim_redc), dtype=complex)

    # cellularization formula:
    #
    #   Gamma^(2x2)_ab(K) = (1/L) sum_Q exp(2i.pi.(K+Q).(r_b - r_a)) Gamma^(1x1)(K+Q)
    #
    # r_a being the positions of the L=4 Cu sites of the plaquette, in physical units, and
    # Q the L reciprocal superlattice vectors of the 2x2 supercell. pyqcm's V matrix is
    # expressed in the "cluster gauge": only the superlattice vector carries a phase, hence
    # the K-dependent prefactor exp(2i.pi.K.(r_b - r_a)) which must NOT be omitted (it is
    # what distinguishes this from the trivial +/- sign combination, valid only at K=0).
    # Note that a scalar Gamma^(1x1) is turned into a full 4x4 matrix: the plaquette resolves
    # intra-cluster Cu-Cu components that the 1x1 model folds into its k dependence.
    r = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=float)  # Cu positions, physical units
    Q = [b*np.array(q, dtype=float) for q in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))]

    for a in range(dim_redc):
        for c in range(dim_redc):
            d = r[c] - r[a]
            for j, q in enumerate(Q):
                # block j of the input file holds Gamma^(1x1) at kc + Q[j]: same ordering as k above
                phase = np.exp(2j*np.pi*((kc + q) @ d))
                hybridc[:, :, a, c] += phase[None, :]*hybrid[:, j*nkc:(j+1)*nkc, 0, 0]

    hybridc /= dim_redc; # normalization (1/L factor in front of the cellularization formula)

    with h5py.File(output, "w") as f:
        f.create_dataset("eta", data = eta, dtype=float)
        f.create_dataset("w", data = wr, dtype=wr.dtype)
        f.create_dataset("weight", data = weight, dtype=weight.dtype)
        f.create_dataset("k", data = 2*kc, dtype=k.dtype)  # in the reciprocal superlattice basis of the 2x2 model
        f.create_dataset("hybrid_real", data=hybridc.real, dtype=float)
        f.create_dataset("hybrid_imag", data=hybridc.imag, dtype=float)
        f.create_dataset("mixing", data=model.mixing, dtype=int)

cellularize2x2('hybrid.h5', 'hybridc.h5')


#============================================================================
# Defining the model based on the 2x2 cluster
#
# LM2: the same 2x2 Cu plaquette as in LM0, but with the oxygens replaced by the cellularized
# hybridization. Its cluster and self-energy are identical to LM0's, so the comparison below is
# exact at any U -- it is the real test of cellularize2x2().

pyqcm.reset_model()

CM2 = pyqcm.cluster_model(4, 0, name='CM2')
clus2 = pyqcm.cluster(CM2, Cu_sites)
model = pyqcm.lattice_model('LM2', clus2, ((4,0,0),(0,4,0)), ((2,0,0),(0,2,0)), hybrid_file = 'hybridc.h5')
model.set_basis(((0.5,0,0),(0,0.5,0),(0,0,1)))
model_operators_reduced(model)

#============================================================================
# Running CDMFT on the plaquette model

# back to the coarser grid: the 2x2 reduced BZ is 1/4 of the Cu zone used for LM1
pyqcm.set_global_parameter('kgrid_side', nk_side)
pyqcm.set_wavevector_grid(nk_side,nk_side,1)

model.set_target_sectors(['N4:S0/N2:S0/N6:S0/N0:S0/N8:S0'])

model.set_parameters(band_params_Cu)
I = model.model_instance()
I.ground_state(pr=True)
print('\naverages on the 2x2 Cu (no oxygen) model:\n')
I.averages(pr=True)

# CPT_Green_function_grid() indexes directly into the frequency and wavevector arrays stored in the
# h5 file, so (iw_test, k_test_label) here denote the same (w, K) as wc[iw_test], kc[k_test_label].
Gcpt_LM2 = I.CPT_Green_function_grid(iw_test, k_test_label)

print('\nCPT Green function at w = ', wc[iw_test], ', K = ', kc[k_test_label], ':')
print('   2x2 + cellularized hybridization : ', Gcpt_LM2[0,0])
print('   4*Cu+8*O reference model         : ', Gcpt_LM0[0,0])
# expected to vanish to machine precision: same cluster, same self-energy, exact Gamma
print('   max |difference| over the 4x4 Cu block : ', np.abs(Gcpt_LM2 - Gcpt_LM0).max())


#============================================================================
# Defining the 2x2 impurity model for CDMFT
#
# From here on the plaquette carries a bath, so the comparison becomes a CDMFT one: the bath is
# fitted to the CDMFT host, which is built from the projected lattice Green function. The two
# models below must produce the same host, hence the same bath and the same averages.

ns = 4

class rep:
    def __init__(self, _C, _signs, _name):
        self.C = _C
        self.signs = _signs
        self.name = _name

irrep = dict()

Cu_sites = ((0, 0, 0), (2, 0, 0), (0, 2, 0), (2, 2, 0))
O_sites  = ((-1, 0, 0),(1, 0, 0), (0,-1, 0), (0, 1, 0), (1, 2, 0),(3, 2, 0), (2, 1, 0), (2, 3, 0))

def system_2x2_C2v(reps):
    """
    Builds a 2x2 cluster model with one bath orbital per entry of 'reps', each bath orbital
    transforming according to the named irrep of C2v.

    param str reps: concatenated irrep names, 2 characters each, e.g. 'A1A1A2A2B1B2'

    Orbital numbering used by new_operator(): 1..ns are the cluster sites (spin up), ns+1..no the
    bath orbitals (spin up), and adding no = ns+nb gives the corresponding spin-down orbital.
    """
    R = [reps[i:i+2] for i in range(0, len(reps), 2)]
    nb = len(R)
    no = ns + nb
    # The two generators of C2v, as permutations of the site labels 1..4 with
    # Cu_sites = (0,0), (2,0), (0,2), (2,2):
    #   g1 = mirror y -> -y  (1<->3, 2<->4)
    #   g2 = mirror x -> -x  (1<->2, 3<->4)
    # Their product is the C2 rotation, so the group has G = 4 elements.
    g1 = [3, 4, 1, 2]
    g2 = [2, 1, 4, 3]
    # For each irrep: C = its characters under (g1, g2), and 'signs' = the amplitude pattern over
    # the 4 sites of the corresponding cluster orbital. With bath_irrep=True the generator entry of
    # a bath orbital is not a permutation but a phase, given as an integer multiple of 2*pi/G:
    # here 0 codes +1 and 2 codes -1.
    irrep['A1'] = rep((0,0), (1, 1, 1, 1), 'A1')
    irrep['A2'] = rep((2,2), (1,-1,-1, 1), 'A2')
    irrep['B1'] = rep((2,0), (1, 1,-1,-1), 'B1')
    irrep['B2'] = rep((0,2), (1,-1, 1,-1), 'B2')
    for i in range(nb):
        g1 += [irrep[R[i]].C[0]]
        g2 += [irrep[R[i]].C[1]]
    CM = pyqcm.cluster_model(ns, nb, name='CM_'+reps, generators=(g1, g2), bath_irrep=True)
    CM.nb = nb
    varia_N = []
    varia_SC = []
    
    # bath energies: one diagonal operator per bath orbital, applied to both spins
    for i in range(nb):
        name = 'eb'+str(i+1)
        lab = i+4+1
        CM.new_operator(name, 'one-body', (
            (lab, lab, 1.0),
            (lab + no, lab + no, 1.0)
        ))
        varia_N.append(name)

    # bath hybridizations: bath orbital i couples to the 4 cluster sites with the sign pattern of
    # its irrep, so that the coupling itself is symmetry-adapted and the C2v symmetry is preserved.
    for i in range(nb):
        elem = []
        for j in range(ns):
            elem.append((j+1, i+ns+1, irrep[R[i]].signs[j]))
            elem.append((j+1+no, i+ns+no+1, irrep[R[i]].signs[j]))
        name = 'tb'+str(i+1)
        CM.new_operator(name, 'one-body', elem)
        varia_N.append(name)

    # singlet pairing between a cluster site and a bath orbital. Defined here so that
    # superconducting solutions are reachable, but left at zero below, so the mixing stays normal.
    for i in range(nb):
        elem = []
        for j in range(ns):
            elem.append((j+1, i+ns+no+1, irrep[R[i]].signs[j]))
            elem.append((i+ns+1, j+1+no, irrep[R[i]].signs[j]))
        name = 'db'+str(i+1)
        CM.new_operator(name, 'anomalous', elem)
        varia_SC.append(name)
    return CM

#============================================================================
# Defining the CDMFT model with Copper (2x2) and oxygens (8 sites)
# This is the reference model

pyqcm.reset_model()

# 6 bath orbitals: two A1, two A2, one B1 and one B2
CM = system_2x2_C2v('A1A1A2A2B1B2')

O8 = pyqcm.cluster_model(8, 0, name='O8')
O8 = pyqcm.cluster(O8, O_sites)

# construction of the lattice model
Cu = pyqcm.cluster(CM, Cu_sites)
model = pyqcm.lattice_model('Emery_2x2+O', (Cu, O8), ((4,0,0), (0,4,0)), ((2,0,0),(0,2,0)))
model_operators(model)
model.nb = 6

# Starting point of the bath. The 6th bath orbital (B2) is tied to the 5th (B1) by a dependency,
# which imposes the x <-> y symmetry of the solution and leaves 5+5 free parameters, not 6+6.
bath_params=""
bath_params += "eb1_1 = 1\n"
bath_params += "eb2_1 = -1\n"
bath_params += "eb3_1 = 1\n"
bath_params += "eb4_1 = -1\n"
bath_params += "eb5_1 = 1\n"
bath_params += "eb6_1 = 1*eb5_1\n"
bath_params += "tb1_1 = 0.5\n"
bath_params += "tb2_1 = 0.5\n"
bath_params += "tb3_1 = 0.5\n"
bath_params += "tb4_1 = 0.5\n"
bath_params += "tb5_1 = 0.5\n"
bath_params += "tb6_1 = 1*tb5_1\n"

# R0 = the trivial irrep (A1) of C2v; N10 = half filling of the 4 sites + 6 bath orbitals.
model.set_target_sectors(['R0:N10:S0','N8:S0'])
model.set_parameters(band_params_Cu+band_params_O+bath_params)

# only the independent parameters are varied: eb6/tb6 follow eb5/tb5 through their dependency
varia = [f'eb{i+1}_1' for i in range(model.nb-1)] 
varia += [f'tb{i+1}_1' for i in range(model.nb-1)] 

# NB: GROUND STATE INCONSISTENCY warnings during the first iterations are expected here. They come
# from the far-from-converged starting bath and disappear once CDMFT has converged; they are
# identical in the two models below, and are not a symptom of the external hybridization.
C = CDMFT(model, varia=varia, grid = cdmft_grid, convergence='self-energy', accur=1e-3, method='trf', iteration='fixed_point', alpha=0.2, maxiter=64)
I = C.I
I.averages(pr=False)
ave = I.averages()
print('<ed> = ', 3*ave['ed'])
print('<tc> = ', 3*ave['tc'])


#============================================================================
# Defining the CDMFT model with Copper (2x2) only, and external hybridization
#
# Same impurity (same cluster, same bath, same starting parameters, same distance function) as
# above, with the 8 oxygens replaced by hybridc.h5. The converged bath parameters and the Cu
# density must match those printed by the reference model just above.

pyqcm.reset_model()
CM = system_2x2_C2v('A1A1A2A2B1B2')
Cu = pyqcm.cluster(CM, Cu_sites)
model = pyqcm.lattice_model('Emery_2x2+O', (Cu), ((4,0,0), (0,4,0)), ((2,0,0),(0,2,0)), hybrid_file = "hybridc.h5")
model_operators_reduced(model)
model.nb = 6

# no sector needed for a second cluster this time: the oxygens are gone
model.set_target_sectors(['N10:S0'])
model.set_parameters(band_params_Cu+bath_params)

C = CDMFT(model, varia=varia, grid = cdmft_grid, convergence='self-energy', accur=1e-3, method='trf', iteration='fixed_point', alpha=0.2, maxiter=64)
I = C.I
I.averages(pr=False)
ave = I.averages()
# NB: no factor of 3 here. That factor corrects the per-orbital normalization of the models
# containing 3 orbitals per unit cell (Cu + 2 O); this model has a single orbital per unit cell.
print('<ed> = ', ave['ed'])
print('<tc> = ', ave['tc'])
