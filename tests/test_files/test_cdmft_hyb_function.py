# Test file
# Goal : CDMFT with a user-supplied hybridization function (hyb_function). When that function returns the
# cluster's own hybridization, the bath fit done in Python must land on the same solution as the built-in C++ fit.
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT, frequency_grid

#--------------------------------------------------------------------------------
# the 2-site graphene cluster with 4 bath orbitals of test_cdmft_methods.py

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('Graphene_2', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))
model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))
model.set_target_sectors(['R0:N6:S0'])

START = """
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
"""
varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']
grid = frequency_grid('legendre', (1, 10, 10, 10, 10))

def own_hybridization(model, clus):
    """The cluster's hybridization on the CDMFT grid, as pyqcm computes it: what the built-in fit uses."""
    I = pyqcm.model_instance(model)
    return np.array([I.hybridization_function(1j * w, clus) for w in grid.wr])

model.set_parameters(START)
x0 = {'tb1_1': 0.5, 'tb2_1': 0.4, 'eb1_1': 1.0, 'eb2_1': -1.1}

def solve(**kw):
    for v in varia:
        model.set_parameter(v, x0[v])  # same starting bath for both fits (set_parameters works only once)
    X = CDMFT(model, varia=varia, grid=grid, method='trf', accur=1e-4, convergence='self-energy', iteration='fixed_point', **kw)
    return np.array([model.parameters()[v] for v in varia])

built_in = solve()
python_side = solve(hyb_function=own_hybridization)
print('built-in fit   :', built_in)
print('hyb_function   :', python_side)
assert np.allclose(built_in, python_side, atol=1e-3), f'the two fits differ: {python_side - built_in}'
print('hyb_function reproduces the built-in fit')
