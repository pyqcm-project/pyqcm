# Test file
# Goal : to test the per-system CDMFT distance (CDMFT(..., by_system=True)).
#   On a model with one system per cluster, the per-system and per-cluster
#   minimizations must converge to identical variational parameters.
# (the genuine multi-system case is covered by test_cdmft_by_system_replicas.py)
# --------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT, frequency_grid

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('G2', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))
model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))
model.set_target_sectors(['R0:N6:S0'])

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']
init = dict(U=4, mu=2.0, t=1, tb1_1=0.5, tb2_1=0.4, eb1_1=1.0, eb2_1=-1.1)
model.set_parameters('\n'.join('{:s}={:s}'.format(k, str(v)) for k, v in init.items()))

grid = frequency_grid('legendre', (1, 10, 10, 10, 10))
res = {}
for bs in (False, True):
    for k, v in init.items():
        model.set_parameter(k, float(v))
    X = CDMFT(model, varia=varia, grid=grid, method='Nelder-Mead', accur=5e-6,
              maxiter=32, miniter=1, depth=1, iteration='broyden', by_system=bs)
    P = model.parameters()
    res[bs] = np.array([P[v] for v in varia])
    print('by_system={!s:<5} -> '.format(bs)
          + ' '.join('{:s}={:.6f}'.format(v, P[v]) for v in varia))

diff = np.max(np.abs(res[False] - res[True]))
print('\nmax |per-cluster - per-system| = {:.3e}'.format(diff))
assert diff < 1e-6, 'per-system and per-cluster results differ for a single-system cluster'

print('\ntest_cdmft_by_system : PASSED')
