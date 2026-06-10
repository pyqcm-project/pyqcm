# Test file
# Goal : to test the per-system CDMFT distance on a cluster that contains two
#   equivalent replica systems. The per-system minimization (by_system=True) must
#   give the two replicas identical bath parameters, whereas the per-cluster
#   minimization (which averages the systems into one host) need not.
# --------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT, frequency_grid

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])
clus2 = pyqcm.cluster_model(2, name='clus2')

# clus0 owns two replica systems of model 'clus'; clus1 is a bath-less leg
clus0 = pyqcm.cluster([clus, clus], ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
clus1 = pyqcm.cluster(clus2, ((0, 0, 0), (1, 0, 0)), pos=(0, 1, 0))
model = pyqcm.lattice_model('1D_2_4b_2sb', [clus0, clus1], ((2, 0, 0),))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('ty', (0, 1, 0), -1)
model.set_target_sectors(['N6:S0', 'N6:S0', 'N2:S0'])

assert model.nsys == 3 and model.nclus == 2, 'unexpected system/cluster count'
assert model.sys_clus == [0, 0, 1], 'unexpected system-to-cluster mapping'

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1', 'eb1_2', 'eb2_2', 'tb1_2', 'tb2_2']
init = dict(U=4, mu=2.0, t=1, ty=0.5,
            tb1_1=0.5, tb2_1=0.4, eb1_1=1.0, eb2_1=-1.1,
            tb1_2=0.5, tb2_2=0.4, eb1_2=1.0, eb2_2=-1.1)
model.set_parameters('\n'.join('{:s}={:s}'.format(k, str(v)) for k, v in init.items()))

grid = frequency_grid('legendre', (0.5, 5, 5, 5, 5))
X = CDMFT(model, varia=varia, grid=grid, accur=1e-3, convergence='self-energy',
          miniter=1, maxiter=64, depth=1, iteration='fixed_point', by_system=True)

P = model.parameters()
sys1 = np.array([P['eb1_1'], P['eb2_1'], P['tb1_1'], P['tb2_1']])
sys2 = np.array([P['eb1_2'], P['eb2_2'], P['tb1_2'], P['tb2_2']])
rep_diff = np.max(np.abs(sys1 - sys2))
print('\nreplica 1 bath:', sys1)
print('replica 2 bath:', sys2)
print('max |replica 1 - replica 2| (per-system) = {:.3e}'.format(rep_diff))
assert rep_diff < 1e-3, 'equivalent replica systems should converge to identical baths'

print('\ntest_cdmft_by_system_replicas : PASSED')
