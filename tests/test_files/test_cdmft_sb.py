# Test file
# Goal : to test CDMFT with two subbaths on a ladder model, one leg with two bath systems, the other one without any.
# Then the solution is written on a file, and read again, and the GF is printed both times to compare
#--------------------------------------------------------------------------------
import pyqcm
from pyqcm.cdmft import CDMFT
import numpy as np

#--------------------------------------------------------------------------------
# First, defining the model (a 2-cluster model for graphene with bath)

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus2 = pyqcm.cluster_model(2, name='clus2')

clus0 = pyqcm.cluster([clus, clus], ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
clus1 = pyqcm.cluster(clus2, ((0, 0, 0), (1, 0, 0)), pos=(0, 1, 0))

model = pyqcm.lattice_model('1D_2_4b_2sb', [clus0, clus1], ((2, 0, 0),))

model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('ty', (1, 0, 0), -1)
#--------------------------------------------------------------------------------


# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['N6:S0', 'N6:S0', 'N2:S0'])

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    ty = 0.5
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
    tb1_2=0.5
    tb2_2=0.4
    eb1_2=1.0
    eb2_2=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1', 'eb1_2', 'eb2_2', 'tb1_2', 'tb2_2']

X = CDMFT(model, varia=varia, wc=[0.5,5,5], grid_type='legendre', accur=1e-3, convergence='self-energy', miniter=1, maxiter=64, depth=1, iteration='fixed_point')


X.I.write_all_hdf5('test.h5')
print('GF written:\n', X.I.cluster_Green_function(0.1j, 1))
I = pyqcm.model_instance(model)
I.read_all_hdf5('test.h5')
print('\nGF read:\n', I.cluster_Green_function(0.1j, 1))

