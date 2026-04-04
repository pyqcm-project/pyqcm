# Test file
# Goal : to test CDMFT in a 2-cluster model
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT

#--------------------------------------------------------------------------------
# First, defining the model (a 2-cluster model for graphene with bath)

clus = pyqcm.cluster_model(4, n_bath=6, name='clus')
clus.new_operator('eb1', 'one-body', [(5, 5, 1), (15, 15, 1), (7, 7, 1), (17, 17, 1), (9, 9, 1), (19, 19, 1)])
clus.new_operator('eb2', 'one-body', [(6, 6, 1), (16, 16, 1), (8, 8, 1), (18, 18, 1), (10, 10, 1), (20, 20, 1)])
clus.new_operator('tb1', 'one-body', [(1, 5, 1), (11, 15, 1), (2, 7, 1), (12, 17, 1), (3, 9, 1), (13, 19, 1)])
clus.new_operator('tb2', 'one-body', [(1, 6, 1), (11, 16, 1), (2, 8, 1), (12, 18, 1), (3, 10, 1), (13, 20, 1)])

clus0 = pyqcm.cluster(clus, ((-1, -1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)), pos=(1, 0, 0))
clus1 = pyqcm.cluster(clus, ((1, 1, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 0)), pos=(-1, 0, 0))

model = pyqcm.lattice_model('graphene_4_2C', [clus0, clus1], ((4, 2, 0), (2, -2, 0)), lattice=((1, -1, 0), (2, 1, 0)))
model.set_basis([( 1, 0, 0),[-0.5,np.sqrt(3)/2,0]])

model.interaction_operator('U')
model.hopping_operator('t', (-1, 0, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (0, -1, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (1, 1, 0), 1, orbitals=(1, 2))
#--------------------------------------------------------------------------------

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N10:S0/R0:N8:S0/R0:N12:S0']*2)

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=0.5
    eb1_1=1.0
    eb2_1=-1.0
    tb1_2=0.5
    tb2_2=0.5
    eb1_2=1.0
    eb2_2=-1.0
""")

I = pyqcm.model_instance(model)  

convergence=['self-energy', 'distance']; accur=[1e-4, 1e-5]

varia = ['tb1_1', 'eb1_1', 'tb2_1', 'eb2_1', 'tb1_2', 'eb1_2', 'tb2_2', 'eb2_2']

sol = CDMFT(model, varia=varia, grid_type='legendre', wc=(1,10,10), accur=accur, convergence=convergence, method='PRAXIS', maxiter=64, depth=1, iteration='fixed_point')