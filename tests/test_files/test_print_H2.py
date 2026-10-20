# Test file
# Goal : to test the explicit printing of the Hamiltonian and its basis
#--------------------------------------------------------------------------------

import pyqcm
import numpy as np

#--------------------------------------------------------------------------------
# First, defining the model
clus = pyqcm.cluster_model(4, name='clus', generators=((3,4,1,2),(2,1,4,3)))
clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('2x2', clus0, ((2, 0, 0), (0, 2, 0)))
model.interaction_operator('U')
model.hopping_operator('tx', (1, 0, 0), 1)
model.hopping_operator('ty', (0, 1, 0), 1)
model.set_target_sectors(['R3:N4:S0'])


model.set_parameters("""
    U=8.0
    mu=0.5*U
    tx=1.1
	ty=0.9
""")

pyqcm.set_global_parameter('print_Hamiltonian')
I = pyqcm.model_instance(model)
I.ground_state(pr=True)

