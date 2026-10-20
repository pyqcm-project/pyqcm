# Test file
# Goal : to test the explicit printing of the Hamiltonian and its basis
#--------------------------------------------------------------------------------

import pyqcm
import numpy as np

#--------------------------------------------------------------------------------
# First, defining the model
clus = pyqcm.cluster_model(2, name='clus', generators=([2,1],))
clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('2x1', clus0, ((2, 0, 0),))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), 1)
model.set_target_sectors(['R1:N2:S0'])


model.set_parameters("""
    U=8.0
    mu=0.5*U
    t=1.0
""")

pyqcm.set_global_parameter('print_Hamiltonian')
I = pyqcm.model_instance(model)
I.ground_state(pr=True)

