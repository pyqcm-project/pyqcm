import numpy as np
import pyqcm
#--------------------------------------------------------------------------------
# model definition
CM = pyqcm.cluster_model(4, name='2x2_C2', generators=[[4, 3, 2, 1]])
clus = pyqcm.cluster(CM, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)))
model = pyqcm.lattice_model('2x2_C2', clus, [(2, 0, 0), (0, 2, 0)])
model.interaction_operator('U')
model.hopping_operator('t', (0, 1, 0), -1)
model.hopping_operator('t', (1, 0, 0), -1)
model.density_wave('M', 'Z', (1, 1, 0))
model.set_target_sectors(['R0:N4:S0'])

#--------------------------------------------------------------------------------
model.set_parameters("""
    M=0.0
    M_1=1e-9
    U=8.0
    mu=4.0
    t=1.0
""")
#--------------------------------------------------------------------------------

I = pyqcm.model_instance(model)
X = I.susceptibility_poles('M@1')  # note the name given to the operator. The @1 is important because it is a density wave

print('list of poles (column 1) and residues (column 2) of the dynamical susceptibility on the cluster:')
print(X)
