# Test file
# Goal : to test replicas
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT

L = 4
N = 4
CM = pyqcm.cluster_model(L, n_bath=0)
pos = [(i,0,0) for i in range(L)]

clus = [pyqcm.cluster(CM, pos)]
for c in range(1,N):
    clus.append(pyqcm.cluster(clus[0], pos, pos=(L*c,0,0)))

model = pyqcm.lattice_model('1D_{:d}x{:d}'.format(L,N), clus, ((L*N, 0, 0),))

model.interaction_operator('U')
model.hopping_operator('t', ( 1, 0, 0), -1)
model.set_target_sectors(['N4:S0'])
model.set_parameters("""
t = 1
mu = 2
U = 4
""")

I = pyqcm.model_instance(model)

I.cluster_spectral_function(w=6,clus=0, file='replica0.pdf')
I.cluster_spectral_function(w=6,clus=1, file='replica1.pdf')
I.cluster_spectral_function(w=6,clus=2, file='replica2.pdf')
I.cluster_spectral_function(w=6,clus=3, file='replica3.pdf')
