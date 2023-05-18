import pyqcm
from pyqcm.cdmft import *

ns = 4
nb = 4
no = ns + nb
CM = pyqcm.cluster_model( ns, nb)  # call to constructor of class "cluster_model"

CM.new_operator('eb1', 'one-body', ((ns+1,ns+1,1), (ns+2,ns+2,1), (ns+no+1,ns+no+1,1), (ns+no+2,ns+no+2,1)))
CM.new_operator('eb2', 'one-body', ((ns+3,ns+3,1), (ns+4,ns+4,1), (ns+no+3,ns+no+3,1), (ns+no+4,ns+no+4,1)))
CM.new_operator('tb1', 'one-body', ((1,ns+1,1), (1+no,ns+no+1,1), (4,ns+2,1), (4+no,ns+no+2,1)))
CM.new_operator('tb2', 'one-body', ((1,ns+3,1), (1+no,ns+no+3,1), (4,ns+4,1), (4+no,ns+no+4,1)))

clus = pyqcm.cluster(CM, [(i,0,0) for i in range(ns)]) # call to constructor of class "cluster"

model = pyqcm.lattice_model('1D_4_4b', clus, ((ns,0,0),)) # call to constructor of class "lattice_model"

model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)

############# model definition end #################

model.set_target_sectors(['R0:N8:S0'])
model.set_parameters("""
U = 4
mu = 2
t = 1
eb1_1 = 1
eb2_1 = -1
tb1_1 = 0.5
tb2_1 = 0.5
""")

I = pyqcm.model_instance(model) # call to constructor of class "model_instance"
I.spectral_function(path='line', file='test_reset1.pdf')

cdmft = CDMFT(model, ('eb1_1', 'eb2_1', 'tb1_1', 'tb2_1'))

############# new model definition #################
pyqcm.reset_model()

ns = 2
nb = 4
no = ns + nb
CM = pyqcm.cluster_model( ns, nb)  # call to constructor of class "cluster_model"

CM.new_operator('eb1', 'one-body', ((ns+1,ns+1,1), (ns+2,ns+2,1), (ns+no+1,ns+no+1,1), (ns+no+2,ns+no+2,1)))
CM.new_operator('eb2', 'one-body', ((ns+3,ns+3,1), (ns+4,ns+4,1), (ns+no+3,ns+no+3,1), (ns+no+4,ns+no+4,1)))
CM.new_operator('tb1', 'one-body', ((1,ns+1,1), (1+no,ns+no+1,1), (2,ns+2,1), (2+no,ns+no+2,1)))
CM.new_operator('tb2', 'one-body', ((1,ns+3,1), (1+no,ns+no+3,1), (2,ns+4,1), (2+no,ns+no+4,1)))

clus = pyqcm.cluster(CM, [(i,0,0) for i in range(ns)]) # call to constructor of class "cluster"

model = pyqcm.lattice_model('1D_2_4b', clus, ((ns,0,0),)) # call to constructor of class "lattice_model"

model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)

model.set_target_sectors(['R0:N6:S0'])
model.set_parameters("""
U = 4
mu = 2
t = 1
eb1_1 = 1
eb2_1 = -1
tb1_1 = 0.5
tb2_1 = 0.5
""")

I = pyqcm.model_instance(model) # call to constructor of class "model_instance"
I.spectral_function(path='line', file='test_reset2.pdf')

cdmft = CDMFT(model, ('eb1_1', 'eb2_1', 'tb1_1', 'tb2_1'))
