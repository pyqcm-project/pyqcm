import pyqcm
import numpy as np

# Constructing a new cluster model
ns = 2 # number of physical sites
nb = 4 # number of bath sites
no = ns+nb # total number of sites
CM = pyqcm.cluster_model( ns, nb)

########################### BATH OPERATORS ########################### 
# Defining the bath hopping operators
CM.new_operator('tb1', 'one-body', [
    (1, 3, -1.0),
    (2, 4, -1.0),
    (1+no, 3+no, -1.0),
    (2+no, 4+no, -1.0)
]) # note that the last two entries pertain to the SPIN DOWN part of the operator

CM.new_operator('tb2', 'one-body', [
    (1, 5, -1.0),
    (2, 6, -1.0),
    (1+no, 5+no, -1.0),
    (2+no, 6+no, -1.0)
])

# Defining the 'orbital energy' of the baths
CM.new_operator('eb1', 'one-body', [
    (3, 3, 1.0),
    (4, 4, 1.0),
    (3+no, 3+no, 1.0),
    (4+no, 4+no, 1.0)
])

CM.new_operator('eb2', 'one-body', [
    (5, 5, 1.0),
    (6, 6, 1.0),
    (5+no, 5+no, 1.0),
    (6+no, 6+no, 1.0)
])

######################################################################
clus = pyqcm.cluster(CM, ((0, 0, 0), (1, 0, 0)))
model = pyqcm.lattice_model('Graphene_2', clus, ((1,-1, 0), (2, 1, 0)), ((1,-1, 0), (2, 1, 0)))
model.set_basis([( 1, 0, 0),[-0.5,np.sqrt(3)/2,0]]) # Classic Graphene basis (for simplicity and graphical purposes)

# Defining the interaction operator on BOTH bands
model.interaction_operator('U', orbitals=(1,1))
model.interaction_operator('U', orbitals=(2,2))

# Defining NN hopping terms
model.hopping_operator('t', ( 1, 0, 0), -1, orbitals=(1,2)) # All hops here are from one band to another
model.hopping_operator('t', ( 0, 1, 0), -1, orbitals=(1,2))
model.hopping_operator('t', [-1,-1,0], -1, orbitals=(1,2))

