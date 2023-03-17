import pyqcm
import numpy as np

# A simple 1D 4 site cluster
CM = pyqcm.cluster_model(4)
clus = pyqcm.cluster(CM, ((0, 0, 0),(1, 0, 0),(2, 0, 0),(3, 0, 0)))
model = pyqcm.lattice_model('1D_4', clus, ((4,0,0),))

model.interaction_operator('U', amplitude=1) # on site interaction
model.interaction_operator('V', link=( 1, 0, 0), amplitude=1) # NN interaction

# Defining an explicit operator for the "Hartree part" of the extended interaction
elems = [(( 0, 0, 0), ( 0, 0, 0), 1/np.sqrt(2)), ([3,0,0], ( 0, 0, 0), 1/np.sqrt(2))] # the 1/sqrt(2) factor ensures normalization
model.explicit_operator('Vm', elems, type='one-body', tau=0) # Vm is an on-site operator ---> tau=0
model.hopping_operator('t', ( 1, 0, 0), -1) 
model.density_wave('Delta', 'N', ( 1, 0, 0), amplitude=1) # A charge density wave with period 2 in position space
