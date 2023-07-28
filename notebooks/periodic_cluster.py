####################################################################################################
# Simple Hubbard model on a periodic cluster
# We Define a simple nearest-neighbor, half-filled Hubbard model with a periodic cluster of length 12 in dimension 1. Make use of translation symmetry. For each value of the 12 possible values of crystal momentum in this system, compute the ground state energy of the cluster for $U=0$ and $U=4$. This problem concerns the cluster only and makes no use of CPT or other quantum cluster methods.
####################################################################################################
import pyqcm
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# global option that toggles periodic boundary conditions
pyqcm.set_global_parameter("periodic") 

# defines the cluster with a translation symmetry generator 
CM = pyqcm.cluster_model(12, generators = ((12, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),)) 

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, [(i,0,0) for i in range(12)]) 

# define a lattice model 
model = pyqcm.lattice_model('periodic12', clus, ((12,0,0),))

# define a few operators in this model. Since this is a 2-orbital model, one needs to specify the relevant orbitals.
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)

model.set_parameters("""
    t=1
    U=4
    mu=0.5*U
""")

# loop over the different irreducible representations (number of electrons and wavevectors)

for n in range(1,13):
    for k in range(12):
        model.set_target_sectors([f"R{k}:N{n}:S{n%2}"]) 
        I = pyqcm.model_instance(model)
        print(f"N={n}\tk={k}\tE0={I.ground_state(file='GS.tsv')[0][0]}")
