import cluster_2x2_2C_8b_C2v_L
from pyqcm import *

#-------------------------------------------------------------------
# construction of the lattice model 
add_cluster('clus', [0,0,0], [[0,0,0], [2,0,0], [0,2,0], [2,2,0]])
add_cluster('clus_uncorrelated',[0,0,0],[[1,0,0],[3,0,0],[0,1,0],[2,1,0],[1,2,0],[3,2,0],[0,3,0],[2,3,0]])

lattice_model('2x2_2C_8b_C2v_L', [[4,0,0],[0,4,0]],[[2,0,0],[0,2,0]])
set_basis([[0.5,0,0],[0,0.5,0],[0,0,0.5]])

interaction_operator('U',orb1=1,orb2=1)
interaction_operator('Ud',orb1=2,orb2=2)
interaction_operator('Ud',orb1=3,orb2=3)

hopping_operator('tc', [2,0,0], -1, orbitals=(1,1)) 
hopping_operator('tc', [0,2,0], -1, orbitals=(1,1)) 

hopping_operator('tpp', [1,1,0], -1, orbitals=(2,3))
hopping_operator('tpp', [1,1,0], -1, orbitals=(3,2))
hopping_operator('tpp', [-1,1,0], 1, orbitals=(2,3))
hopping_operator('tpp', [-1,1,0], 1, orbitals=(3,2))
hopping_operator('tppp', [2,0,0], 1, orbitals=(2,2))
hopping_operator('tppp', [0,2,0], 1, orbitals=(3,3))

hopping_operator('e',[0,0,0],1,orb1=2,orb2=2,tau=0,sigma=0)
hopping_operator('e',[0,0,0],1,orb1=3,orb2=3,tau=0,sigma=0)

hopping_operator('tpd', [1,0,0], 1, orbitals=(1,2))
hopping_operator('tpd', [1,0,0], -1, orbitals=(2,1))
hopping_operator('tpd', [0,1,0], 1, orbitals=(1,3))
hopping_operator('tpd', [0,1,0], -1, orbitals=(3,1))

anomalous_operator('D', [2,0,0], 1, orbitals=(1,1)) # NN singlet
anomalous_operator('D', [0,2,0], -1, orbitals=(1,1)) # NN singlet

hopping_operator('hx',[0,0,0],1,orb1=1,orb2=1,tau=0,sigma=1)











