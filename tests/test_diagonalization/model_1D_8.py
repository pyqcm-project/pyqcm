from pyqcm import *

L = 8
new_cluster_model('clus', L, 0)
pos = []
for i in range(L):
    pos.append((i,0,0))


add_cluster('clus', [0, 0, 0], pos)
lattice_model('1D_{:d}'.format(L), [[L, 0, 0]])
interaction_operator('U')
interaction_operator('V', link=( 1, 0, 0))
hopping_operator('t', ( 1, 0, 0), -1)  # NN hopping
hopping_operator('ti', ( 1, 0, 0), -1, tau=2)  # NN hopping
hopping_operator('tp', ( 2, 0, 0), -1)  # NNN hopping
hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)  # field in the x direction
anomalous_operator('D', ( 1, 0, 0), 1)  # NN singlet
