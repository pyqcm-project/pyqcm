import pyqcm

CM = pyqcm.cluster_model(4, generators=((4,3,2,1),))
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (2,0,0), (3,0,0)))
model = pyqcm.lattice_model('1D_L4', clus, ((4, 0, 0),))

model.interaction_operator('U')
model.interaction_operator('J', link=( 1, 0, 0), type='Hund')
model.hopping_operator('t', ( 1, 0, 0), -1)  # NN hopping
model.hopping_operator('tp', ( 2, 0, 0), -1)  # NNN hopping
model.hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)  # field in the x direction
model.hopping_operator('h', [0, 0, 0], 1, tau=0, sigma=3)  # field in the x direction
model.anomalous_operator('D', ( 1, 0, 0), 1)  # NN singlet
model.anomalous_operator('Di', ( 1, 0, 0), 1j)  # NN singlet with imaginary amplitude
model.anomalous_operator('S', [0, 0, 0], 1)  # on-site singlet
model.anomalous_operator('Si', [0, 0, 0], 1j)  # on-site singlet with imaginary amplitude
model.density_wave('H', 'Z', [0, 0, 0])
model.density_wave('Hx', 'X', [0, 0, 0])
