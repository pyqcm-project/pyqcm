import pyqcm

CM = pyqcm.cluster_model(4)
clus = pyqcm.cluster(CM,((0,0,0),(1,0,0),(2,0,0),(3,0,0)))
model = pyqcm.lattice_model('1D_L4', clus, ((4,0,0),))

model.current_dir='x'
model.interaction_operator('U')
model.interaction_operator('J', link=(1,0,0), type='Hund')
model.hopping_operator('t', (1,0,0), -1)  # NN hopping
model.hopping_operator('tsi', (1,0,0), -1, sigma=3, tau=2)  # NN hopping
model.hopping_operator('ti', (1,0,0), -1, tau=2)  # NN hopping with imaginary amplitude
model.hopping_operator('tp', (2,0,0), -1)  # NNN hopping
model.hopping_operator('hx', (0,0,0), 1, tau=0, sigma=1)  # field in the x direction
model.hopping_operator('h', (0,0,0), 1, tau=0, sigma=3)  # field in the x direction
model.anomalous_operator('D', (1,0,0), 1)  # NN singlet
model.anomalous_operator('Di', (1,0,0), 1j)  # NN singlet with imaginary amplitude
model.anomalous_operator('S', (0,0,0), 1)  # on-site singlet
model.anomalous_operator('Si', (0,0,0), 1j)  # on-site singlet with imaginary amplitude
model.anomalous_operator('Pz', (1,0,0), 1, type='dz')  # NN triplet
model.anomalous_operator('Py', (1,0,0), 1, type='dy')  # NN triplet
model.anomalous_operator('Px', (1,0,0), 1, type='dx')  # NN triplet
model.density_wave('M', 'Z', (1,0,0))
model.density_wave('H', 'Z', (0,0,0))
model.density_wave('Hx', 'X', (0,0,0))
