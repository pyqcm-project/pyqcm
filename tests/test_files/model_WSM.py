import pyqcm
CM = pyqcm.cluster_model( 4, 0)
clus = pyqcm.cluster(CM, ((0, 0, 0), ( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0)))
model = pyqcm.lattice_model('WSM_2x2', clus, (( 2, 0, 0), ( 0, 2, 0), (0, 0, 1)))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), 1, tau=1, sigma=1)
model.hopping_operator('t', (0, 1, 0), 1, tau=2, sigma=2)
model.hopping_operator('t', (0, 0, 1), 1, tau=2, sigma=3)
model.hopping_operator('t', (0, 0, 0), -0.765367, tau=0, sigma=1)
model.hopping_operator('m', (0, 1, 0), -0.5, tau=1, sigma=1)
model.hopping_operator('m', (0, 0, 1), -0.5, tau=1, sigma=1)
model.hopping_operator('m', (0, 0, 0), 2, tau=0, sigma=1)

model.set_target_sectors(['R0:N4'])
model.set_parameters("""
U = 4
mu= 0.5*U
t = 1
m = 1.5
""")

