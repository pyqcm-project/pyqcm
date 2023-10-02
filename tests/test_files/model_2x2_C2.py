import pyqcm

CM = pyqcm.cluster_model(4, generators=[[4, 3, 2, 1]])
clus = pyqcm.cluster(CM, ((0, 0, 0), ( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0)))
model = pyqcm.lattice_model('2x2_C2', clus, [( 2, 0, 0), ( 0, 2, 0)])
model.interaction_operator('U')
model.interaction_operator('V', link=( 1, 0, 0), amplitude=1)
model.interaction_operator('V', link=( 0, 1, 0), amplitude=1)
model.hopping_operator('Vm', [0, 0, 0], 1, tau=0)  # NN hopping
V_eig = 0.5
model.interaction_operator('J', link=( 1, 0, 0), type='Hund')

model.hopping_operator('t', ( 0, 1, 0), -1)
model.hopping_operator('t', ( 1, 0, 0), -1)
model.hopping_operator('t2', ( 1, 1, 0), -1)
model.hopping_operator('t2', (-1, 1, 0), -1)
model.hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
model.density_wave('M', 'Z', ( 1, 1, 0))
model.hopping_operator('t3', ( 2, 0, 0), -1)
model.hopping_operator('t3', ( 0, 2, 0), -1)
model.anomalous_operator('D', ( 1, 0, 0), 1)
model.anomalous_operator('D', ( 0, 1, 0), -1)


################################################
# model.set_target_sectors(['R0:N4:S0'])
model.set_target_sectors(['R0'])
model.set_parameters("""
t = 1
t2 = 1e-9
U = 8
mu = 4
M = 0
M_1 = 0.15
t_1 = 1
D = 0.001
hx = 0.001
""")
                     
