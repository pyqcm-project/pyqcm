import pyqcm
import pyqcm.vca as vca
import numpy as np


pyqcm.set_global_parameter("verb_integrals")

# declare a cluster model of 9 sites, named 'clus'
CM = pyqcm.cluster_model(9)

sites = []
for j in range(3):
    for i in range(3):
        sites.append((i,j,0))

# defining a hopping operator that only hops around the perimeter of the clusters
ns = 9
CM.new_operator('tperim', 'one-body',[
    (1, 2, -1.0), 
    (2, 3, -1.0),
    (3, 6, -1.0), 
    (6, 9, -1.0),
    (8, 9, -1.0), 
    (7, 8, -1.0),
    (4, 7, -1.0), 
    (1, 4, -1.0),
    (1+ns, 2+ns, -1.0), 
    (2+ns, 3+ns, -1.0),
    (3+ns, 6+ns, -1.0), 
    (6+ns, 9+ns, -1.0),
    (8+ns, 9+ns, -1.0), 
    (7+ns, 8+ns, -1.0),
    (4+ns, 7+ns, -1.0), 
    (1+ns, 4+ns, -1.0)
])

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus1 = pyqcm.cluster(CM, sites) 
clus2 = pyqcm.cluster(CM, sites, (3,0,0)) # second cluster, offset from the first

# define a lattice model named '1D_4' made of the cluster(s) clus and superlattice vector (4,0,0)
model = pyqcm.lattice_model('model_3x3_2C', (clus1, clus2), ((6,0,0),(1,3,0)))

# define a few operators in this model
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)
model.hopping_operator('t', (0,1,0), -1)
model.density_wave('M', 'Z', (1,1,0)) # Spin density wave at Q = (pi,pi)

model.set_target_sectors(('R0:N9:S-1', 'R0:N9:S1')) # half filling for both clusters
model.set_parameters("""
t = 1
U = 8
mu = 0.5*U
M = 0
M_1 = 0.1
M_2 = 1*M_1
t_1 = 1
t_2 = 1*t_1
tperim_1 = 1e-9
tperim_2 = 1*tperim_1
""")

V = vca.VCA(model, varia=('M_1', 't_1', 'tperim_1'), start=(0.17, 1.15, 0.04), steps=(0.005, 0.005, 0.005), accur=(2e-3, 2e-3, 2e-3), max=(10, 10, 10))

