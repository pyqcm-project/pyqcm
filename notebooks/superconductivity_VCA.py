####################################################################################################
# Simple Hubbard model in d=2 with superconductivity on a cluster of 4 sites
####################################################################################################
import pyqcm
import pyqcm.vca as vca
import numpy as np

# declare a cluster model of 4 sites, named 'clus'
CM = pyqcm.cluster_model( 4)

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (0,1,0),(1,1,0))) 

# define a lattice model named '2x2' made of the cluster(s) clus and superlattice vectors (2,0,0) & (0,2,0)
model = pyqcm.lattice_model('2x2', clus, ((2,0,0), (0,2,0)))

# define a few operators in this model
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)
model.hopping_operator('t', (0,1,0), -1)
model.hopping_operator('tp', (1,1,0), -1)
model.hopping_operator('tp', (-1,1,0), -1)

# On-site s-wave
model.anomalous_operator('S', [0,0,0], 1)

# Extended s-wave
model.anomalous_operator('xS', [1,0,0], 1)
model.anomalous_operator('xS', [0,1,0], 1)

# d_{x^2-y^2}-wave
model.anomalous_operator('D', [1,0,0], 1)
model.anomalous_operator('D', [0,1,0], -1)

# d_{xy}-wave, similar as previous wave but defined with "diagonal" links
model.anomalous_operator('Dxy', [1,1,0], 1)
model.anomalous_operator('Dxy', [-1,1,0], -1)

# Defining the appropriate sector of Hilbert space
model.set_target_sectors('R0:S0')

# The first three parameters are used for the graph of the Potthoff functionnal
model.set_parameters("""
t=1
U=8
mu=1.2
S_1=1e-9
xS_1=1e-9
D_1=1e-9
Dxy_1=1e-9
""")

# Generating a grid for the Potthoff functionnals
grid_size = 25
Delta_grid = np.linspace(1e-9, 0.3, grid_size)
pyqcm.set_global_parameter('accur_SEF', 1e-5)
# The different types of superconductivity are explored by plotting the Potthoff functionnal for each variety
vca.plot_sef(model, 'S_1', Delta_grid)
model.set_parameter("S_1", 1e-9)

vca.plot_sef(model, 'xS_1', Delta_grid)
model.set_parameter('xS_1', 1e-9)

vca.plot_sef(model, 'D_1', Delta_grid)
model.set_parameter('D_1', 1e-9)

vca.plot_sef(model, 'Dxy_1', Delta_grid)
model.set_parameter('Dxy_1', 1e-9)

