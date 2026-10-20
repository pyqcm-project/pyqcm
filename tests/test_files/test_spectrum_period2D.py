# Test file
# Goal : to test the different values GF_method with periodization G and L, for mdc's
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm

#--------------------------------------------------------------------------------
CM = pyqcm.cluster_model(4, generators=[[4, 3, 2, 1]])
clus = pyqcm.cluster(CM, ((0, 0, 0), ( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0)))
model = pyqcm.lattice_model('2x2_C2', clus, [( 2, 0, 0), ( 0, 2, 0)])
model.interaction_operator('U')

model.hopping_operator('t', ( 0, 1, 0), -1)
model.hopping_operator('t', ( 1, 0, 0), -1)
model.hopping_operator('t2', ( 1, 1, 0), -1)
model.hopping_operator('t2', (-1, 1, 0), -1)
model.hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
model.density_wave('M', 'Z', ( 1, 1, 0))
model.anomalous_operator('D', ( 1, 0, 0), 1)
model.anomalous_operator('D', ( 0, 1, 0), -1)
#--------------------------------------------------------------------------------


L = 4

pyqcm.set_global_parameter('GF_method', 'L')
# pyqcm.set_global_parameter('block_Lanczos_QR')

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors('N{:d}:S0'.format(L))

# Simulation parameters
model.set_parameters("""
    U=8
    mu=1
    t=1
    t2=0.3
""")

I = pyqcm.model_instance(model)  
pyqcm.set_global_parameter('periodization', 'G')
I.spectral_function(period = 'G', w=6, nk=32, file = 'spectrum2D_G.pdf')
I.mdc(file = 'mdc_G.pdf')

pyqcm.set_global_parameter('combine_mcf')
I = pyqcm.model_instance(model)  
pyqcm.set_global_parameter('periodization', 'L')
I.spectral_function(period = 'L', w=6, nk=32, file = 'spectrum2D_L.pdf')
I.mdc(file = 'mdc_L.pdf')
