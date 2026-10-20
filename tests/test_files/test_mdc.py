# Test file
# Goal : to test the different options for mdc()
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

pyqcm.set_global_parameter('GF_method', 'L')

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors('S0')

# Simulation parameters
model.set_parameters("""
    U=8
    mu=1
    t=1
    t2=0.3
	D = 0.1
""")

I = pyqcm.model_instance(model)  
I.mdc(file = 'mdc_A.pdf')
I.mdc(file = 'mdc_phase.pdf', opt='anom_phase', orb=(1,1))
I.mdc(file = 'mdc_self.pdf', opt='self')
I.mdc(file = 'mdc_Z.pdf', opt='Z')
