import pyqcm
from pyqcm.cdmft import CDMFT

from model_graphene_bath import model

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    t_1=1e-9
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
""")

I = pyqcm.model_instance(model)                
A = I.cluster_hopping_matrix()
print(A)
