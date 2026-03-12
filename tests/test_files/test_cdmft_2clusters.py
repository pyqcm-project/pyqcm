import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT

from model_G4_6b_2C import model

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N10:S0']*2)

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    t_1 = 1.1
    t_2 = 0.9
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
    tb1_2=0.5
    tb2_2=1*tb1_1
    eb1_2=1.0
    eb2_2=-1.0*eb1_1
""")

I = pyqcm.model_instance(model)  

convergence=['self-energy', 'distance']; accur=[1e-4, 1e-5]

varia = ['tb1_1', 'eb1_1', 'tb1_2', 'eb1_2']

sol = CDMFT(model, varia=varia, wc=10, grid_type='self', accur=accur, convergence=convergence, method='Nelder-Mead', maxiter=64, depth=1, iteration='fixed_point')