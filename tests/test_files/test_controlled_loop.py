import pyqcm
import numpy as np
from pyqcm.cdmft import CDMFT

from model_1D_2_4b import model

pyqcm.set_global_parameter('accur_OP', 1e-5)
# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
model.set_parameters("""
    U=4
    mu=2
    t=1
    t_1=1e-9
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
""")

# Defining a function that will run a cdmft procedure within controlled_loop()
def run_cdmft():
    X = CDMFT(model, varia=["tb1_1", "eb1_1"], accur=1e-5) 
    return X.I

def control(I):
    a = I.averages()
    if np.abs(a['mu']-1.0) > 0.01:
        return False
    else:
        return True

# Looping over values of U
model.controlled_loop(
    task=run_cdmft, 
    varia=["tb1_1", "eb1_1"], 
    loop_param="mu", 
    loop_range=(2, 1e-9, -0.02),
    control_func = control,
    predict = True,
    retry = 'skip'
)
