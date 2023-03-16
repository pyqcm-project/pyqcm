import pyqcm
from pyqcm.cdmft import CDMFT

import model_graphene_bath as M

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
M.model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
M.model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    t_1=1e-9
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
""")

# Defining a function that will run a cdmft procedure within controlled_loop()
def run_cdmft():
    X = CDMFT(M.model, varia=["tb1_1", "eb1_1"], wc=10, grid_type='self') # setting the bath operators as the variationnal 
    return X.I

# Looping over values of U
M.model.controlled_loop(
    task=run_cdmft, 
    varia=["tb1_1", "eb1_1"], 
    loop_param="U", 
    loop_range=(2, 4.1, 0.5)
)