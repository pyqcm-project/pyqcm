import pyqcm
from pyqcm.cdmft import CDMFT

import model_graphene_bath as M

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
M.model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
M.model.set_parameters("""
    U=4
    mu=2
    t=1
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']

# convergence = 'GS energy'; accur = 1e-3
convergence='self-energy'; accur = 1e-4
# convergence='t'; accur = 1e-4
# convergence='parameters'; accur = 1e-4
# convergence=('parameters', 'hybridization', 'self-energy'); accur = (5e-4, 1e-4, 1e-4)
# convergence=('mu', 't'); accur=(1e-3, 1e-3)
# convergence=[]; accur = []
# convergence=('t',); accur=(1e-3,)

# Defining a function that will run a cdmft procedure within controlled_loop()
def run_cdmft():
    try:
        alpha = X.alpha
    except:
        alpha = 0.0
    X = CDMFT(M.model, varia=varia, wc=10, grid_type='self', accur=accur, convergence=convergence, converge_with_stdev=False, miniter=1, maxiter=64, depth=1, iteration='Broyden', alpha=alpha)
    return X.I

# Looping over values of U
M.model.controlled_loop(
    task=run_cdmft, 
    varia=["tb1_1", "eb1_1"], 
    loop_param="U", 
    loop_range=(2, 4.1, 0.5)
)