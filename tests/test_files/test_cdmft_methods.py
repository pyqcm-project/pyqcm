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
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']

# convergence = 'GS energy'; accur = 1e-3
# convergence='self-energy'; accur = 1e-5
# convergence='t'; accur = 1e-4
# convergence='parameters'; accur = 1e-4
convergence=('parameters', 'hybridization', 'self-energy'); accur = (5e-6, 1e-4, 1e-4)
# convergence=('mu', 't'); accur=(1e-3, 1e-3)
# convergence=[]; accur = []
# convergence=('t',); accur=(1e-3,)

# Defining a function that will run a cdmft procedure within controlled_loop()
def run_cdmft(Z):
    try:
        alpha = X.alpha
    except:
        alpha = 0.0
    X = CDMFT(M.model, varia=varia, wc=10, grid_type='self', method=Z, accur=accur, convergence=convergence, converge_with_stdev=False, miniter=1, maxiter=64, depth=1, iteration='Broyden', alpha=alpha)
    return X.I

# loop over methods:

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'ANNEAL', 'NELDERMEAD', 'COBYLA', 'BOBYQA', 'PRAXIS', 'SUBPLEX']

for Z in methods:
    print('\n\nMETHOD = ', Z, '\n')
    M.model.set_parameter('eb1_1', 1.0)
    M.model.set_parameter('eb2_1', -1.1)
    M.model.set_parameter('tb1_1', 0.5)
    M.model.set_parameter('tb2_1', 0.4)
    try:
    	run_cdmft(Z)
    except: 
        print('Failure with method ', Z)