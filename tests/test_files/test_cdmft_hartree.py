import pyqcm
from pyqcm.cdmft import CDMFT, variational_set
from model_1D_2_4b import model

# import numpy as np
# A = np.kron([1,2], [10,20]).reshape((2,2))
# print(A); exit()

model.set_target_sectors('R0:N6:S0')
model.set_parameters("""
    U=4
    V=1
    Vm = 1
    mu=2
    t=1
    tb1_1=0.5
    tb2_1=0.55
    eb1_1=1.0
    eb2_1=-1.0
""")

Vm_H = pyqcm.hartree(model, 'Vm', 'V', model.Vm_eig, lattice=False) 
# Vm_H = pyqcm.hartree(model, 'Vm', 'V', model.Vm_eig, lattice=True) 

varia = variational_set(['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1'], hartree=(Vm_H,))
def run_cdmft(iteration):
    try:
        alpha = X.alpha
    except:
        alpha = 0.0
    U = model.parameters()['U']
    V = model.parameters()['V']
    model.set_parameter('mu', 0.5*U+2*V)
    try:
        X = CDMFT(model, varia=varia, iteration=iteration, convergence=('GS energy', 'parameters'), accur=(1e-4, 1e-4), eps_algo=2, alpha = alpha) 
        return X.I
    except:
        raise pyqcm.SolverError('Failure of the CDMFT method')

# Looping over values of U
model.controlled_loop(
    task=lambda : run_cdmft('fixed_point'), 
    varia=varia.var+['Vm'],
    loop_param='U', 
    loop_range=(2, 4.1, 0.5)
)

model.controlled_loop(
    task=lambda : run_cdmft('Broyden'), 
    varia=varia.var+['Vm'],
    loop_param='U', 
    loop_range=(2, 4.1, 0.5)
)