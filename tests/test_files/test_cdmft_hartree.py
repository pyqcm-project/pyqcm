# Test file
# Goal : to test CDMFT on a 1D cluster with extended interactions treated in the Hartree approximation
#--------------------------------------------------------------------------------
import pyqcm
import numpy as np
from pyqcm.cdmft import CDMFT

#--------------------------------------------------------------------------------
# defining the model
clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])
clus.new_operator('sb1', 'anomalous', [(1, 9, -1.0), (2, 10, -1.0), (3, 7, 1.0), (4, 8, 1.0)])
clus.new_operator('sb2', 'anomalous', [(1, 11, -1.0), (2, 12, -1.0), (5, 7, 1.0), (6, 8, 1.0)])
clus.new_operator_complex('sbi1', 'anomalous', [(1, 9, (-0-1j)), (2, 10, (-0-1j)), (3, 7, 1j), (4, 8, 1j)])
clus.new_operator_complex('sbi2', 'anomalous', [(1, 11, (-0-1j)), (2, 12, (-0-1j)), (5, 7, 1j), (6, 8, 1j)])
clus.new_operator('pb1', 'anomalous', [(3, 10, 1.0), (4, 9, -1.0), (5, 12, 1.0), (6, 11, -1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('1D_2_4b', clus0, ((2, 0, 0),))

model.interaction_operator('U')
model.interaction_operator('V', link=(1, 0, 0))
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('ti', (1, 0, 0), -1, tau=2)
model.hopping_operator('tp', (2, 0, 0), -1)
model.hopping_operator('sf', (0, 0, 0), -1, tau=0, sigma=1)
model.hopping_operator('h', (0, 0, 0), -1, tau=0, sigma=3)
model.anomalous_operator('D', (1, 0, 0), 1)
model.anomalous_operator('Di', (1, 0, 0), 1j)
model.anomalous_operator('S', (0, 0, 0), 1)
model.anomalous_operator('Si', (0, 0, 0), 1j)
model.anomalous_operator('dz', (1, 0, 0), 1, type='dz')
model.anomalous_operator('dy', (1, 0, 0), 1, type='dy')
model.anomalous_operator('dx', (1, 0, 0), 1, type='dx')
model.density_wave('M', 'Z', (1, 0, 0))
model.density_wave('pT', 'dz', (1, 0, 0), amplitude=1, link=(1, 0, 0))
model.explicit_operator('Vm', [((0, 0, 0), (0, 0, 0), np.float64(0.7071067811865475)), ((1, 0, 0), (0, 0, 0), np.float64(0.7071067811865475))], type='one-body', tau=0)
model.Vm_eig = 1
#--------------------------------------------------------------------------------

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

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']
def run_cdmft(iteration):
    try:
        alpha = X.alpha
    except:
        alpha = 0.0
    U = model.parameters()['U']
    V = model.parameters()['V']
    model.set_parameter('mu', 0.5*U+2*V)
    try:
        X = CDMFT(model, varia, iteration=iteration, convergence=('GS energy', 'parameters'), accur=(1e-4, 1e-4), eps_algo=2, alpha = alpha, hartree=(Vm_H,)) 
        return X.I
    except:
        raise pyqcm.SolverError('Failure of the CDMFT method')

# Looping over values of U
model.controlled_loop(
    task=lambda : run_cdmft('fixed_point'),
    varia = varia,
    loop_param='U', 
    loop_range=(2, 4.1, 0.5)
)

model.controlled_loop(
    task=lambda : run_cdmft('broyden'), 
    varia = varia,
    loop_param='U', 
    loop_range=(2, 4.1, 0.5)
)