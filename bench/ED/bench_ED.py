import pyqcm
import numpy as np
from model_1D import model1D
import timeit

pyqcm.set_global_parameter('verb_ED')
pyqcm.set_global_parameter('verb_Hilbert')
#pyqcm.set_global_parameter('parallel_sectors')
pyqcm.set_global_parameter('seed',1234)
F = 'E'
L = 14
model = model1D(L,sym=True)
pyqcm.set_global_parameter('Hamiltonian_format', F)
model.set_target_sectors('R0:N{:d}:S0'.format(L))
model.set_parameters("""
    U = 4
    t = 1
    mu = 0.5*U
""")
I = pyqcm.model_instance(model)
T = timeit.default_timer()
print("ground state : ", I.ground_state())
I.cluster_Green_function(0.1j, 0)
T = timeit.default_timer() - T
print('format {:s} : time = {:1.2f}'.format(F, T))
