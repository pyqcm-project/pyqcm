import pyqcm
import numpy as np
from model_2D import model2D
import timeit

pyqcm.set_global_parameter('verb_ED')
pyqcm.set_global_parameter('verb_Hilbert')
# pyqcm.set_global_parameter('parallel_sectors')
# pyqcm.set_global_parameter('Ground_state_method', 'L')
# pyqcm.set_global_parameter('PRIMME_algorithm', 1)
# pyqcm.set_global_parameter('Davidson_states', 2)
# pyqcm.set_global_parameter('PRIMME_preconditionning', 1)

# pyqcm.set_global_parameter('CSR_sym_store')
F = 'E'
Lx = 7
Ly = 2
model = model2D(Lx, Ly, sym=True)
pyqcm.set_global_parameter('Hamiltonian_format', F)
model.set_target_sectors('R0:N{:d}:S0'.format(Lx*Ly))
model.set_parameters("""
    U = 4
    t = 1
    mu = 0.5*U
""")
I = pyqcm.model_instance(model)
T = timeit.default_timer()
print("ground state : ", I.ground_state())
print('GF = ', I.cluster_Green_function(0.1j, 0)[0,0])
T = timeit.default_timer() - T
print('format {:s} : time = {:1.2f}'.format(F, T))


