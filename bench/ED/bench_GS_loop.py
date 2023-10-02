import pyqcm
import numpy as np
from model_1D import model1D
import timeit

# pyqcm.set_global_parameter('verb_ED')
# pyqcm.set_global_parameter('verb_Hilbert')
# pyqcm.set_global_parameter('max_dim_full', 6)
pyqcm.set_global_parameter('Ground_state_init_last')
# pyqcm.set_global_parameter('Ground_state_method', 'P')

# pyqcm.set_global_parameter('CSR_sym_store')
F = 'E'
L = 12
model = model1D(L,sym=False)
pyqcm.set_global_parameter('Hamiltonian_format', F)
model.set_target_sectors('R0:N{:d}:S0'.format(L))
model.set_parameters("""
    U = 4
    t = 1
    mu = 0.5*U
""")
                     
# I = pyqcm.model_instance(model)
# print("ground state : ", I.ground_state())
# model.set_parameter('U', 4.1)                  
# I = pyqcm.model_instance(model)
# print("ground state : ", I.ground_state())
# exit()


for u in np.arange(4,4.1,0.01):
    model.set_parameter('U', u)
    I = pyqcm.model_instance(model)
    T = timeit.default_timer()
    print("ground state : ", I.ground_state())
    T = timeit.default_timer() - T
    print('format {:s} : time = {:1.2f}'.format(F, T))


