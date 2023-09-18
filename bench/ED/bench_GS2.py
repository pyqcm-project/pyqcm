import pyqcm
import numpy as np
import gc
from model_2D import model2D
import timeit
import psutil

# pyqcm.set_global_parameter('verb_ED')
# pyqcm.set_global_parameter('verb_Hilbert')
# pyqcm.set_global_parameter('Ground_state_method', 'L')
# pyqcm.set_global_parameter('PRIMME_algorithm', 1)
# pyqcm.set_global_parameter('Davidson_states', 2)
# pyqcm.set_global_parameter('PRIMME_preconditionning', 1)

# pyqcm.set_global_parameter('CSR_sym_store')
F = 'E'
Lx = 6
Ly = 2
model = model2D(Lx, Ly, sym=True)
pyqcm.set_global_parameter('Hamiltonian_format', F)
model.set_target_sectors('R0:N{:d}:S0'.format(Lx*Ly))
model.set_parameters("""
    U = 4
    t = 1
    mu = 0.5*U
""")

process = psutil.Process()

# Iset = []
def go():
    for i in range(5):
        I = pyqcm.model_instance(model, i)
        # Iset.append(I)
        T = timeit.default_timer()
        print("ground state : ", I.ground_state())
        T = timeit.default_timer() - T
        print('RAM Used (MB):', process.memory_info().rss/(1024*1024))
        print('instance is tracked : ', gc.is_tracked(I))
        del I
        # print('instance is finalized : ', gc.is_finalized(I))
        # print('instance is tracked : ', gc.is_tracked(I))
        print('GC : ', gc.get_count())
        gc.collect()
        print('RAM Used (MB):', process.memory_info().rss/(1024*1024))
        print('format {:s} : time = {:1.2f}'.format(F, T))
    print('go is finished')

go()

