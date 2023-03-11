from pyqcm import *
from pyqcm._spectral import *
set_global_parameter('nosym')
import model_1D_4

mixing = 1

if mixing == 4:
    set_target_sectors(['R0:N4:S0'])
    set_parameters("""
    t=1
    tsi=1
    mu = 1
    """)

elif mixing == 3:
    set_target_sectors(['R0'])
    set_parameters("""
    t=1
    tsi=1
    S = 1e-9
    Hx = 1e-9
    mu = 1
    """)

elif mixing == 2:
    set_target_sectors(['R0:N4'])
    set_parameters("""
    t=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    set_target_sectors(['R0:S0'])
    set_parameters("""
    t=1
    tsi=1
    S = 1e-9
    mu = 1
    """)

elif mixing == 0:
    set_target_sectors(['R0:N4:S0'])
    set_parameters("""
    t=1
    mu = 1
    """)

new_model_instance()
spectral_function(wmax = 6, nk = 32, path='line', file='test_mixing_{:d}.pdf'.format(mixing))

