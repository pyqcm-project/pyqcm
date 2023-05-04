import pyqcm
pyqcm.set_global_parameter('nosym')
from model_1D_4 import model

mixing = 1

if mixing == 4:
    model.set_target_sectors(['R0:N4:S0'])
    model.set_parameters("""
    t=1
    tsi=1
    mu = 1
    """)

elif mixing == 3:
    model.set_target_sectors(['R0'])
    model.set_parameters("""
    t=1
    tsi=1
    S = 1e-9
    Hx = 1e-9
    mu = 1
    """)

elif mixing == 2:
    model.set_target_sectors(['R0:N4'])
    model.set_parameters("""
    t=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    model.set_target_sectors(['R0:S0'])
    model.set_parameters("""
    t=1
    tsi=1
    S = 1e-9
    mu = 1
    """)

elif mixing == 0:
    model.set_target_sectors(['R0:N4:S0'])
    model.set_parameters("""
    t=1
    mu = 1
    """)

I = pyqcm.model_instance(model)
I.spectral_function(wmax = 6, nk = 32, path='line', file='test_mixing_{:d}.pdf'.format(mixing))

