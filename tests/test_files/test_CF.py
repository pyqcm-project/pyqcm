import pyqcm
pyqcm.set_global_parameter('nosym')
from model_1D_4 import model

mixing = 0

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
    tsi=1
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
    tsi=1
    mu = 1
    """)

import matplotlib.pyplot as plt

plt.gcf().set_size_inches(12/2.54, 15/2.54)
plt.subplot(2,1,1)

I = pyqcm.model_instance(model)
I.spectral_function(wmax = 6, nk = 32, path='line', plt_ax = plt.gca())
plt.title('Lehmann')

I = pyqcm.model_instance(model, 1)
pyqcm.set_global_parameter('continued_fraction')
plt.subplot(2,1,2)
I = pyqcm.model_instance(model)
I.spectral_function(wmax = 6, nk = 32, path='line', plt_ax = plt.gca())
plt.title('Continued fraction')

plt.tight_layout()
plt.savefig('test_CF.pdf')