import pyqcm
pyqcm.set_global_parameter('nosym')

L=4
from model_1D_4 import model

mixing = 0
cluster = False

if mixing == 4:
    model.set_target_sectors(['R0:N{:d}:S0'.format(L)])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    mu = 1
    """)

elif mixing == 3:
    model.set_target_sectors(['R0'])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    S = 1e-9
    Hx = 1e-9
    mu = 1
    """)

elif mixing == 2:
    model.set_target_sectors(['R0:N{:d}'.format(L)])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    model.set_target_sectors(['R0:S0'])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    S = 1e-9
    mu = 1
    """)

elif mixing == 0:
    model.set_target_sectors(['R0:N{:d}:S0'.format(L)])
    model.set_parameters("""
U = 4
# t=1
ti=1
mu = 1
    """)

import matplotlib.pyplot as plt

# plt.gcf().set_size_inches(12/2.54, 12/2.54)

I = pyqcm.model_instance(model)
if cluster:
    I.cluster_spectral_function(wmax = 6, plt_ax = plt.gca(), full=True, offset=14)
else:
    I.spectral_function(wmax = 6, plt_ax = plt.gca(), path='line')

I = pyqcm.model_instance(model)
pyqcm.set_global_parameter('continued_fraction')
I = pyqcm.model_instance(model)
if cluster:
    I.cluster_spectral_function(wmax = 6, full=True, offset=14, plt_ax = plt.gca(), color='r')
else:
    I.spectral_function(wmax = 6, plt_ax = plt.gca(), path='line', color='r')

plt.savefig('test_CF.pdf')