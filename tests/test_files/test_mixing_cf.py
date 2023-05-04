import pyqcm
from model_1D_4 import model
import matplotlib.pyplot as plt
mixing = 3

if mixing == 4:
    model.set_target_sectors(['R0:N4:S0'])
    model.set_parameters("""
    U=1
    t=1
    H = -1
    mu = 1
    """)

elif mixing == 3:
    model.set_target_sectors(['R0'])
    model.set_parameters("""
    U=1
    t=1
    S = 0.4
    H = 0.5
    Hx = 1
    mu = 1
    """)

elif mixing == 2:
    model.set_target_sectors(['R0:N4'])
    model.set_parameters("""
    U=1
    t=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    model.set_target_sectors(['R0:S0'])
    model.set_parameters("""
    U=1
    t=1
    S = 0.4
    H = -1
    mu = 1
    """)

elif mixing == 0:
    model.set_target_sectors(['R0:N4:S0'])
    model.set_parameters("""
    U=1
    t=1
    mu = 1
    """)

I = pyqcm.model_instance(model)
I.spectral_function(wmax = 6, nk = 32, path='line', file='test_mixing_{:d}.pdf'.format(mixing))

I.cluster_spectral_function(wmax = 4, plt_ax=plt.gca())
pyqcm.set_global_parameter('continued_fraction')
model.set_parameter('U', 1)
I = pyqcm.model_instance(model)
I.cluster_spectral_function(wmax = 4, plt_ax=plt.gca(), color='r')
plt.savefig("test_mixing_cf.pdf")
