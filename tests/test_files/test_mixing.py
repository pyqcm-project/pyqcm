import pyqcm
import model_1D_4_C2 as M
import matplotlib.pyplot as plt

mixing = 0

if mixing == 4:
    M.model.set_target_sectors(['R0:N4:S0'])
    M.model.set_parameters("""
    U=1
    t=1
    H = -1
    mu = 1
    """)

elif mixing == 3:
    M.model.set_target_sectors(['R0'])
    M.model.set_parameters("""
    U=1
    t=1
    S = 0.4
    H = 0.5
    Hx = 1
    mu = 1
    """)

elif mixing == 2:
    M.model.set_target_sectors(['R0:N4'])
    M.model.set_parameters("""
    U=1
    t=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    M.model.set_target_sectors(['R0:S0'])
    M.model.set_parameters("""
    U=1
    t=1
    S = 0.4
    H = -1
    mu = 1
    """)

elif mixing == 0:
    M.model.set_target_sectors(['R0:N4:S0'])
    M.model.set_parameters("""
    U=1
    t=1
    mu = 1
    """)

I = pyqcm.model_instance(M.model)
I.cluster_spectral_function(wmax = 4, plt_ax=plt.gca(), file="test_mixing_nocf.pdf")
# pyqcm.set_global_parameter('continued_fraction')
M.model.set_parameter('U', 1)
I = pyqcm.model_instance(M.model)
I.cluster_spectral_function(wmax = 4, plt_ax=plt.gca(), file="test_mixing_cf.pdf", color='r')

