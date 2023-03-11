from pyqcm import *
from pyqcm._spectral import *
import model_1D_4_C2

mixing = 3

if mixing == 4:
    set_target_sectors(['R0:N4:S0'])
    set_parameters("""
    U=1
    t=1
    H = -1
    mu = 1
    """)

elif mixing == 3:
    set_target_sectors(['R0'])
    set_parameters("""
    U=1
    t=1
    S = 0.4
    H = 0.5
    Hx = 1
    mu = 1
    """)

elif mixing == 2:
    set_target_sectors(['R0:N4'])
    set_parameters("""
    U=1
    t=1
    H = -1
    Hx = 1e-8 
    mu = 1
    """)

elif mixing == 1:
    set_target_sectors(['R0:S0'])
    set_parameters("""
    U=1
    t=1
    S = 0.4
    H = -1
    mu = 1
    """)

elif mixing == 0:
    set_target_sectors(['R0:N4:S0'])
    set_parameters("""
    U=1
    t=1
    mu = 1
    """)

new_model_instance()
cluster_spectral_function(wmax = 4, plt_ax=plt.gca(), file="test_mixing_nocf.pdf")
set_global_parameter('continued_fraction')
set_parameter('U', 1)
new_model_instance()
cluster_spectral_function(wmax = 4, plt_ax=plt.gca(), file="test_mixing_cf.pdf", color='r')
plt.show()
