import pyqcm
from pyqcm.spectral import *
# pyqcm.set_global_parameter("nosym")
import model_1D_4

pyqcm.set_global_parameter("continued_fraction")

pyqcm.set_target_sectors(['R0:N4:S0'])
pyqcm.set_parameters("""
t=1
U=2
mu = 1
""")

DoS(w=5, plt_ax=0)
# spectral_function(path="line")

