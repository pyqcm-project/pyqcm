from pyqcm import *
from pyqcm.cdmft import *
from pyqcm.loop import *
from pyqcm.variable_parameter import *


import model_1D_2_4b

acc = 1e-3
np.set_printoptions(precision=4, linewidth=512, suppress=True)

def Uvsn(n):
    return 4 -2*(1-n)

Uvar = variable_parameter('U', Uvsn, obs='mu', lattice=False)

sec = 'R0:N6:S0/R0:N4:S0/R0:N2:S0'
set_target_sectors([sec])

set_parameters("""
    U=4
    mu=1
    t=1
    tb1_1=0.6
    tb2_1=0.5
    eb1_1=1.2
    eb2_1=-1
""")

varia=['tb1_1', 'tb2_1', 'eb1_1', 'eb2_1']

def F():
    cdmft(varia=varia, accur=1e-4, accur_hybrid=1e-8, accur_dist=1e-10, check_sectors=True)

variable_parameter_self_consistency(F, Uvar)
