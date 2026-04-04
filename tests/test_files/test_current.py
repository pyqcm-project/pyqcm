# Test file
# Goal : to test the computation of the current in a 1D system
#--------------------------------------------------------------------------------
import pyqcm
import numpy as np
from model_1D import model1D
model = model1D(4, sym=False)

pyqcm.set_global_parameter('accur_OP', 1e-8)
sec = 'R0:S0'

print(model.currents)

phase = np.pi/5
model.set_target_sectors([sec])
model.set_parameters("""
t={:g}
ti={:g}
U = 4
mu = 1
D = 0.5
It = 0
Iti = 0
""".format(np.cos(phase), np.sin(phase)))

P = model.parameters()
I = pyqcm.model_instance(model)
ave = I.averages(pr=True)
print('current = ', ave['It']*P['t'] + ave['Iti']*P['ti'] )

