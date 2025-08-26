import pyqcm
import numpy as np
from model_1D_4 import model

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
I = pyqcm.model_instance(model)
ave = I.averages(pr=True)
P = model.parameters()
print('current = ', ave['It']*P['t'] + ave['Iti']*P['ti'] )

