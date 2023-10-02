import pyqcm
from model_small import model

pyqcm.set_global_parameter('verb_integrals')
pyqcm.set_global_parameter('verb_Hilbert')
pyqcm.set_global_parameter('seed', 1234)
model.set_target_sectors('R0:S0')
model.set_parameters("""
t = 1
tz = 1
U = 1
mu = 0
D = 0.2
""")
I = pyqcm.model_instance(model)
I.averages(pr=True)

