import pyqcm
import matplotlib.pyplot as plt
from model_1D_4 import model

sec = 'R0:N4:S0'
model.set_target_sectors([sec])
model.set_parameters("""
t=1
U = 4
mu = 0.5*U
""")

n = 4
U = [1e-9, 2, 4, 6]
I = []
for i,u in enumerate(U):
    model.set_parameter('U', u)
    I.append(pyqcm.model_instance(model))

for i,u in enumerate(U):
    I[i].spectral_function(wmax=6, path='line', plt_ax=plt.gca(), file=f"test_instances_U{u}.pdf")
