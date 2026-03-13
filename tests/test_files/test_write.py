import pyqcm
from model_1D import model1D

L=6
model = model1D(L)
model.set_target_sectors('R0:N{:d}:S0'.format(L))

model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
""")

I = pyqcm.model_instance(model)
print('GF written:\n', I.cluster_Green_function(0.1j, 0))
I.write('test.out')
I = pyqcm.model_instance(model)
I.read('test.out')
print('\nGF read:\n', I.cluster_Green_function(0.1j, 0))
