# Test file
# Goal : to test the hdf5 I/O of a single instance 
#--------------------------------------------------------------------------------
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
I.write_hdf5('test.h5')
I = pyqcm.model_instance(model)
I.read_hdf5('test.h5', set_parameters=True)
print('\nGF read:\n', I.cluster_Green_function(0.1j, 0))
