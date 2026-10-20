# Test file
# Goal : to test the compact_tiling() function
#--------------------------------------------------------------------------------
import pyqcm
import numpy as np

L=6
from model_1D import model1D
model = model1D(L, sym=False)

model.set_target_sectors(['R0:N{:d}:S0'.format(L)])
model.set_parameters("""
U = 0
t=-1
tp = 0.5
mu = 0
""")


I = pyqcm.model_instance(model)
A = I.cluster_hopping_matrix()

print('A = \n', A)
k = np.array([0.25, 0, 0])
Act = model.compact_tiling(A, k)
print('Act = \n', Act)

print('tk = \n', I.tk(k))
