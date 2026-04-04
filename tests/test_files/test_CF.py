# Test file
# Goal : to test various GF solvers
#--------------------------------------------------------------------------------
import pyqcm
pyqcm.set_global_parameter('nosym')

L=2
f=8
from model_1D import model1D
model = model1D(L, sym=False)
pyqcm.set_global_parameter('max_iter_BL', f)

mixing = 0
cluster = False
# cluster = True

if mixing == 4:
    model.set_target_sectors(['R0:N{:d}:S0'.format(L)])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    mu = 1
    """)

elif mixing == 3:
    model.set_target_sectors(['R0'])
    model.set_parameters("""
    U = 4
    t=1
    tsi=1
    S = 0.1
    Hx = 0.2
    mu = 1
    """)

elif mixing == 2:
    model.set_target_sectors(['R0:N{:d}'.format(L)])
    model.set_parameters("""
    U = 4
    t=1
    H = 0
    Hx = 0.1 
    mu = 1
    """)

elif mixing == 1:
    model.set_target_sectors(['R0:S0'])
    model.set_parameters("""
    U = 4
    ti=1
    S = 0.2
    mu = 1
    """)

elif mixing == 0:
    model.set_target_sectors(['R0:N{:d}:S0'.format(L)])
    model.set_parameters("""
U = 4
t=1
# ti=1
mu = 2
    """)

w = 0.5+0.01j

pyqcm.set_global_parameter('GF_method', 'L')
I = pyqcm.model_instance(model)
print(I.cluster_Green_function(w))
 
pyqcm.set_global_parameter('GF_method', 'F')
I = pyqcm.model_instance(model)
print(I.cluster_Green_function(w))

pyqcm.set_global_parameter('GF_method', 'M')
I = pyqcm.model_instance(model)
print(I.cluster_Green_function(w))

# pyqcm.set_global_parameter('combine_mcf')
# pyqcm.set_global_parameter('GF_method', 'M')
# I = pyqcm.model_instance(model)
# print(I.cluster_Green_function(w))

import matplotlib.pyplot as plt

w =  -0.5+0.01j
# plt.gcf().set_size_inches(12/2.54, 12/2.54)

# Lehman representation, band Lanczos
pyqcm.set_global_parameter('GF_method', 'L')
I = pyqcm.model_instance(model)
print("GF_method = 'L'")
print(I.cluster_Green_function(w))

# MCF representation, band Lanczos
pyqcm.set_global_parameter('GF_method', 'L')
pyqcm.set_global_parameter('combine_mcf')
I = pyqcm.model_instance(model)
print("GF_method = 'L', combined_mfc = True")
print(I.cluster_Green_function(w))
W, A, B = I.combined_mcf(pr=True)
pyqcm.set_global_parameter('combine_mcf', False)

# CF representation
pyqcm.set_global_parameter('GF_method', 'F')
I = pyqcm.model_instance(model)
print("GF_method = 'F'")
print(I.cluster_Green_function(w))

# MCF representation (Ge and Gh separate)
pyqcm.set_global_parameter('GF_method', 'M')
I = pyqcm.model_instance(model)
print("GF_method = 'M'")
print(I.cluster_Green_function(w))

# MCF representation (Ge + Gh combined into one MCF)
pyqcm.set_global_parameter('combine_mcf')
pyqcm.set_global_parameter('GF_method', 'M')
I = pyqcm.model_instance(model)
print("GF_method = 'M', combined_mfc = True")
print(I.cluster_Green_function(w))
W, A, B = I.combined_mcf(pr=True)

pyqcm.set_global_parameter('combine_mcf', False)

