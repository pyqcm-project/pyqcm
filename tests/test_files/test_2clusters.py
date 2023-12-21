import numpy as np
import pyqcm
from pyqcm.cdmft import CDMFT

from model_G4_6b_2C import model

# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N10:S0']*2)

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    t_1 = 1.1
    t_2 = 0.9
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
    tb1_2= -1*tb1_1
    tb2_2= -1*tb2_1
    eb1_2=1*eb1_1
    eb2_2=1*eb2_1
""")

I = pyqcm.model_instance(model)  
print('\nfull hopping matrix of cluster 1 :')              
A = I.cluster_hopping_matrix(clus=0, full=True)
print(np.real(A))

print('\nfull hopping matrix of cluster 2 :')              
A = I.cluster_hopping_matrix(clus=1, full=True)
print(np.real(A))

print('\nstrict hopping matrix of cluster 1 :')              
A = I.cluster_hopping_matrix(clus=0, full=False)
print(np.real(A))

print('\nstrict hopping matrix of cluster 2 :')              
A = I.cluster_hopping_matrix(clus=1, full=False)
print(np.real(A))

print('\ninteractions of cluster 1 :')              
print(I.interactions(clus=0))
print('\ninteractions of cluster 2 :')              
print(I.interactions(clus=1))


I.write_impurity_problem(clus=0, bath_diag=False, file='impurity1.tsv')
I.write_impurity_problem(clus=1, bath_diag=False, file='impurity2.tsv')

print('\ncluster Green function of cluster 1 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=0, spin_down=False, blocks=False))
print('\ncluster Green function of cluster 2 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=1, spin_down=False, blocks=False))

print('\ncluster self-energy of cluster 1 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=0, spin_down=False))
print('\ncluster self-energy of cluster 2 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=1, spin_down=False))

print('\nGreen function average of cluster 1:')
print(I.Green_function_average(clus=0, spin_down=False))
print('\nGreen function average of cluster 2:')
print(I.Green_function_average(clus=1, spin_down=False))



I.write('cluster1.out', clus=0)
I.write('cluster2.out', clus=1)

print('\nQ-matrix of cluster 1:')
print(I.qmatrix(clus=0))
print('\nQ-matrix of cluster 2:')
print(I.qmatrix(clus=1))

I.write_Green_function(clus=0, file='GF1.tsv')
I.write_Green_function(clus=1, file='GF2.tsv')

# I.plot_profile()