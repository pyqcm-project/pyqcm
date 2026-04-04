# Test file
# Goal : to test a 2-cluster model in an elementary way, I/O on hdf5 file included
#--------------------------------------------------------------------------------

import numpy as np
import pyqcm


#--------------------------------------------------------------------------------
# First, defining the model (a 2-cluster model for graphene with bath)

clus = pyqcm.cluster_model(4, n_bath=6, name='clus')
clus.new_operator('eb1', 'one-body', [(5, 5, 1), (15, 15, 1), (7, 7, 1), (17, 17, 1), (9, 9, 1), (19, 19, 1)])
clus.new_operator('eb2', 'one-body', [(6, 6, 1), (16, 16, 1), (8, 8, 1), (18, 18, 1), (10, 10, 1), (20, 20, 1)])
clus.new_operator('tb1', 'one-body', [(1, 5, 1), (11, 15, 1), (2, 7, 1), (12, 17, 1), (3, 9, 1), (13, 19, 1)])
clus.new_operator('tb2', 'one-body', [(1, 6, 1), (11, 16, 1), (2, 8, 1), (12, 18, 1), (3, 10, 1), (13, 20, 1)])

clus0 = pyqcm.cluster(clus, ((-1, -1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0)), pos=(1, 0, 0))
clus1 = pyqcm.cluster(clus, ((1, 1, 0), (0, -1, 0), (-1, 0, 0), (0, 0, 0)), pos=(-1, 0, 0))

model = pyqcm.lattice_model('graphene_4_2C', [clus0, clus1], ((4, 2, 0), (2, -2, 0)), lattice=((1, -1, 0), (2, 1, 0)))
model.set_basis([( 1, 0, 0),[-0.5,np.sqrt(3)/2,0]])

model.interaction_operator('U')
model.hopping_operator('t', (-1, 0, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (0, -1, 0), 1, orbitals=(1, 2))
model.hopping_operator('t', (1, 1, 0), 1, orbitals=(1, 2))
#--------------------------------------------------------------------------------


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

#--------------------------------------------------------------------------------
# testing write_definition
model.write_definition('model_2clusters.py')


#--------------------------------------------------------------------------------
# various Green function related quantites

I = pyqcm.model_instance(model)  
print('\nstrict hopping matrix of cluster 1 :')              
A = I.cluster_hopping_matrix(clus=0)
print(np.real(A))

print('\nstrict hopping matrix of cluster 2 :')              
A = I.cluster_hopping_matrix(clus=1)
print(np.real(A))

print('\ninteractions of cluster 1 :')              
print(I.interactions(sys=0))
print('\ninteractions of cluster 2 :')              
print(I.interactions(sys=1))

print('\ncluster Green function of cluster 1 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=0, spin_down=False, blocks=False))
print('\ncluster Green function of cluster 2 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=1, spin_down=False, blocks=False))

print('\ncluster self-energy of cluster 1 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=0, spin_down=False))
print('\ncluster self-energy of cluster 2 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=1, spin_down=False))

print('\nGreen function average of cluster 1:')
print(I.Green_function_average(sys=0, spin_down=False))
print('\nGreen function average of cluster 2:')
print(I.Green_function_average(sys=1, spin_down=False))

I.write_all_hdf5('instance_2clusters.hd5')

#--------------------------------------------------------------------------------
# resets, then reads the model instance from hdf5 and recomputes the same

pyqcm.reset_model()


I = pyqcm.read_model_instance('instance_2clusters.hd5')

print('\nstrict hopping matrix of cluster 1 :')              
A = I.cluster_hopping_matrix(clus=0)
print(np.real(A))

print('\nstrict hopping matrix of cluster 2 :')              
A = I.cluster_hopping_matrix(clus=1)
print(np.real(A))

print('\ninteractions of cluster 1 :')              
print(I.interactions(sys=0))
print('\ninteractions of cluster 2 :')              
print(I.interactions(sys=1))

print('\ncluster Green function of cluster 1 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=0, spin_down=False, blocks=False))
print('\ncluster Green function of cluster 2 at z=0.1j:')
print(I.cluster_Green_function(0.1j, clus=1, spin_down=False, blocks=False))

print('\ncluster self-energy of cluster 1 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=0, spin_down=False))
print('\ncluster self-energy of cluster 2 at z=0.1j:')
print(I.cluster_self_energy(0.1j, clus=1, spin_down=False))

print('\nGreen function average of cluster 1:')
print(I.Green_function_average(sys=0, spin_down=False))
print('\nGreen function average of cluster 2:')
print(I.Green_function_average(sys=1, spin_down=False))
