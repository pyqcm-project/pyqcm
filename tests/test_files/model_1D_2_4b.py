import pyqcm
import numpy as np

ns = 2
nb = 4
no = ns+nb
CM = pyqcm.cluster_model(ns, nb)

CM.new_operator('tb1', 'one-body', [
    (1, 3, -1.0),
    (2, 4, -1.0),
    (7, 9, -1.0),
    (8, 10, -1.0)
])

CM.new_operator('tb2', 'one-body', [
    (1, 5, -1.0),
    (2, 6, -1.0),
    (7, 11, -1.0),
    (8, 12, -1.0)
])

CM.new_operator('eb1', 'one-body', [
    (3, 3, 1.0),
    (4, 4, 1.0),
    (9, 9, 1.0),
    (10, 10, 1.0)
])

CM.new_operator('eb2', 'one-body', [
    (5, 5, 1.0),
    (6, 6, 1.0),
    (11, 11, 1.0),
    (12, 12, 1.0)
])

CM.new_operator('sb1', 'anomalous', [
    (1, 3+no, -1.0),
    (2, 4+no, -1.0),
    (3, 1+no, 1.0),
    (4, 2+no, 1.0)
])

CM.new_operator('sb2', 'anomalous', [
    (1, 5+no, -1.0),
    (2, 6+no, -1.0),
    (5, 1+no, 1.0),
    (6, 2+no, 1.0)
])

CM.new_operator_complex('sbi1', 'anomalous', [
    (1, 3+no, -1.0j),
    (2, 4+no, -1.0j),
    (3, 1+no, 1.0j),
    (4, 2+no, 1.0j)
])

CM.new_operator_complex('sbi2', 'anomalous', [
    (1, 5+no, -1.0j),
    (2, 6+no, -1.0j),
    (5, 1+no, 1.0j),
    (6, 2+no, 1.0j)
])

CM.new_operator('pb1', 'anomalous', [
    (3, 4+no, 1.0),
    (4, 3+no, -1.0),
    (5, 6+no, 1.0),
    (6, 5+no, -1.0)
])

#-------------------------------------------------------------------
# construction of the lattice model 

clus = pyqcm.cluster(CM, (( 0, 0, 0), ( 1, 0, 0)))
model = pyqcm.lattice_model('1D_2_4b', clus, ((2, 0, 0),))

model.interaction_operator('U')
model.interaction_operator('V', link=(1,0,0))
model.hopping_operator('t', ( 1, 0, 0), -1) # NN hopping
model.hopping_operator('ti', ( 1, 0, 0), -1, tau=2) # NN hopping with imaginary amplitude
model.hopping_operator('tp', ( 2, 0, 0), -1) # NNN hopping
model.hopping_operator('sf', ( 0, 0, 0), -1, tau=0, sigma=1) # on-site spin flip
model.hopping_operator('h', ( 0, 0, 0), -1, tau=0, sigma=3) # on-site spin flip
model.anomalous_operator('D', ( 1, 0, 0), 1) # NN singlet
model.anomalous_operator('Di', ( 1, 0, 0), 1j) # NN singlet with imaginary amplitude
model.anomalous_operator('S', ( 0, 0, 0), 1) # on-site singlet 
model.anomalous_operator('Si', ( 0, 0, 0), 1j) # on-site singlet with imaginary amplitude
model.anomalous_operator('dz', ( 1, 0, 0), 1, type='dz') # NN triplet
model.anomalous_operator('dy', ( 1, 0, 0), 1, type='dy') # NN triplet
model.anomalous_operator('dx', ( 1, 0, 0), 1, type='dx') # NN triplet
model.density_wave('M', 'Z', ( 1, 0, 0))
model.density_wave('pT', 'dz', ( 1, 0, 0), amplitude=1, link=( 1, 0, 0))

# Defining an explicit operator for the "Hartree part" of the extended interaction
elems = [(( 0, 0, 0), ( 0, 0, 0), 1/np.sqrt(2)), ((1,0,0), (0,0,0), 1/np.sqrt(2))] 
model.explicit_operator('Vm', elems, type='one-body', tau=0) 
model.Vm_eig = 1