import pyqcm
import pyqcm.vca as vca
import numpy as np

import matplotlib.pyplot as plt

# A simple 1D 4 site cluster
CM = pyqcm.cluster_model( 4, 0)
clus = pyqcm.cluster(CM, ((0,0,0),(1,0,0),(2,0,0),(3,0,0)))
model = pyqcm.lattice_model('1D_4', clus, ((4,0,0),))

model.interaction_operator('U', amplitude=1) # on site interaction
model.interaction_operator('V', link=(1,0,0), amplitude=1) # NN interaction
model.hopping_operator('t', [1,0,0], -1) 
model.density_wave('Delta', 'N', (1,0,0), amplitude=1) # A charge density wave with period 2 in position space

# Defining an explicit operator for the "Hartree part" of the extended interaction
elems = [((0,0,0), (0,0,0), 1/np.sqrt(2)), ((3,0,0), (0,0,0), 1/np.sqrt(2))] # the 1/sqrt(2) factor ensures normalization
model.explicit_operator('Vm1', elems, type='one-body', tau=0) # Vm is an on-site operator ---> tau=0

elems = [((0,0,0), (0,0,0), 1/np.sqrt(2)), ((3,0,0), (0,0,0), -1/np.sqrt(2))] # the 1/sqrt(2) factor ensures normalization
model.explicit_operator('Vm2', elems, type='one-body', tau=0) # Vm is an on-site operator ---> tau=0

# Parameters required for half-filling
model.set_target_sectors('R0:N4:S0')
model.set_parameters("""
    U=4
    V=4
    Vm1=5
    mu=10
    t=1
    Delta_1=1e-9
    Delta=1e-9
""")
I = pyqcm.model_instance(model)

# Defining an object of the hartree class with eigenvalue 1
MF1 = pyqcm.hartree(model, 'Vm1', 'V', 1, accur=1e-2, lattice=True) # lattice=True ---> use lattice averages

#########################################################################################
############################## - Range for sweep over V - ###############################

V_start = 1e-9
V_stop = 4
V_step = 0.1
U = model.parameters()['U']

#########################################################################################
############################# - Self-consistency approach - #############################

# This is simply a loop that applies the self_consistency approach over V_range

def F():
    model.set_parameter('V', V)
    model.set_parameter('mu', 2*V + U/2) # Condition to impose half-filling
    return pyqcm.model_instance(model)

# for V in np.arange(V_start, V_stop, V_step):
#     model.Hartree_procedure(F, MF1, maxiter=32)

#########################################################################################
################################### - VCA approach - ####################################

# This is the function to run inside of controlled_loop to perform the vca itself
def run_vca():
    V = model.parameters()['V']
    model.set_parameter('mu', 2*V + U/2) # Condition to impose half-filling
    V = vca.VCA(model, varia='Vm1', steps=0.05, accur=2e-3, max=20, hartree=MF1, max_iter=64)    
    return V.I

# performing the vca loop over V_range
model.controlled_loop(run_vca, varia='Vm1', loop_param='V', loop_range=(V_start, V_stop, V_step))

#########################################################################################
############################# - VCA over both parameters - ##############################

# Function to perform the VCA
varia=('Vm1', 'Delta_1')
def run_vca_2():
    V = model.parameters()['V']
    model.set_parameter('mu', 2*V + U/2) # Condition to impose half-filling
    V = vca.VCA(model, varia=varia, steps=0.05, accur=2e-3, max=20, hartree=MF1, max_iter=64)    
    return V.I

# Looping from 4 to 3
model.controlled_loop(run_vca_2, varia=varia, loop_param='V', loop_range=(V_start, V_stop, V_step))

#########################################################################################
######################### - Hybrid VCA with self-consistency - ##########################

def run_vca_3():
    V = model.parameters()['V']
    model.set_parameter('mu', 2*V + U/2) # Condition to impose half-filling
    V = vca.VCA(model, varia='Delta_1', steps=0.05, accur=2e-3, max=20, max_iter=64, hartree= MF1, hartree_self_consistent=True)    
    return V.I

model.controlled_loop(run_vca_3, varia=('Delta_1', 'Vm1'), loop_param='V', loop_range=(V_start, V_stop, V_step))

