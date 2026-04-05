# Test file
# Goal : to test the Hartree procedure in two ways : self-consistent and VCA
#--------------------------------------------------------------------------------
import pyqcm
from pyqcm.vca import *
import numpy as np

#--------------------------------------------------------------------------------
# model definition
clus = pyqcm.cluster_model(4, name='clus')

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('1D_4', clus0, ((4, 0, 0),))

model.interaction_operator('U', amplitude=1)
model.interaction_operator('V', link=(1, 0, 0), amplitude=1)
model.explicit_operator('Vm', [((0, 0, 0), (0, 0, 0), np.float64(0.7071067811865475)), ([3, 0, 0], (0, 0, 0), np.float64(0.7071067811865475))], type='one-body', tau=0)
model.hopping_operator('t', (1, 0, 0), -1)
model.density_wave('Delta', 'N', (1, 0, 0), amplitude=1)
#--------------------------------------------------------------------------------
# Parameters required for half-filling
model.set_target_sectors(["R0:N4:S0"])
model.set_parameters("""
    U=4
    V=4
    Vm=5
    mu=10
    t=1
    Delta_1=1e-9
    Delta=1e-9
""")


#--------------------------------------------------------------------------------
# Updates mu as a function of V and U
def _adjust_mu():
    V = model.parameters()['V']
    U = model.parameters()['U']
    model.set_parameter('mu', 2*V + U/2) # Condition to impose half-filling
    
# Defining an object of the hartree class with eigenvalue 1
Vm_H = pyqcm.hartree(model, 'Vm', 'V', 1, lattice=True) # lattice=True ---> use lattice averages

# Range for sweep over V 
V_start = 4
V_stop = 2
V_step = -0.5

def F():
    return pyqcm.model_instance(model)

############################# - Self-consistency approach - #############################
alpha = 0.0

# This is simply a loop that applies the self_consistency approach over V_range
for V in np.arange(V_start, V_stop, V_step):
    model.set_parameter('V', V)
    _adjust_mu()

    model.Hartree_procedure(F, Vm_H, iteration='fixed_point')

for V in np.arange(V_start, V_stop, V_step):
    model.set_parameter('V', V)
    _adjust_mu()

    I, niter, alpha = model.Hartree_procedure(F, Vm_H, iteration='Broyden', alpha = alpha)
################################### - VCA approach - ####################################

# This is the function to run inside of controlled_loop to perform the vca itself
def _run_vca():
    _adjust_mu()
    V = VCA(model, varia=['Vm'], steps=[0.05], accur=[2e-3], max=[20], hartree=Vm_H, max_iter=300)
    return V.I 


model.set_parameter('Vm', 5.65) # Initial guess for the VCA based on the Hartee self-consistency
_adjust_mu()

# performing the vca loop over V_range
model.controlled_loop(_run_vca, varia='Vm', loop_param='V', loop_range=(V_start, V_stop, V_step))
