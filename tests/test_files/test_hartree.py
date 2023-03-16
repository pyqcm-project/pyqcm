import pyqcm
from pyqcm.vca import *
import numpy as np

import model_hartree as M

# Parameters required for half-filling
M.model.set_target_sectors(["R0:N4:S0"])
M.model.set_parameters("""
    U=4
    V=4
    Vm=5
    mu=10
    t=1
    Delta_1=1e-9
    Delta=1e-9
""")


# Updates mu as a function of V and U
def _adjust_mu():
    V = M.model.parameters()["V"]
    U = M.model.parameters()["U"]
    M.model.set_parameter("mu", 2*V + U/2) # Condition to impose half-filling
    
# Defining an object of the hartree class with eigenvalue 1
Vm_obj = pyqcm.hartree(M.model, "Vm", "V", 1, lattice=True) # lattice=True ---> use lattice averages

# Range for sweep over V 
V_start = 4
V_stop = 0
V_step = -0.2

def F():
    return pyqcm.model_instance(M.model)

############################# - Self-consistency approach - #############################

# This is simply a loop that applies the self_consistency approach over V_range
for V in np.arange(V_start, V_stop, V_step):
    M.model.set_parameter("V", V)
    _adjust_mu()

    M.model.Hartree_procedure(F, [Vm_obj])

################################### - VCA approach - ####################################

# This is the function to run inside of controlled_loop to perform the vca itself
def _run_vca():
    _adjust_mu()
    V = VCA(M.model, varia=["Vm"], steps=[0.05], accur=[2e-3], max=[20], hartree=[Vm_obj], max_iter=300)
    return V.I 


M.model.set_parameter("Vm", 5.65) # Initial guess for the VCA based on the Hartee self-consistency
_adjust_mu()

# performing the vca loop over V_range
M.model.controlled_loop(_run_vca, varia=["Vm"], loop_param="V", loop_range=(V_start, V_stop, V_step))
