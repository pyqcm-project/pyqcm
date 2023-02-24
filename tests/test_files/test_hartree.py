from pyqcm import *
from pyqcm.hartree import *
from pyqcm.loop import controlled_loop, Hartree
from pyqcm.vca import *
import numpy as np

import model_hartree

# Parameters required for half-filling
set_target_sectors(["R0:N4:S0"])
set_parameters("""
    U=4
    V=4
    Vm=5
    mu=10
    t=1
    Delta_1=1e-9
    Delta=1e-9
""")


# Updates mu as a function of V and U
def update_mu():
    new_model_instance()

    V = parameters()["V"]
    U = parameters()["U"]
    set_parameter("mu", 2*V + U/2) # Condition to impose half-filling
    
    new_model_instance()


# Defining an object of the hartree class with eigenvalue 1
Vm_obj = hartree("Vm", "V", 1, lattice=True) # lattice=True ---> use lattice averages


#########################################################################################
############################## - Range for sweep over V - ###############################

V_start = 4
V_stop = 0
V_step = -0.2

#########################################################################################
############################# - Self-consistency approach - #############################

# This is simply a loop that applies the self_consistency approach over V_range
for V in np.arange(V_start, V_stop, V_step):
    set_parameter("V", V)
    new_model_instance()
    update_mu()

    Hartree(new_model_instance, [Vm_obj])

#########################################################################################
################################### - VCA approach - ####################################

# This is the function to run inside of controlled_loop to perform the vca itself
def run_vca():
    update_mu()
    vca(varia=["Vm"], steps=[0.05], accur=[2e-3], max=[20], hartree=[Vm_obj], max_iter=300)    


set_parameter("Vm", 5.65) # Initial guess for the VCA based on the Hartee self-consistency
new_model_instance()
update_mu()

# performing the vca loop over V_range
controlled_loop(run_vca, varia=["Vm"], loop_param="V", loop_range=(V_start, V_stop, V_step))
