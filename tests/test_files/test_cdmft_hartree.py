import pyqcm
from pyqcm.cdmft import CDMFT
from model_1D_2_4b import model

model.set_target_sectors('R0:N6:S0')
model.set_parameters("""
    U=4
    V=1
    Vm = 1
    mu=2
    t=1
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
""")

Vm_H = pyqcm.hartree(model, 'Vm', 'V', model.Vm_eig, lattice=False) 
# Vm_H = pyqcm.hartree(model, 'Vm', 'V', model.Vm_eig, lattice=True) 

def run_cdmft():
    U = model.parameters()['U']
    V = model.parameters()['V']
    model.set_parameter('mu', 0.5*U+2*V)
    X = CDMFT(model, varia=('tb1_1', 'eb1_1'), hartree=Vm_H) 
    return X.I

# Looping over values of U
model.controlled_loop(
    task=run_cdmft, 
    varia=('tb1_1', 'eb1_1', 'Vm'),
    loop_param='U', 
    loop_range=(2, 4.1, 0.5)
)