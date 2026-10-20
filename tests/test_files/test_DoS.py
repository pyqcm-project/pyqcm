# Test file
# Goal : to test computing and plotting the density of states
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm


from model_1D import model1D

L = 8 # length of 1D system
model = model1D(L, sym=True)

pyqcm.set_global_parameter('GF_method', 'L')

model.set_target_sectors('N{:d}:S0'.format(L))

# Simulation parameters
model.set_parameters("""
    U=4
    mu=2
    t=1
""")

I = pyqcm.model_instance(model)
I.plot_DoS(w=6, eta=0.1, file='DoS.pdf')

import matplotlib.pyplot as plt
I.plot_DoS(w=6, eta=0.1, file='DoS.pdf', plt_ax = plt.gca())
plt.title('Alternate title')
plt.savefig('Dos_alt.pdf')