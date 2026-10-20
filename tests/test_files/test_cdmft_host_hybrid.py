# Test file
# Goal : to test CDMFT on a 1D cluster with various minimization methods
#--------------------------------------------------------------------------------
import pyqcm
import timeit
from pyqcm.cdmft import CDMFT, frequency_grid

#--------------------------------------------------------------------------------
# First, defining the model (a 2-cluster model for graphene with bath)

clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('Graphene_2', clus0, ((1, -1, 0), (2, 1, 0)), lattice=((1, -1, 0), (2, 1, 0)))

model.interaction_operator('U', orbitals=(1, 1))
model.interaction_operator('U', orbitals=(2, 2))
model.hopping_operator('t', (1, 0, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (0, 1, 0), -1, orbitals=(1, 2))
model.hopping_operator('t', (-1, -1, 0), -1, orbitals=(1, 2))
model.hopping_operator('M', (0, 0, 0), 1, orbitals=(1, 1), tau=0, sigma=3)
model.hopping_operator('M', (0, 0, 0), -1, orbitals=(2, 2), tau=0, sigma=3)
model.hopping_operator('Mx', (0, 0, 0), 1, orbitals=(1, 1), tau=0, sigma=1)
model.hopping_operator('Mx', (0, 0, 0), -1, orbitals=(2, 2), tau=0, sigma=1)
model.hopping_operator('cdw', (0, 0, 0), 1, orbitals=(1, 1), tau=0)
model.hopping_operator('cdw', (0, 0, 0), -1, orbitals=(2, 2), tau=0)

#--------------------------------------------------------------------------------


# Imposing half-filling at 6 particles in cluster + bath sites and setting total spin to 0
model.set_target_sectors(['R0:N6:S0'])

# Simulation parameters
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=0.4
    eb1_1=1.0
    eb2_1=-1.1
""")

varia = ['eb1_1', 'eb2_1', 'tb1_1', 'tb2_1']

convergence=('parameters', 'hybridization', 'self-energy'); accur = (5e-6, 1e-4, 1e-4)




t1 = timeit.default_timer()
X = CDMFT(model, varia=varia, grid=frequency_grid('legendre', (1, 10, 10, 10, 10)), method='trf', accur=1e-3, convergence='self-energy', iteration='broyden')

import numpy as np
import matplotlib.pyplot as plt
w = np.arange(0.1,10.001,0.1)


fig,ax = plt.subplots(3,1)
fig.set_size_inches(12/2.54, 25/2.54)
X.I.plot_host_hybrid(w, e=None, plt_ax = ax[0])
ax[0].set_title('Frobenius norm')
ax[0].grid()
ax[0].set_xlim(w[0], w[-1])
X.I.plot_host_hybrid(w, e=(0,0), plt_ax = ax[1])
ax[1].set_title('element (0,0)')
ax[1].grid()
ax[1].set_xlim(w[0], w[-1])
X.I.plot_host_hybrid(w, e=(1,0), plt_ax = ax[2])
ax[2].set_title('element (1,0)')
ax[2].grid()
ax[2].set_xlim(w[0], w[-1])
plt.savefig('host_hybrid.pdf')