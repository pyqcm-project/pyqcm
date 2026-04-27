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

time_needed = {}
# Defining a function that will run a cdmft procedure within controlled_loop()
def run_cdmft(Z):
    try:
        alpha = X.alpha
    except:
        alpha = 0.0
    t1 = timeit.default_timer()
    X = CDMFT(model, varia=varia, grid=frequency_grid('legendre', (1, 10, 10, 10, 10)), method=Z, accur=accur, convergence=convergence, converge_with_stdev=False, miniter=1, maxiter=32, depth=1, iteration='broyden', alpha=alpha)
    t2 = timeit.default_timer()
    time_needed[Z] = t2-t1
    return X.I

# loop over methods:

methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'NELDERMEAD', 'COBYLA', 'BOBYQA', 'PRAXIS', 'SUBPLEX']

print('Time needed by the various methods:\n')
for Z in methods:
    print('\n\nMETHOD = ', Z, '\n')
    model.set_parameter('eb1_1', 1.0)
    model.set_parameter('eb2_1', -1.1)
    model.set_parameter('tb1_1', 0.5)
    model.set_parameter('tb2_1', 0.4)

    try:
    	run_cdmft(Z)
    except: 
        print('Failure with method ', Z)

for x in time_needed:
    print(x, ' : ', time_needed[x])

for x in methods:
    if x not in time_needed:
        print('Method ', x, ' did not succeed under 32 iterations.')