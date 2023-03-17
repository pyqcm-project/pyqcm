####################################################################################################
# CDMFT and the Mott transition on the graphene lattice
# We construct a model for the Hubbard model on the graphene lattice, based on a two-site cluster with a bath of 4 sites.
# We check that the correct dispersion relation is found at $U=0$ by plotting the spectral function.
# At half filling, we find the CDMFT solutions (normal state) as a function of $U$, from $U=10$ to $U=0$ in steps of $\Delta U=0.2$. We stay at half-filling, using particle-hole symmetry.
####################################################################################################
import pyqcm
import pyqcm.cdmft as cdmft
import numpy as np
import os

# declare a cluster model of 6 sites, named 'clus'
ns = 2 # number of physical sites
nb = 4 # number of bath sites
no = ns+nb # total number of sites
CM = pyqcm.cluster_model( ns, nb)

# Defining the bath hopping operators
CM.new_operator('tb1', 'one-body', [
    (1, 3, -1.0),
    (2, 4, -1.0),
    (1+no, 3+no, -1.0),
    (2+no, 4+no, -1.0)
]) # note that the last two entries pertain to the SPIN DOWN part of the operator

CM.new_operator('tb2', 'one-body', [
    (1, 5, -1.0),
    (2, 6, -1.0),
    (1+no, 5+no, -1.0),
    (2+no, 6+no, -1.0)
])

# Defining the 'orbital energy' of the baths
CM.new_operator('eb1', 'one-body', [
    (3, 3, 1.0),
    (4, 4, 1.0),
    (3+no, 3+no, 1.0),
    (4+no, 4+no, 1.0)
])

CM.new_operator('eb2', 'one-body', [
    (5, 5, 1.0),
    (6, 6, 1.0),
    (5+no, 5+no, 1.0),
    (6+no, 6+no, 1.0)
])

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0))) 

# define a lattice model named '2x2' made of the cluster(s) clus and superlattice vectors (3,0,0) & (0,3,0)
# the lattice vectors are (1,-1,0) and (2,1,0), different from the default ones (1,0,0) & (0,1,0)
model = pyqcm.lattice_model('graphene2_4b', clus, ((1,-1,0), (2,1,0)), ((1,-1,0), (2,1,0)))

# we define a physical basis (expression of the working basis in terms of physical coordinates), for plotting purposes
model.set_basis(((1,0,0), (-0.5,np.sqrt(3)/2,0)))


# define a few operators in this model. Since this is a 2-orbital model, one needs to specify the relevant orbitals.
model.interaction_operator('U', orbitals=(1,1))
model.interaction_operator('U', orbitals=(2,2))
model.hopping_operator('t', (1,0,0), -1, orbitals=(1,2))
model.hopping_operator('t', (0,1,0), -1, orbitals=(1,2))
model.hopping_operator('t', (-1,-1,0), -1, orbitals=(1,2))


model.set_target_sectors(['R0:N6:S0'])
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    tb1_1=0.5
    tb2_1=1*tb1_1
    eb1_1=1.0
    eb2_1=-1.0*eb1_1
""")

# Setting a trivially small interaction value and plotting the spectral function
model.set_parameter("U", 1e-9)
I = pyqcm.model_instance(model)
I.spectral_function(path="graphene", orb=1) # Using the built-in graphene path is recommended here

# Setting up the range of values of U over which to loop the CDMFT simulation 
U_start = 0.2 # starting at U=0.2 because the nearly non-interacting case is too long to calculate and useless
U_stop = 10.1
U_step = 0.2

# Removes the default output file for display purposes
if os.path.isfile("./cdmft.tsv"):
    os.remove("./cdmft.tsv")

# Defining a function that will run a cdmft procedure within controlled_loop()
varia=('tb1_1', 'eb1_1')
def run_cdmft():
    U = model.parameters()['U']
    cdmft_simulation = cdmft.CDMFT(model, varia=varia, wc=10, grid_type='self', beta=50)
    return cdmft_simulation.I

# Looping over values of U
model.controlled_loop(
    task=run_cdmft, 
    varia=varia, 
    loop_param='U', 
    loop_range=(U_start, U_stop, U_step)
)


# plotting the results
import matplotlib.pyplot as plt

cdmft_data = np.genfromtxt("cdmft.tsv", names=True)

# plotting both orbital energies and site<-->bath hopping operator amplitude
ax1 = plt.subplot(211)
ax1.plot(cdmft_data['U'], cdmft_data['eb1_1'], "o", markersize=2, label="eb1_1")
ax1.set_ylabel("$\epsilon$")
ax1.legend()

ax2 = plt.subplot(212, sharex=ax1)
ax2.plot(cdmft_data['U'], cdmft_data['tb1_1'], "o", markersize=2, label="tb1_1")
ax2.set_xlabel("U")
ax2.set_ylabel("t")
ax2.legend()

plt.show()


index_of_U = np.where(cdmft_data['U'] == 4)[0][0] # Finding the index at which U=4
print(index_of_U)
model.set_params_from_file('cdmft.tsv', n=index_of_U)
I = pyqcm.model_instance(model)
I.spectral_function(path="graphene", orb=1) 

index_of_U = np.where(cdmft_data['U'] == 5)[0][0] # Finding the index at which U=4
model.set_params_from_file('cdmft.tsv', n=index_of_U)
I = pyqcm.model_instance(model)
I.spectral_function(path="graphene", orb=1) 

index_of_U = np.where(cdmft_data['U'] == 6)[0][0] # Finding the index at which U=4
model.set_params_from_file('cdmft.tsv', n=index_of_U)
I = pyqcm.model_instance(model)
I.spectral_function(path="graphene", orb=1) 
