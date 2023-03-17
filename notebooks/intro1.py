####################################################################################################
# Simple Hubbard model in d=1 with a cluster of 4 sites
####################################################################################################
import pyqcm

# declare a cluster model of 4 sites, named 'clus'
CM = pyqcm.cluster_model( 4)  

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (2,0,0),(3,0,0))) 

# define a lattice model named '1D_4' made of the cluster(s) clus and superlattice vector (4,0,0)
model = pyqcm.lattice_model('1D_4', clus, ((4,0,0),))

# define a few operators in this model
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)

# (optional) check the definition of an operator by drawing it on the screen
model.draw_operator('t')

#---------------------------------------------------------------------------------------------------
# declare the Hilbert space sectors into which the ground state will be searched
model.set_target_sectors('R0:N4:S0')

# declare the values of the parameters
model.set_parameters("""
t=1
U=2
mu=1
""")

# define an instance of the model with those values
I = pyqcm.model_instance(model)

# solve for the ground state and prints the ground state averages
print('ground_state sector:', I.ground_state())
I.cluster_averages(pr=True)

# draws the spectral function of the cluster model
I.cluster_spectral_function(wmax=6)

# draws the spectral function of the lattice model, for wavevectors along a line from -pi to pi
I.spectral_function(wmax=6, path='line')

# draws the density of states of the lattice model, from w = -6 to 6
I.plot_DoS(w = 6)
