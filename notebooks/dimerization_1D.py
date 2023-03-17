####################################################################################################
# Simple Hubbard model in d=1 with dimerization a cluster of 2 sites
####################################################################################################
import pyqcm

# declare a cluster model of 2 sites, named 'clus'
CM = pyqcm.cluster_model( 2)  

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0))) 

# define a lattice model named '1D_4' made of the cluster(s) clus and superlattice vector (2,0,0), same as the lattice vector
model = pyqcm.lattice_model('1D_4', clus, ((2,0,0),), ((2,0,0),))

# define a few operators in this model
model.interaction_operator('U', orbitals=(1,1))
model.interaction_operator('U', orbitals=(2,2))
model.hopping_operator('t', (1,0,0), -1, orbitals=(1,2))
model.hopping_operator('tp', (1,0,0), -1, orbitals=(2,1))


model.set_target_sectors(['R0:N2:S0']) # 2 particles, total spin 0
model.set_parameters("""
t=1.1
tp=0.9
U=1e-9
mu=0.5*U
""")


model.set_parameter("U", 1e-9) # To make sure correct U is used no matter execution order of cells
I = pyqcm.model_instance(model)
print(f"Ground state: {I.ground_state()}")
print(I.cluster_averages())
I.spectral_function(path="line") 


model.set_parameter('U', 4)
I = pyqcm.model_instance(model)
print(f"Ground state: {I.ground_state()}")
print(I.cluster_averages())
I.spectral_function(path="line") 
