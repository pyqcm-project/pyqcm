####################################################################################################
# Simple one-dimensional Hubbard model with spin-orbit coupling
# We construct a one-dimensional Hubbard model with the following non-interacting Hamiltonian, here expressed in $k$-space:
# $$
#     H_0 = \sum_k \mathcal{H}_k  ~~~~~ \mathcal{H}_k = -2t\cos k\;c^\dagger_k c_k - 2m \sin k\;c^\dagger_k \sigma_x c_k
# $$
# where $c_k$ is a two-component annihilation operator in spin space : $(c_{k,\uparrow}, c_{k,\downarrow})$ and $\sigma_x$ is a Pauli matrix.
# Plot the non-interacting $\left(U=0\right)$ dispersion of this model as a function of $k$ in the interval $[-\pi,\pi]$ for $t=1$ and $m=0.2$ and check that it agrees with your analytical computation.
####################################################################################################
import pyqcm
import numpy as np
import matplotlib.pyplot as plt

# declare a cluster model of 4 sites, named 'clus'
CM = pyqcm.cluster_model( 4)

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (2,0,0),(3,0,0))) 

# define a lattice model named '1D_4' made of the cluster(s) clus and superlattice vector (4,0,0)
model = pyqcm.lattice_model('1D_4', clus, ((4,0,0),))

# define a few operators in this model
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)
# Defining the one-body Rashba coupling term
# tau=2 ---> imaginary contribution, ensures sin(k), sigma=1 ---> spin flip
model.hopping_operator("m", (1,0,0), -1, tau=2, sigma=1) 


# Note that spin is not conserved due to the Rashba term
model.set_target_sectors('R0:N4') 
model.set_parameters("""
    t=1
    m=0.4
""")

# Plotting the spectral function
I = pyqcm.model_instance(model)
I.spectral_function(wmax=2.5, path="line", nk=64, orb=1)


####################################################################################################
# Plotting the analytically-derived dispersion relation

# creating a grid of wavevectors in the Brillouin zone
k_grid = np.linspace(-np.pi, np.pi, 150)

t=1
m=0.4
# Defining the energy eigenvalues found analytically
def eig_1(k, t, m):
    return -2*(t*np.cos(k) + m*np.sin(k))

def eig_2(k, t, m):
    return -2*(t*np.cos(k) - m*np.sin(k))

# Plotting the analytic dispersion curve
plt.plot(k_grid, eig_1(k_grid, t, m), label="$\epsilon_1(k)$")
plt.plot(k_grid, eig_2(k_grid, t, m), label="$\epsilon_2(k)$")
plt.xlabel("k")
plt.ylabel("$\epsilon(k)$")
plt.xticks((-np.pi,0,np.pi), ("$-\pi$","$0$","$-\pi$"))
plt.legend()
plt.show()
