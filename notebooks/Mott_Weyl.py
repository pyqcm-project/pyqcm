####################################################################################################
# Mott transition in a Weyl semimetal
# A simple model of a Weyl semimetal is defined by the following Hamiltonian:
# 
# $$ H = H_0 + U\sum_{i} n_{i\uparrow}n_{i_\downarrow} $$
# 
# where
# 
# $$ H_0 = \sum_{\mathbf{k}} c_\mathbf{k}^\dagger \mathcal{H}_\mathbf{k} c_\mathbf{k} \qquad\qquad
# c_\mathbf{k}=(c_{\mathbf{k}\uparrow}, c_{\mathbf{k}\downarrow}) $$
# 
# with the $k$-dependent matrix 
# 
# $$ \mathcal{H}_\mathbf{k} =  \left\{ 2t(\cos k_x - \cos k_0) + m(2-\cos k_y -\cos k_z)\right\}\sigma_x
# + 2t\sin k_y \sigma_y + 2t\sin k_z \sigma_z $$
# 
# where $\sigma_{x,y,z}$ are the Pauli matrices.
# We will set $m=\frac32$ and $k_0=3\pi/8$.
# See Witczak-Krempa *et al.*, Phys. Rev. Lett. **113**, 136402 (2014).
# 
# 1. Construct this model, with the operators $t$, $m$ and $U$.
# 2. Plot the Fermi surface at $U=0$ using `spectral.mdc()` and identify the Weyl nodes along the $k_x$ axis.
# 3. Using CPT, show how the position of the Weyl node changes for $U=6$ compared with $U=0$.
####################################################################################################
import pyqcm
import numpy as np

# declare a cluster model of 8 sites, named 'clus'
CM = pyqcm.cluster_model( 8)

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus = pyqcm.cluster(CM, ((0,0,0), (1,0,0), (0,1,0), (1,1,0), (0,0,1), (1,0,1), (0,1,1), (1,1,1))) 

# define a lattice model named '2x2' made of the cluster(s) clus and superlattice vectors (2,0,0) & (0,2,0)
model = pyqcm.lattice_model('2x2x2', clus, ((2,0,0), (0,2,0), (0,0,2)))

# define a few operators in this model
model.interaction_operator('U')
# Assembling the parts of operator `t`
model.hopping_operator("t", (1,0,0), 1, tau=1, sigma=1)
model.hopping_operator("t", (0,0,0), -np.cos(3*np.pi/8), tau=0, sigma=1)
model.hopping_operator("t", (0,1,0), 1, tau=2, sigma=2)
model.hopping_operator("t", (0,0,1), 1, tau=2, sigma=3)

# Assembling the parts of operator `m`
model.hopping_operator("m", (0,0,0), 1, tau=0, sigma=1)
model.hopping_operator("m", (0,1,0), -0.5, tau=1, sigma=1)
model.hopping_operator("m", (0,0,1), -0.5, tau=1, sigma=1)


# Defining a model at half-filling 
model.set_target_sectors(["R0:N8"])
model.set_parameters("""
    U = 4
    mu = 0.5*U
    t=1
    m=1.5
""")


print("plotting different Fermi surface plots at U=0...")
model.set_parameter("U", 1e-9)
I = pyqcm.model_instance(model)
I.mdc(plane="xy")
I.mdc(plane="yz")

print("plotting different Fermi surface plots at U=6...")
model.set_parameter("U", 6)
I = pyqcm.model_instance(model)
I.mdc(plane="xy")
I.mdc(plane="yz")


print("plotting the spectral function along a high-symmetry path at U=0...")
model.set_parameter("U", 1e-9)
I = pyqcm.model_instance(model)
I.spectral_function(path="cubic")

print("plotting the spectral function along a high-symmetry path at U=0...")
model.set_parameter("U", 6)
I = pyqcm.model_instance(model)
I.spectral_function(path="cubic")

print("plotting the Berry curvature...")
I.Berry_curvature(nk=100, eta=0.0, period='G', range=None, label=0, orb=1, subdivide=False, plane='xy', k_perp=0.1, file=None, plt_ax=None)

print("computing the flux of the Berry curvature through loops located at different values of kx...")
for kx in np.arange(0,1,0.1):
    print('kx/pi = {:.2f} : Berry flux = {:.2f}'.format(kx, 1e-7+I.Berry_flux([kx,0,0], 0.05, nk=10, plane='xy')))

print("Mapping the Berry flux...")
I.Berry_flux_map(nk=50, plane='z', dir='z', k_perp=0.01, label=0, npoints=8, radius=None, file=None, plt_ax=None)

print("Mapping the Berry field lines on the xy plane, then on the xz plane...")
I.Berry_field_map(nk=80, nsides=4, plane='z', k_perp=0.01, label=0, file=None, plt_ax=None)
I.Berry_field_map(nk=40, nsides=4, plane='y', k_perp=0.01, label=0, file=None, plt_ax=None)

print("Mapping the Berry monopoles at various value of kx...")
for kx in np.arange(0,1,0.025):
    print('kx/pi = {:.2f} : monopole = {:.4f}'.format(kx, 1e-7+I.monopole([kx,0,0], a=0.04, nk=30, subdivide=True)))

print("Mapping the Berry monopoles...")
I.monopole_map(k_perp=0.01)

