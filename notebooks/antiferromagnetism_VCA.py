####################################################################################################
"""
VCA for antiferromagnetism
Find the VCA solution for Néel antiferromagnetism in the nearest-neighbor, particle-hole symmetric Hubbard model at half-filling, using **two** $3\times 3$ clusters embedded in a two-dimensional square lattice.
Use only one variational parameter: the Weiss field $M$, defined as the coefficient of the antiferromagnetic operator. Plot the order parameter $\langle M\rangle$ as a function of $U$, from $U=10$ to $U=0$, in steps of $\Delta U=0.5$.
"""
####################################################################################################
import pyqcm
import pyqcm.vca as vca
import numpy as np

pyqcm.set_global_parameter('accur_SEF', 1e-4)

# declare a cluster model of 9 sites, named 'clus'
CM = pyqcm.cluster_model(9)  

sites = []
for j in range(3):
    for i in range(3):
        sites.append((i,j,0))

# defining a hopping operator that only hops around the perimeter of the clusters
ns = 9
CM.new_operator('tperim', 'one-body',[
    (1, 2, -1.0), 
    (2, 3, -1.0),
    (3, 6, -1.0), 
    (6, 9, -1.0),
    (8, 9, -1.0), 
    (7, 8, -1.0),
    (4, 7, -1.0), 
    (1, 4, -1.0),
    (1+ns, 2+ns, -1.0), 
    (2+ns, 3+ns, -1.0),
    (3+ns, 6+ns, -1.0), 
    (6+ns, 9+ns, -1.0),
    (8+ns, 9+ns, -1.0), 
    (7+ns, 8+ns, -1.0),
    (4+ns, 7+ns, -1.0), 
    (1+ns, 4+ns, -1.0)
])

# define a physical cluster based on that model, with base position (0,0,0) and site positions
clus1 = pyqcm.cluster(CM, sites) 
clus2 = pyqcm.cluster(CM, sites, (3,0,0)) # second cluster, offset from the first

# define a lattice model named '1D_4' made of the cluster(s) clus and superlattice vector (4,0,0)
model = pyqcm.lattice_model('model_3x3_2C', (clus1, clus2), ((6,0,0),(1,3,0)))

# define a few operators in this model
model.interaction_operator('U')
model.hopping_operator('t', (1,0,0), -1)
model.hopping_operator('t', (0,1,0), -1)
model.density_wave('M', 'Z', (1,1,0)) # Spin density wave at Q = (pi,pi)

# model.draw_operator('U', values=True); exit()

# model.draw_operator('t')
# model.draw_cluster_operator(model.clus[0], 'tperim')

# Setting target sectors with S=1 and S=-1 such that S_total = 0
model.set_target_sectors(['R0:N9:S-1/R0:N9:S1','R0:N9:S-1/R0:N9:S1'])
model.set_parameters("""
t = 1
U = 8
mu = 0.5*U
M = 0
M_1 = 0.1
M_2 = 1*M_1
# t_1 = 1
# t_2 = 1*t_1
# tperim_1 = 1e-9
# tperim_2 = 1*tperim_1
""")


#---------------------------------------------------------------------------------------------------
# Estimating the starting value of `M_1` 
# Here we make a rough plot of the self-energy functional as a function of `M_1` to find a good starting point for the VCA procedure.

# model.set_parameter("U", 8)
I = pyqcm.model_instance(model)
print(I.ground_state())
# pyqcm.vca.plot_sef(model, 'M_1', np.arange(0.05, 0.151, 0.01)); exit()


#---------------------------------------------------------------------------------------------------
# Controlled VCA loop over $U$
# Here we loop over U in the range from 10 to 0 and perform the VCA with M_1 as a variational parameter. 
# Looping downwards over U allows for easier convergence on a solution
# We use the function controlled_loop() to benefit from predictors.

model.set_parameter("M_1", 0.11) # initial guess for M_1 based on the sef
I = pyqcm.model_instance(model)

def run_vca():
    V = vca.VCA(model, varia='M_1', steps=0.005, accur=2e-3, max=10, method='altNR')
    return V.I

# controlled_loop performs successive VCAs with starting value prediction
model.controlled_loop(run_vca, varia=["M_1"], loop_param="U", loop_range=(10, -0.1, -0.5)) # loops from 10 to 0 in increments of 0.5


#---------------------------------------------------------------------------------------------------
# VCA for U=8 with M_1 and t_1 as variational parameters

model.set_parameter("U", 8)
V = vca.VCA(model, varia=('M_1', 't_1'), start=(0.17, 1.15), steps=(0.005, 0.005), accur=(2e-3, 2e-3), max=(10, 10)) 


#---------------------------------------------------------------------------------------------------
# VCA for U=8 with M_1, t_1 and tperim_1 as variational parameters

V = vca.VCA(model, varia=('M_1', 't_1', 'tperim_1'), start=(0.17, 1.15, 0.04), steps=(0.005, 0.005, 0.005), accur=(2e-3, 2e-3, 2e-3), max=(10, 10, 10))

exit()
#---------------------------------------------------------------------------------------------------
# Getting data from prerun VCAs
d_M = np.genfromtxt("./example_data/EX2_data_M.tsv", names=True)
d_M_t = np.genfromtxt("./example_data/EX2_data_M_t.tsv", names=True)
d_M_t_tperim = np.genfromtxt("./example_data/EX2_data_M_t_tperim.tsv", names=True)


# #### Plotting $|\langle M \rangle|$ as a function of $U$ for all three simulations

# In[ ]:


fig, ax = plt.subplots()

ax.plot(d_M["U"], np.abs(d_M["ave_M"]), "s", markersize=3, label="M_1")
ax.plot(d_M_t["U"], np.abs(d_M_t["ave_M"]), "s", markersize=3, label="M_1 & t_1")
ax.plot(d_M_t_tperim["U"], np.abs(d_M_t_tperim["ave_M"]), "s", markersize=3, label="M_1, t_1 & tperim_1")

ax.legend(loc="upper left")

ax.set_xlabel("U")
ax.set_ylabel("$|\langle M\\rangle|$")

subax = fig.add_axes([0.5, 0.25, 0.35, 0.35])

subax.plot(d_M["U"][14:17], np.abs(d_M["ave_M"][14:17]), "s", markersize=3)
subax.plot(d_M_t["U"], np.abs(d_M_t["ave_M"]), "s", markersize=3)
subax.plot(d_M_t_tperim["U"], np.abs(d_M_t_tperim["ave_M"]), "s", markersize=3)

subax.set_xlabel("U")
subax.set_ylabel("$|\langle M\\rangle|$")

subax.set_xticks((d_M["U"][14:17]))

fig.show()


# #### Plotting the Weiss field as a function of $U$ for all three simulations

# In[ ]:


fig, ax = plt.subplots()

ax.plot(d_M["U"], d_M["M_1"], "s", markersize=3, label="M_1")
ax.plot(d_M_t["U"], d_M_t["M_1"], "s", markersize=3, label="M_1 & t_1")
ax.plot(d_M_t_tperim["U"], d_M_t_tperim["M_1"], "s", markersize=3, label="M_1, t_1 & tperim_1")

ax.legend(loc="lower right")

ax.set_xlabel("U")
ax.set_ylabel("|M_1|")

fig.show()


# ### Interpretation
# 
# In the first figure, we can see how $|\langle M \rangle|$ seems to approach 1 relatively quickly as a function of $U$. On the other hand, $|M\_1|$ seems to peak around $U\approx 5$ (fig.2). This means that as $U$ rises above 5, a rise in order parameter actually requires a *diminishing* Weiss field due to the rising interaction strength.
# 
# In both figures, adding in t_1 to the variational parameters seems to cause a rise in the values. However, adding in tperim_1 only causes a very minute increase. This makes sense since there doesn't seem to be a good reason for hopping exclusively around the perimeter of the cluster as described in the definition of the given operator.
