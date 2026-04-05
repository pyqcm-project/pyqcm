# Test file
# Goal : to test the existence of several model instances that are not deleted
#--------------------------------------------------------------------------------
import pyqcm
import matplotlib.pyplot as plt


clus = pyqcm.cluster_model(4, name='clus', generators=[[4, 3, 2, 1]])
clus0 = pyqcm.cluster(clus, [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], pos=(0, 0, 0))
model = pyqcm.lattice_model('1D_4', clus0, ((4, 0, 0),))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)


sec = 'R0:N4:S0'
model.set_target_sectors([sec])
model.set_parameters("""
t=1
U = 4
mu = 0.5*U
""")

n = 4
U = [1e-9, 2, 4, 6]
I = []
for i,u in enumerate(U):
    model.set_parameter('U', u)
    I.append(pyqcm.model_instance(model))

for i,u in enumerate(U):
    I[i].spectral_function(wmax=6, path='line', plt_ax=plt.gca(), file=f"test_instances_U{u}.pdf")
