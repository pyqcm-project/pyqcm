# Test file
# Goal : to test various plotting routines related to topology
#--------------------------------------------------------------------------------

import pyqcm
import numpy as np

#--------------------------------------------------------------------------------
# First, defining the model
clus = pyqcm.cluster_model(4, name='clus')
clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), pos=(0, 0, 0))
model = pyqcm.lattice_model('WSM_2x2', clus0, ((2, 0, 0), (0, 2, 0), (0, 0, 1)))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), 1, tau=1, sigma=1)
model.hopping_operator('t', (0, 1, 0), 1, tau=2, sigma=2)
model.hopping_operator('t', (0, 0, 1), 1, tau=2, sigma=3)
model.hopping_operator('t', (0, 0, 0), -0.765367, tau=0, sigma=1)
model.hopping_operator('m', (0, 1, 0), -0.5, tau=1, sigma=1)
model.hopping_operator('m', (0, 0, 1), -0.5, tau=1, sigma=1)
model.hopping_operator('m', (0, 0, 0), 2, tau=0, sigma=1)
model.set_target_sectors(['R0:N4'])


model.set_parameters("""
    U=4.0
    U_1=1.0*U
    m=1.5
    m_1=1.0*m
    mu=0.5*U
    mu_1=1.0*mu
    t=1.0
    t_1=1.0*t
""")
                     
I = pyqcm.model_instance(model)

k = np.array([0.695,0,-1e-8])

F = 'test_Berry_curvature.pdf'
pyqcm.banner('testing Berry_curvature()', c='#', skip=1); 
I.Berry_curvature(nk=20, k_perp=0.01, range=[0.5, 0, 0.5], plane='xz', file=F)

F = 'test_Berry_field_map.pdf'
pyqcm.banner('testing Berry_field_map()', c='#', skip=1); 
I.Berry_field_map(nk=20, k_perp=0.01, plane='xy', file=F)

pyqcm.banner('testing Berry_flux()', c='#', skip=1); 
print('Berry flux : ', I.Berry_flux(k, 0.1))

F = 'test_Berry_flux_map.pdf'
pyqcm.banner('testing Berry_flux_map()', c='#', skip=1); 
I.Berry_flux_map(nk=20, k_perp=-0.01, file=F)

pyqcm.banner('testing Chern_number()', c='#', skip=1); 
print('Chern number : ', I.Chern_number())

if I.model.dim == 3:
    pyqcm.banner('testing monopole()', c='#', skip=1); 
    print('monopole at ', k, ' : ', I.monopole(k, a=0.05, nk=20))

    F = 'test_monopole_map.pdf'
    pyqcm.banner('testing monopole_map()', c='#', skip=1); 
    I.monopole_map(nk=20, plane='xy', k_perp=0.0, file=F)

