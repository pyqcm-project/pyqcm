# Test file
# Goal : to test the VCA procedure
#--------------------------------------------------------------------------------
import pyqcm
import pyqcm.vca as V
import numpy as np

#--------------------------------------------------------------------------------
# model definition
clus = pyqcm.cluster_model(4, name='clus', generators=[[4, 3, 2, 1]])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('2x2_C2', clus0, [(2, 0, 0), (0, 2, 0)])

model.interaction_operator('U')
model.interaction_operator('V', link=(1, 0, 0), amplitude=1)
model.interaction_operator('V', link=(0, 1, 0), amplitude=1)
model.hopping_operator('Vm', [0, 0, 0], 1, tau=0)
model.interaction_operator('J', link=(1, 0, 0), type='Hund')
model.hopping_operator('t', (0, 1, 0), -1)
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('t2', (1, 1, 0), -1)
model.hopping_operator('t2', (-1, 1, 0), -1)
model.hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
model.density_wave('M', 'Z', (1, 1, 0))
model.hopping_operator('t3', (2, 0, 0), -1)
model.hopping_operator('t3', (0, 2, 0), -1)
model.anomalous_operator('D', (1, 0, 0), 1)
model.anomalous_operator('D', (0, 1, 0), -1)
model.set_target_sectors(['R0:N4:S0'])

#--------------------------------------------------------------------------------
model.set_parameters("""
    M=0.0
    M_1=0.15
    U=8.0
    U_1=1.0*U
    mu=4.0
    mu_1=4.0
    t=1.0
    t2=1e-09
    t2_1=1.0*t2
    t_1=1.0
""")
#--------------------------------------------------------------------------------

F = 'test_vca.pdf'

# f, w = pyqcm.legendre_frequency_grid(1, 10, 10, 10, 10)
# pyqcm.discrete_integration_grid(f, w)


pyqcm.banner('testing vca()', c='#', skip=1)
vca = V.VCA(model, varia=['M_1', 't_1'], accur=[0.005, 0.005], steps=[5e-5, 5e-5], max=[10,10], method='NR')

pyqcm.banner('testing vca() with Nelder-Mead', c='#', skip=1)
vca = V.VCA(model, varia=['M_1', 't_1'], accur=(0.005, 0.005), steps=(5e-5, 5e-5), max=(10,10), method='Nelder-Mead')

pyqcm.banner('testing vca() with Newton-Raphson', c='#', skip=1)
vca = V.VCA(model, varia=['M_1', 't_1'], accur=[0.005, 0.005], steps=[5e-5, 5e-5], max=[10,10], method='NR')

pyqcm.banner('testing vca() with explicit starting values', c='#', skip=1)
vca = V.VCA(model, varia=['M_1', 't_1'], start=[0.1, 1.3], accur=[0.005, 0.005], steps=[5e-5, 5e-5], max=[10,10], method='SYMR1')

pyqcm.banner('testing vca() with minimax', c='#', skip=1)
vca = V.VCA(model, varia=['M_1', 'mu_1'], accur=[0.005, 0.005], steps=[5e-5, 5e-5], max=[10,10], method='minimax', var_max_start=1)

pyqcm.banner('testing plot_sef()', c='#', skip=1)
V.plot_sef(model, 'M_1', np.arange(1e-9, 0.3, 0.02), accur_SEF=1e-4, show=False)

pyqcm.banner('testing plot_GS_energy()', c='#', skip=1)
V.plot_GS_energy(model, 'M_1', np.arange(1e-9, 0.3, 0.02), file="GS_energy.pdf")
