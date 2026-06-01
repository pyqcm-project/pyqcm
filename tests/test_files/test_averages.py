import pyqcm

#--------------------------------------------------------------------------------
# defining the model
clus = pyqcm.cluster_model(4, name='clus')
clus0 = pyqcm.cluster(clus, [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], pos=(0, 0, 0))
model = pyqcm.lattice_model('1D_4', clus0, ((4, 0, 0),))
model.current_dir = 'x'
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('tp', (2, 0, 0), -1)
model.hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)
model.hopping_operator('h', [0, 0, 0], 1, tau=0, sigma=3)
model.anomalous_operator('D', (1, 0, 0), 1)
model.anomalous_operator('S', (0, 0, 0), 1)
model.hopping_operator('H', (0, 0, 0), 1, tau=0, sigma=3)
model.hopping_operator('Hx', (0, 0, 0), 1, tau=0, sigma=1)
model.hopping_operator('tsi', (1, 0, 0), -1, sigma=3, tau=2)
model.hopping_operator('ti', (1, 0, 0), -1, tau=2)
#--------------------------------------------------------------------------------

sec = 'R0:S0'
model.set_target_sectors([sec])
model.set_parameters("""
ti=1
U = 4
mu = 1
D = 0.2
""")
I = pyqcm.model_instance(model)

print("\ntesting lattice averages (all operators, also with Potthoff functional and potential energy):\n")
print('Potthoff functional = ', I.Potthoff_functional())
print('potential energy = ', I.potential_energy())
I.averages(pr=True)

print("\ntesting averages from a Legendre grid:\n")
wr, weight = pyqcm.legendre_frequency_grid(1, 10, 10, 20, 10)
# wr, weight = pyqcm.regular_frequency_grid(20, 200, 10)
pyqcm.qcm.frequency_grid(wr, weight)
pyqcm.set_global_parameter('kgrid_side', 64)
I = pyqcm.model_instance(model)
print('Potthoff functional = ', I.Potthoff_functional())
print('potential energy = ', I.potential_energy())
I.averages(pr=True)

print("\ntesting averages from a regular (trapezoidal) grid:\n")
wr, weight = pyqcm.regular_frequency_grid(20, 200, 10)
pyqcm.qcm.frequency_grid(wr, weight)
I = pyqcm.model_instance(model)
print('Potthoff functional = ', I.Potthoff_functional())
print('potential energy = ', I.potential_energy())
I.averages(pr=True)

I.cluster_averages(pr=True)

print("\ntesting cluster averages from the wavefunction:\n")
I.cluster_averages(pr=True)

print("\ntesting cluster Green function averages:\n")
ave = I.Green_function_average()
print(r'\naverages of c^\dagger_i c_j :\n\n', ave)

if model.mixing == 4:
    ave = I.Green_function_average(spin_down=True)
    print(r'\naverages of c^\dagger_i c_j (spin down):\n\n', ave)

print('\naverage of t from GF= ', -(ave[0,1]+ave[1,2]+ave[2,3]).real)
print('average of mu from GF = ', I.Green_function_density())


# testing grid integration:
W, P = pyqcm.legendre_frequency_grid(1,10,5,10,5)

I = pyqcm.model_instance(model)
pyqcm.discrete_integration_grid(W, P)
pyqcm.set_wavevector_grid(16,16,1)
print('grille en k : 16,16,1')
I.averages(pr=True)

I = pyqcm.model_instance(model)
pyqcm.set_wavevector_grid(8,8,1)
print('grille en k : 8,8,1')
I.averages(pr=True)

I = pyqcm.model_instance(model)
pyqcm.set_wavevector_grid(4,4,1)
print('grille en k : 4,4,1')
I.averages(pr=True)

I = pyqcm.model_instance(model)
pyqcm.set_wavevector_grid(2,2,1)
print('grille en k : 2,2,1')
I.averages(pr=True)

I.GS_consistency()
