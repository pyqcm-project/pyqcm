import pyqcm
import model_1D_4_C2 as M

sec = 'R0:S0'
M.model.set_target_sectors([sec])
M.model.set_parameters("""
t=1
U = 4
mu = 1
D = 0.2
""")
I = pyqcm.model_instance(M.model)

print("\ntesting lattice averages (all operators, also with Potthoff functional and potential energy):\n")
print('Potthoff functional = ', I.Potthoff_functional())
print('potential energy = ', I.potential_energy())
I.averages(pr=True)

print("\ntesting cluster averages from the wavefunction:\n")
I.cluster_averages(pr=True)

print("\ntesting cluster Green function averages:\n")
ave = I.Green_function_average()
print('\naverages of c^\dagger_i c_j :\n\n', ave)

if M.model.mixing == 4:
    ave = I.Green_function_average(spin_down=True)
    print('\naverages of c^\dagger_i c_j (spin down):\n\n', ave)

print('\naverage of t from GF= ', -(ave[0,1]+ave[1,2]+ave[2,3]).real)
print('average of mu from GF = ', I.Green_function_density())

I.GS_consistency()
