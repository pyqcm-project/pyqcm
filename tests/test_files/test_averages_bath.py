import pyqcm
import model_1D_2_4b as model

sec = 'R0:N6:S0'
model.model.set_target_sectors([sec])
model.model.set_parameters("""
t=1
U = 4
mu = 0.5*U
eb1_1 = 0.5
eb2_1 = -0.5
tb1_1 = 0.5
tb2_1 = 0.5
""")

I = pyqcm.model_instance(model.model)
I.cluster_averages(pr=True)

ave = I.Green_function_average()
print('\naverages of c^\dagger_i c_j :\n\n', ave)

print('\naverage of t from GF= ', -2*(ave[0,1]))
print('average of mu from GF = ', I.Green_function_density())



