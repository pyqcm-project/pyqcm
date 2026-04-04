import pyqcm

#--------------------------------------------------------------------------------
# defining the model
clus = pyqcm.cluster_model(2, n_bath=4, name='clus')
clus.new_operator('tb1', 'one-body', [(1, 3, -1.0), (2, 4, -1.0), (7, 9, -1.0), (8, 10, -1.0)])
clus.new_operator('tb2', 'one-body', [(1, 5, -1.0), (2, 6, -1.0), (7, 11, -1.0), (8, 12, -1.0)])
clus.new_operator('eb1', 'one-body', [(3, 3, 1.0), (4, 4, 1.0), (9, 9, 1.0), (10, 10, 1.0)])
clus.new_operator('eb2', 'one-body', [(5, 5, 1.0), (6, 6, 1.0), (11, 11, 1.0), (12, 12, 1.0)])

clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0)), pos=(0, 0, 0))

model = pyqcm.lattice_model('1D_2_4b', clus0, ((2, 0, 0),))

model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
#--------------------------------------------------------------------------------

sec = 'R0:N6:S0'
model.set_target_sectors([sec])
model.set_parameters("""
t=1
U = 4
mu = 0.5*U
eb1_1 = 0.5
eb2_1 = -0.5
tb1_1 = 0.5
tb2_1 = 0.5
""")


I = pyqcm.model_instance(model)
I.cluster_averages(pr=True)

ave = I.Green_function_average()
print('\naverages of c^\dagger_i c_j :\n\n', ave)

print('\naverage of t from GF= ', -2*(ave[0,1]))
print('average of mu from GF = ', I.Green_function_density())



