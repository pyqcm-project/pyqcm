import pyqcm
from model_1D_2_4b import model

# L=6
# model = model1D(L)
# model.set_target_sectors('R0:N{:d}:S0'.format(L))
model.set_target_sectors('R0:N6:S0')
model.set_parameters("""
    U=4
    mu=0.5*U
    t=1
    eb1_1 = 1
    eb2_1 = -1
    tb1_1 = 0.5
    tb2_1 = 0.5
""")

I = pyqcm.model_instance(model)
print(I.cluster_Green_function(0.1j, 0)[0,0])
I.cluster_averages(pr=True)
# I.write_impurity_problem(clus=0, file='impurity.tsv')
# I.write_Green_function(clus=0, file='GF.tsv')
I.write('test.out')
I = pyqcm.model_instance(model)
I.read('test.out')
print(I.cluster_Green_function(0.1j, 0)[0,0])
I.cluster_averages(pr=True)
