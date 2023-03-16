import pyqcm
import model_1D_2_4b as M

sec = 'R0:N4:S0'
M.model.set_target_sectors([sec])
M.model.set_parameters("""
t=1
U = 4
mu = 0.5*U
eb1_1 = 0.5
eb2_1 = -0.5
tb1_1 = 0.5
tb2_1 = 0.5
""")

I = pyqcm.model_instance(M.model)
I.cluster_spectral_function(wmax=6, opt='hyb', file="cluster_spectral_hybridization.pdf")

# print(pyqcm.hybridization_Lehmann())

