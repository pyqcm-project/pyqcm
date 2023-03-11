import pyqcm
import pyqcm._spectral as SP
import model_1D_2_4b

import pyqcm.qcm as qcm


sec = 'R0:N4:S0'
pyqcm.set_target_sectors([sec])
pyqcm.set_parameters("""
t=1
U = 4
mu = 0.5*U
eb1_1 = 0.5
eb2_1 = -0.5
tb1_1 = 0.5
tb2_1 = 0.5
""")

pyqcm.new_model_instance()


SP.cluster_spectral_function(wmax=6, opt='hyb', file="cluster_spectral_hybridization.pdf")

# print(pyqcm.hybridization_Lehmann())

