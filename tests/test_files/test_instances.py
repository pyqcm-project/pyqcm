import pyqcm
import pyqcm._spectral
import matplotlib.pyplot as plt
import model_1D_4

sec = 'R0:N4:S0'
pyqcm.set_target_sectors([sec])
pyqcm.set_parameters("""
t=1
U = 4
mu = 0.5*U
""")

n = 4
U = [1e-9, 2, 4, 6]
for i,u in enumerate(U):
    pyqcm.set_parameter('U', u)
    I = pyqcm.new_model_instance(i, record=True)
    I.print('record_test_{:d}.py'.format(i))

for i,u in enumerate(U):
    # plt.subplot(2,2,i+1)
    pyqcm._spectral.spectral_function(wmax=6, path='line', label=i, plt_ax=plt.gca(), file=f"test_instances_U{u}.pdf")
    # plt.title('U = {:1.2f}, instance {:d}'.format(u, i))

# plt.tight_layout()
# plt.show()

##################################################################
# TEST UNITAIRE

