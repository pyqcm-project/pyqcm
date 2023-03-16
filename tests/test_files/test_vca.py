import pyqcm
import pyqcm.vca as V
import matplotlib.pyplot as plt
import numpy as np

# ax = plt.gca()
ax = None
F = None

# print(dir(pyqcm.vca)); exit()
#-----------------------------------------------------------------

# plt.show()
##################################################################
# TEST UNITAIRE

def test_vca():
    x = None
    import model_2x2_C2_vca as M

    F = 'test_vca.pdf'

    pyqcm.banner('testing vca()', c='#', skip=1)
    vca = V.VCA(M.model, varia=['M_1', 't_1'], accur=[5e-4, 5e-4], steps=[5e-5, 5e-5], max=[10,10], method='NR')

    pyqcm.banner('testing vca() with Newton-Raphson', c='#', skip=1)
    vca = V.VCA(M.model, varia=['M_1', 't_1'], accur=[5e-4, 5e-4], steps=[5e-5, 5e-5], max=[10,10], method='NR')

    pyqcm.banner('testing vca() with explicit starting values', c='#', skip=1)
    vca = V.VCA(M.model, varia=['M_1', 't_1'], start=[0.1, 1.1], accur=[5e-4, 5e-4], steps=[5e-5, 5e-5], max=[10,10], method='SYMR1')

    pyqcm.banner('testing vca() with minimax', c='#', skip=1)
    vca = V.VCA(M.model, varia=['M_1', 'mu_1'], accur=[5e-4, 5e-4], steps=[5e-5, 5e-5], max=[10,10], method='minimax', var_max_start=1)

    pyqcm.banner('testing plot_sef()', c='#', skip=1)
    V.plot_sef(M.model, 'M_1', np.arange(1e-9, 0.3, 0.02), accur_SEF=1e-4, show=False)

    pyqcm.banner('testing plot_GS_energy()', c='#', skip=1)
    V.plot_GS_energy(M.model, 'M_1', np.arange(1e-9, 0.3, 0.02), file="GS_energy.pdf")

    # pyqcm.banner('testing transition_line()', c='#', skip=1) ########## Temporary fix ##########
    # M.model.set_parameter('U', 1)
    # M.model.set_parameter('t2', -0.5)
    # V.transition_line(M.model, 'M_1', 'U', np.arange(1, 8.01, 0.25), 'mu', [0.8, 1], delta=1, verb=True)


test_vca()

# transition
# transition_line
# vca_min