import pyqcm
from pyqcm.hartree import *
from pyqcm.loop import *
from pyqcm.vca import *
import model_1D_4

pyqcm.set_target_sectors(['R0:N4:S0'])
pyqcm.set_parameters("""
t=1
U=4
V=4
mu=10
V0m = 5
# V1m = -0.15
# cdw=0
# cdw_1=0.09
""")

V0m_H = hartree('V0m', 'V', 1.0, accur=1e-4, lattice=True)
# V1m_H = hartree('V1m', 'V', 1.0, accur=1e-4, lattice=True)
# plot_sef('V1m', np.arange(-0.5,0.5,0.02), hartree=[V1m_H]); exit()


def F():
    pyqcm.new_model_instance()
    pyqcm.averages()

Hartree(F, [V0m_H], maxiter=10, eps_algo=0); exit()

first = True
for v in np.arange(4, 3, -0.02):
    pyqcm.set_parameter('V', v)
    pyqcm.set_parameter('mu', 2+2*v)
    # Hartree(F, [V0m_H, V1m_H], maxiter=10, eps_algo=0)
    # vca(varia=['cdw_1', 'V0m', 'V1m'], steps=[0.01]*3, accur=[1e-4, 1e-2, 1e-2], max=[10]*3, hartree=[V0m_H, V1m_H], max_iter=100)
    # vca(varia=['cdw_1', 'V0m'], steps=[0.01]*2, accur=[1e-4, 1e-2], max=[10]*2, hartree=[V0m_H], max_iter=100)
    vca(varia=['cdw_1'], steps=[0.01], accur=[1e-4], max=[10], hartree=[V0m_H], max_iter=100, hartree_self_consistent=True)
    # f = open('cdw.tsv', 'a')
    # des, val = pyqcm.properties()
    # if first:
    #     f.write(des + '\n')
    #     first = False
    # f.write(val + '\n')
    # f.close()


