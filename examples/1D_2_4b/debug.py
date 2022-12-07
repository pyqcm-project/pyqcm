from pyqcm import *
from pyqcm.cdmft import *
from pyqcm.loop import *

import model_1D_2_4b

np.set_printoptions(precision=4, linewidth=512, suppress=True)

sec = 'R0:N6:S0'
set_target_sectors([sec])

set_parameters("""
    U=4
    mu=2
    t=1
    tb1_1=0.5
    tb2_1=0.5
    eb1_1=1
    eb2_1=-1
""")

varia=['tb1_1', 'tb2_1', 'eb1_1', 'eb2_1']

min_iter_E0 = 4
def F():
    cdmft(varia=varia, accur=1e-14, miniter=10, accur_E0=1e-4, accur_hybrid=1e-14, accur_dist=1e-14)

def G():
    cdmft(varia)
    d = dos(0.01j)
    write_summary('dos.tsv', suppl_descr='DoS(0.01i)\t', suppl_values='{:.6g}\t'.format(d[0]))


# F(); exit()

controlled_loop(F, varia, 'mu', (2, 1, -0.05))