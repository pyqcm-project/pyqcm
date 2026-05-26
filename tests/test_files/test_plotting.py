# Test file
# Goal : to test the different plotting functions of _spectral.py, especially when providing matplotlib axis objects as arguments
#--------------------------------------------------------------------------------
import numpy as np
import pyqcm

#--------------------------------------------------------------------------------
CM = pyqcm.cluster_model(4, generators=[[4, 3, 2, 1]])
clus = pyqcm.cluster(CM, ((0, 0, 0), ( 1, 0, 0), ( 0, 1, 0), ( 1, 1, 0)))
model = pyqcm.lattice_model('2x2_C2', clus, [( 2, 0, 0), ( 0, 2, 0)])
model.interaction_operator('U')

model.hopping_operator('t', ( 0, 1, 0), -1)
model.hopping_operator('t', ( 1, 0, 0), -1)
model.hopping_operator('t2', ( 1, 1, 0), -1)
model.hopping_operator('t2', (-1, 1, 0), -1)
model.hopping_operator('hx', [0, 0, 0], -1, tau=0, sigma=1)
model.density_wave('M', 'Z', ( 1, 1, 0))
model.anomalous_operator('D', ( 1, 0, 0), 1)
model.anomalous_operator('D', ( 0, 1, 0), -1)
#--------------------------------------------------------------------------------


L = 4

pyqcm.set_global_parameter('GF_method', 'L')
model.set_target_sectors('R0')

# Simulation parameters
model.set_parameters("""
    U=8
    mu=1
    t=1
    t2=0.3
	hx = 0.2
	D = 0.1
""")

import matplotlib.pyplot as plt
I = model.model_instance()
w = np.arange(-5, 5, 0.1)

live = False

if live:
    I.spectral_function(w=6, nk=32, period='G')
    I.mdc()
    I.cluster_spectral_function(w=6)
    I.spectral_function_Lehmann(nk=16)
    I.plot_DoS(np.arange(-2, 2, 0.1), eta=0.2, progress=True)
    I.spin_mdc(nk=64)
    I.mdc_anomalous(nk=64)
    I.plot_dispersion(nk=32)
    I.segment_dispersion(nk=32)
    I.segment_dispersion_fat(orb=1, nk=32)
    I.Fermi_surface(nk=32)
    I.G_dispersion(nk=32)
    I.Luttinger_surface(nk=64)
    I.plot_momentum_profile('t', nk=32)
    I.Berry_curvature(nk=32)

else:
    nrows, ncols = 5, 3
    fig, ax = plt.subplots(nrows, ncols)
    fig.set_size_inches(ncols * 8 / 2.54, nrows * 6 / 2.54)
    ax = ax.flatten()

    I.spectral_function(w=6, nk=32, period='G', plt_ax=ax[0]); ax[0].set_title('spectral_function', fontsize=9)
    I.mdc(plt_ax=ax[1]); ax[1].set_title('mdc', fontsize=9)
    I.cluster_spectral_function(w=6, plt_ax=ax[2]); ax[2].set_title('cluster_spectral_function', fontsize=9)
    I.spectral_function_Lehmann(nk=16, plt_ax=ax[3]); ax[3].set_title('spectral_function_Lehmann', fontsize=9)
    I.plot_DoS(np.arange(-2, 2, 0.1), eta=0.2, progress=True, plt_ax=ax[4]); ax[4].set_title('plot_DoS', fontsize=9)
    I.spin_mdc(nk=64, plt_ax=ax[5]); ax[5].set_title('spin_mdc', fontsize=9)
    I.mdc_anomalous(nk=64, plt_ax=ax[6]); ax[6].set_title('mdc_anomalous', fontsize=9)
    ax[7] = I.plot_dispersion(nk=32, plt_ax=ax[7]); ax[7].set_title('plot_dispersion', fontsize=9)
    I.segment_dispersion(nk=32, plt_ax=ax[8]); ax[8].set_title('segment_dispersion', fontsize=9)
    I.segment_dispersion_fat(orb=1, nk=32, plt_ax=ax[9]); ax[9].set_title('segment_dispersion_fat', fontsize=9)
    I.Fermi_surface(nk=32, plt_ax=ax[10]); ax[10].set_title('Fermi_surface', fontsize=9)
    ax[11] = I.G_dispersion(nk=32, plt_ax=ax[11]); ax[11].set_title('G_dispersion', fontsize=9)
    I.Luttinger_surface(nk=64, plt_ax=ax[12]); ax[12].set_title('Luttinger_surface', fontsize=9)
    I.plot_momentum_profile('t', nk=32, plt_ax=ax[13]); ax[13].set_title('plot_momentum_profile', fontsize=9)
    I.Berry_curvature(nk=32, plt_ax=ax[14]); ax[14].set_title('Berry_curvature', fontsize=9)

    plt.tight_layout()
    plt.savefig('test_plotting.pdf')

