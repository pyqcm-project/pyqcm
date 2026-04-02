import numpy as np
import pyqcm


from model_1D import model1D

L = 8 # length of 1D system
f = 9 # number of floors of the matrix continued fraction (MCF)
model = model1D(L, sym=True)

pyqcm.set_global_parameter('GF_method', 'M')
pyqcm.set_global_parameter('combine_mcf')  # instructs pyqcm to produce a MCF for the Green function

model.set_target_sectors('N{:d}:S0'.format(L))

# Simulation parameters
model.set_parameters("""
    U=4
    mu=2
    t=1
""")


for f in range(1,12):
    pyqcm.set_global_parameter('max_iter_BL', f)
    I = pyqcm.model_instance(model)  

    I.spectral_function(wmax=6, nk=32, path='halfline', period = 'G', file="spectrumG_{:d}_{:d}f.pdf".format(L, f))

    I.spectral_function(wmax=6, nk=32, path='halfline', period = 'L', file="spectrumL_{:d}_{:d}f.pdf".format(L, f))

    pyqcm.set_global_parameter('compact_tiling_per_site')
    I.spectral_function(wmax=6, nk=32, path='halfline', period = 'L', file="spectrumLS_{:d}_{:d}f.pdf".format(L, f))
    pyqcm.set_global_parameter('compact_tiling_per_site', False)

    # printing the MCF
    pyqcm.set_global_parameter('combine_mcf')
    I.combined_mcf(pr=True)
    print('-----------------')
    # printing an example reduced MCF for a given wavevector (pi/2)
    I.combined_mcf(k = np.array([0.25, 0, 0]), pr=True)
