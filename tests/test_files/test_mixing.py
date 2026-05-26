# Test file
# Goal : to test the various mixing possibilities with CF and BL
#--------------------------------------------------------------------------------
import pyqcm
import matplotlib.pyplot as plt

def define_model():
    clus = pyqcm.cluster_model(4, name='clus', generators=((4, 3, 2, 1),))
    clus0 = pyqcm.cluster(clus, ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)), pos=(0, 0, 0))
    model = pyqcm.lattice_model('1D_L4', clus0, ((4, 0, 0),))
    model.interaction_operator('U')
    model.interaction_operator('J', link=(1, 0, 0), type='Hund')
    model.hopping_operator('t', (1, 0, 0), -1)
    model.hopping_operator('tp', (2, 0, 0), -1)
    model.hopping_operator('hx', [0, 0, 0], 1, tau=0, sigma=1)
    model.hopping_operator('h', [0, 0, 0], 1, tau=0, sigma=3)
    model.anomalous_operator('D', (1, 0, 0), 1)
    model.anomalous_operator('Di', (1, 0, 0), 1j)
    model.anomalous_operator('S', [0, 0, 0], 1)
    model.anomalous_operator('Si', [0, 0, 0], 1j)
    model.density_wave('H', 'Z', [0, 0, 0])
    model.density_wave('Hx', 'X', [0, 0, 0])
    return model


def test(M, mix):
    pyqcm.set_global_parameter('GF_method', 'B')
    I = M.model_instance()
    I.cluster_spectral_function(w = 4, plt_ax=plt.gca(), file="test_mixing{:d}_nocf.pdf".format(mix))

    pyqcm.set_global_parameter('GF_method', 'C')
    I = M.model_instance()
    I.cluster_spectral_function(w = 4, plt_ax=plt.gca(), file="test_mixing{:d}_cf.pdf".format(mix), color='r')


#--------------------------------------------------------------------------------
# mixing = 0
pyqcm.banner('mixing = 0', '*')
pyqcm.reset_model()
M = define_model()
M.set_target_sectors(['R0:N4:S0'])
M.set_parameters("""
U=1
t=1
mu = 1
""")
test(M,0)



#--------------------------------------------------------------------------------
# mixing = 1
pyqcm.banner('mixing = 1', '*')
pyqcm.reset_model()
M = define_model()
M.set_target_sectors(['R0:S0'])
M.set_parameters("""
U=1
t=1
S = 0.4
H = -1
mu = 1
""")
test(M,1)



#--------------------------------------------------------------------------------
# mixing = 2
pyqcm.banner('mixing = 2', '*')
pyqcm.reset_model()
M = define_model()
M.set_target_sectors(['R0:N4'])
M.set_parameters("""
U=1
t=1
H = -1
Hx = 1e-8 
mu = 1
""")
test(M,2)



#--------------------------------------------------------------------------------
# mixing = 3
pyqcm.banner('mixing = 3', '*')
pyqcm.reset_model()
M = define_model()
M.set_target_sectors(['R0'])
M.set_parameters("""
U=1
t=1
S = 0.4
H = 0.5
Hx = 1
mu = 1
""")
test(M,3)


#--------------------------------------------------------------------------------
# mixing = 4
pyqcm.banner('mixing = 4', '*')
pyqcm.reset_model()
M = define_model()
M.set_target_sectors(['R0:N4:S0'])
M.set_parameters("""
U=1
t=1
H = -1
mu = 1
""")
test(M,4)
