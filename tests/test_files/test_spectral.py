import numpy as np
import pyqcm
import matplotlib.pyplot as plt
import model_2x2_C2 as M

I = pyqcm.model_instance(M.model)

ax = None
F = None
# F = None
#-----------------------------------------------------------------
# import record_2x2_anom
# import record_spin

# DoS(w=4, eta=0.2, file = F, plt_ax=ax)
# Fermi_surface(file = F, plt_ax=ax)
# G_dispersion(max=20, contour=True, file = F, plt_ax=ax)
# Luttinger_surface(file = F, plt_ax=ax)
# cluster_spectral_function(file = F, plt_ax=ax)
# dispersion(file = F, contour=False, plt_ax=ax, labels=True)
# mdc(file = F, plt_ax=ax)
# momentum_profile('t', file = F, plt_ax=ax)
# segment_dispersion(file = F, plt_ax=ax)
# spectral_function(file = F, plt_ax=ax)
# spectral_function_Lehmann(lims=(-5,5), file = F, plt_ax=ax)
# spin_mdc(nk=20, orb=1, file = F, plt_ax=ax, opt='spins')
# spin_mdc(nk=200, orb=1, file = F, plt_ax=ax, opt='spinp')
# spin_mdc(nk=200, orb=1, file = F, plt_ax=ax, opt='sz')
# spin_mdc(nk=25, orb=1, file = F, plt_ax=ax, opt='spins')
# mdc_anomalous(orbs=(1,1), self=True, quadrant=True, file = F, plt_ax=ax)

#-----------------------------------------------------------------
# import record_2x2_8b

# hybridization_function(file = F, plt_ax=ax)

#-----------------------------------------------------------------
# import record_2x2_anom

# plt.show()


##################################################################
# TEST UNITAIRE
Lab = 9
def test_spectral():
    x = None
    # import record_2x2_anom
    # read_cluster_model_instance(record_2x2_anom.solution[0], 9)

    F = 'test_cluster_spectral_function.pdf'
    pyqcm.banner('testing cluster_spectral_function()', c='#', skip=1); I.cluster_spectral_function(file = F, plt_ax=ax)

    F = 'test_DoS.pdf'
    pyqcm.banner('testing DoS()', c='#', skip=1); I.plot_DoS(w=1, eta=0.2, file = F, plt_ax=ax)
    pyqcm.banner('testing DoS()', c='#', skip=1); I.plot_DoS(w=np.linspace(-1,1,40), eta=0.2, file = F, plt_ax=ax)

    F = 'test_G_dispersion.pdf'
    pyqcm.banner('testing G_dispersion()', c='#', skip=1); I.G_dispersion(max=20, file = F, plt_ax=ax)
    
    F = 'test_Luttinger_surface.pdf'
    pyqcm.banner('testing Luttinger_surface()', c='#', skip=1); I.Luttinger_surface(file = F, plt_ax=ax)
    
    F = 'test_mdc.pdf'
    pyqcm.banner('testing mdc()', c='#', skip=1); I.mdc(nk = 50, file = F, plt_ax=ax)
    
    F = 'test_momentum_profile.pdf'
    pyqcm.banner('testing momentum_profile()', c='#', skip=1); I.plot_momentum_profile('t', nk=10, file = F, plt_ax=ax)
    
    F = 'test_spectral_function.pdf'
    pyqcm.banner('testing spectral_function()', c='#', skip=1); I.spectral_function(nk=8, file = F, plt_ax=ax)
    F2 = 'test_spectral_function_M.pdf'
    pyqcm.banner('testing spectral_function()', c='#', skip=1); I.spectral_function(nk=8, file = F2, period='M', plt_ax=ax)
    F2 = 'test_spectral_function_N.pdf'
    pyqcm.banner('testing spectral_function()', c='#', skip=1); I.spectral_function(nk=8, file = F2, period='N', plt_ax=ax)
    F2 = 'test_spectral_function_self.pdf'
    pyqcm.banner('testing spectral_function()', c='#', skip=1); I.spectral_function(nk=8, file = F2, opt='self', offset=30, plt_ax=ax)

    F = 'test_spectral_function_Lehmann.pdf'
    pyqcm.banner('testing spectral_function_Lehmann()', c='#', skip=1); I.spectral_function_Lehmann(lims=(-5,5), nk=8, file = F, plt_ax=ax)
    
    if I.model.mixing&2:
        F = 'test_spin_mdc.pdf'
        pyqcm.banner('testing spin_mdc()', c='#', skip=1); I.spin_mdc(nk=40, opt='spins', file = F, plt_ax=ax)

    F = 'test_plot_hybridization_function.pdf'
    pyqcm.banner('testing plot_hybridization_function()', c='#', skip=1); I.plot_hybridization_function(file = F, plt_ax=ax)

    if I.model.mixing&1:
        F = 'test_mdc_anomalous.pdf'
        pyqcm.banner('testing mdc_anomalous()', c='#', skip=1); I.mdc_anomalous(nk=40, file = F, plt_ax=ax)

        F = 'test_plot_profile.pdf'
        pyqcm.banner('testing plot_profile()', c='#', skip=1); I.plot_profile(file = F)

    F = 'test_Fermi_surface.pdf'
    pyqcm.banner('testing Fermi_surface()', c='#', skip=1); I.Fermi_surface(file = F, plt_ax=ax)
    
    F = 'test_dispersion.pdf'
    pyqcm.banner('testing dispersion()', c='#', skip=1); I.plot_dispersion(file = F, plt_ax=ax)
    
    F = 'test_dispersionC.pdf'
    pyqcm.banner('testing dispersion() with contours', c='#', skip=1); I.plot_dispersion(contour=True, file = F, plt_ax=ax)

    F = 'test_segment_dispersion.pdf'
    pyqcm.banner('testing segment_dispersion()', c='#', skip=1); I.segment_dispersion(file = F, plt_ax=ax)
    

test_spectral()