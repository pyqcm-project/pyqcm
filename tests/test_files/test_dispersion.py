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

K = np.array([(1,1,0.)])
print('testing dispersion()')
print(I.dispersion(K))

print('testing epsilon()')
print(I.epsilon(K))
