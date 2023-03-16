import pyqcm
import matplotlib.pyplot as plt
import model_WSM as M
import numpy as np
I = pyqcm.model_instance(M.model)

# ax = plt.gca()
ax = None
F = None

#-----------------------------------------------------------------
# import record_WSM
# k = np.array([0.695,0,-1e-8])
# Berry_curvature(k_perp=0.01, range=[0.5, 0, 0.5], plane='xz', file=F, plt_ax=ax)
# Berry_field_map(k_perp=0.01, plane='xy', file=F, plt_ax=ax)
# Berry_flux(k, 0.1)
# print('Berry flux : ', Berry_flux(k, 0.1))
# Berry_flux_map(nk=100, k_perp=-0.01, file=F, plt_ax=ax)
# print('Chern number : ', Chern_number())
# print('monopole at ', k, ' : ', monopole(k, a=0.05, nk=40))
# monopole_map(plane='xy', k_perp=0.0, file=F, plt_ax=ax)

# plt.show()
##################################################################
# TEST UNITAIRE

def test_berry():
    x = None
    # import record_WSM
    k = np.array([0.695,0,-1e-8])

    F = 'test_Berry_curvature.pdf'
    pyqcm.banner('testing Berry_curvature()', c='#', skip=1); 
    I.Berry_curvature(nk=20, k_perp=0.01, range=[0.5, 0, 0.5], plane='xz', file=F, plt_ax=ax)

    F = 'test_Berry_field_map.pdf'
    pyqcm.banner('testing Berry_field_map()', c='#', skip=1); 
    I.Berry_field_map(nk=20, k_perp=0.01, plane='xy', file=F, plt_ax=ax)

    pyqcm.banner('testing Berry_flux()', c='#', skip=1); 
    print('Berry flux : ', I.Berry_flux(k, 0.1))
    
    F = 'test_Berry_flux_map.pdf'
    pyqcm.banner('testing Berry_flux_map()', c='#', skip=1); 
    I.Berry_flux_map(nk=20, k_perp=-0.01, file=F, plt_ax=ax)
    
    pyqcm.banner('testing Chern_number()', c='#', skip=1); 
    print('Chern number : ', I.Chern_number())
    
    if I.model.dim == 3:
        pyqcm.banner('testing monopole()', c='#', skip=1); 
        print('monopole at ', k, ' : ', I.monopole(k, a=0.05, nk=20))
    
        F = 'test_monopole_map.pdf'
        pyqcm.banner('testing monopole_map()', c='#', skip=1); 
        I.monopole_map(nk=20, plane='xy', k_perp=0.0, file=F, plt_ax=ax)


test_berry()