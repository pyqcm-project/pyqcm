import pyqcm
import model_1D_4_C2 as M
import numpy as np

M.model.set_target_sectors(['R0:N4:S0'])
M.model.set_parameters("""
t=1
U = 4
mu = 8
""")

##################################################################
# TEST UNITAIRE


def test_fixed_density_loop():

    pyqcm.banner('testing fixed_density_loop()', c='#', skip=1)

    def F():
        return pyqcm.model_instance(M.model)

    M.model.fixed_density_loop(
        F,
        1.1,  # target density
        5,  # starting  value of mu
        kappa=1.0,
        maxdmu=0.5,  # maximum change in mu
        loop_param='U',
        loop_values=np.arange(6, 4, -0.2),
        dens_tol=0.002,
        dir='',
        measure=None
    )


# test_fixed_density_loop()
