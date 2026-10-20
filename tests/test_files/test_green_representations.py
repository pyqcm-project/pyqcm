# Test file
# Goal : check model_instance.green_representation() against a direct
#        cluster_Green_function() evaluation, for every combination of the
#        GF_method / combine_mcf global parameters, and check its error paths.
# --------------------------------------------------------------------------------
import numpy as np
import pyqcm
from pyqcm.green_structure import lehmann_matrix, continued_fraction_matrix

# --------------------------------------------------------------------------------
# model definition: a 4-site 1D Hubbard chain, no bath, no symmetry (mixing = 0)

CM = pyqcm.cluster_model(4, n_bath=0)
clus = pyqcm.cluster(CM, [(i, 0, 0) for i in range(4)])
model = pyqcm.lattice_model('1D_4', clus, ((4, 0, 0),))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)

model.set_target_sectors(['R0:N4:S0'])
model.set_parameters("""
U = 4
mu = 2
t = 1
""")

z = 0.7 + 0.3j

def max_diff(G, Gref):
    return np.max(np.abs(np.asarray(G) - np.asarray(Gref)))

# --------------------------------------------------------------------------------
# 1) GF_method = 'L', combine_mcf = False (default) -> lehmann_matrix

pyqcm.set_global_parameter('GF_method', 'L')
pyqcm.set_global_parameter('combine_mcf', False)
I = pyqcm.model_instance(model)
Gref = I.cluster_Green_function(z)
rep = I.green_representation()

print('Lehmann representation:\n', rep)
assert isinstance(rep, lehmann_matrix), type(rep)
err = max_diff(rep.evaluate(z), Gref)
print("L, combine_mcf=False -> lehmann_matrix       : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# 2) GF_method = 'L', combine_mcf = True -> continued_fraction_matrix

pyqcm.set_global_parameter('combine_mcf', True)
I = pyqcm.model_instance(model)
Gref = I.cluster_Green_function(z)
rep = I.green_representation()
print('combined matrix continued fraction representation (from Lehmann):\n', rep)
assert isinstance(rep, continued_fraction_matrix), type(rep)
err = max_diff(rep.evaluate(z), Gref)
print("L, combine_mcf=True  -> continued_fraction_matrix : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# 3) GF_method = 'M', combine_mcf = True -> continued_fraction_matrix

pyqcm.set_global_parameter('GF_method', 'M')
I = pyqcm.model_instance(model)
Gref = I.cluster_Green_function(z)
rep = I.green_representation()
print('combined matrix continued fraction representation (from block Lanczos):\n', rep)
assert isinstance(rep, continued_fraction_matrix), type(rep)
err = max_diff(rep.evaluate(z), Gref)
print("M, combine_mcf=True  -> continued_fraction_matrix : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# 4) GF_method = 'M', combine_mcf = False -> no binding for it, must raise

pyqcm.set_global_parameter('combine_mcf', False)
I = pyqcm.model_instance(model)
try:
    I.green_representation()
    raise AssertionError("expected a RuntimeError for GF_method='M', combine_mcf=False")
except RuntimeError:
    print("M, combine_mcf=False -> raises RuntimeError, as expected")

# --------------------------------------------------------------------------------
# 5) GF_method = 'F' -> plain callable g(z) -> G(z)

pyqcm.set_global_parameter('GF_method', 'F')
I = pyqcm.model_instance(model)
Gref = I.cluster_Green_function(z)
rep = I.green_representation()
assert callable(rep)
err = max_diff(rep(z), Gref)
print("F -> callable                                 : max|diff| =", err)
assert err < 1e-9

# --------------------------------------------------------------------------------
# 6) k is rejected outside the combined-MCF path

pyqcm.set_global_parameter('GF_method', 'L')
I = pyqcm.model_instance(model)
try:
    I.green_representation(k=np.array([0.1, 0.2, 0.0]))
    raise AssertionError("expected a ValueError when k is passed with GF_method='L', combine_mcf=False")
except ValueError:
    print("k rejected for GF_method='L', combine_mcf=False, as expected")

print("\ntest_green_representations: all checks passed")
