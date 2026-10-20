# Test file
# Goal : to test the definition of general (two-body) interaction operators on a cluster model.
#
# The operator defined by cluster_model.general_interaction_operator() is
#
#     H = sum_{ijkl} sum_{s,s'}  v_{ijkl} c^+_{i,s} c^+_{j,s'} c_{k,s'} c_{l,s}
#
# This script checks that:
#   1. the Hubbard interaction expressed as a general interaction gives the same result as the
#      built-in interaction operator, both in the spin-summed and in the spin-dependent form;
#   2. the equivalent forms (i,j,k,l) and (j,i,l,k) of the same term are correctly merged, and that
#      the Hermitian conjugate of a term is added automatically when absent;
#   3. a complex-valued general interaction is correctly transmitted (it requires a complex
#      Hilbert space);
#   4. ill-formed or non-Hermitian sets of matrix elements are rejected.
# --------------------------------------------------------------------------------
import numpy as np
import pyqcm

ns = 4  # number of sites of the cluster
sites = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0))


def ground_state(name, define_interaction, params):
    """builds a 2x2 lattice model whose interaction is defined by define_interaction(CM)"""
    CM = pyqcm.cluster_model(ns, name=name)
    define_interaction(CM)
    clus = pyqcm.cluster(CM, sites)
    model = pyqcm.lattice_model(name + '_2x2', clus, ((2, 0, 0), (0, 2, 0)))
    model.hopping_operator('t', (1, 0, 0), -1)
    model.hopping_operator('t', (0, 1, 0), -1)
    model.set_target_sectors(['R0:N4:S0'])
    model.set_parameters(params)
    I = pyqcm.model_instance(model)
    E = I.ground_state()[0][0]
    print('{:24s} : E0 = {:.12f}'.format(name, E))
    return E


# --------------------------------------------------------------------------------
# 1. the Hubbard interaction, three different ways

CM = pyqcm.cluster_model(ns, name='ref')
clus = pyqcm.cluster(CM, sites)
model = pyqcm.lattice_model('ref_2x2', clus, ((2, 0, 0), (0, 2, 0)))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
model.hopping_operator('t', (0, 1, 0), -1)
model.set_target_sectors(['R0:N4:S0'])
model.set_parameters("U = 4\nt = 1\n")
E_ref = pyqcm.model_instance(model).ground_state()[0][0]
print('{:24s} : E0 = {:.12f}'.format('built-in interaction', E_ref))
pyqcm.reset_model()

# same thing as a general interaction : U/2 sum_{i,s,s'} c^+_{i,s} c^+_{i,s'} c_{i,s'} c_{i,s}
E1 = ground_state(
    'spin_summed',
    lambda CM: CM.general_interaction_operator('Ug', [(i, i, i, i, 0.5) for i in range(1, ns + 1)]),
    "Ug_1 = 4\nt = 1\n",
)
pyqcm.reset_model()

# same thing with spin-dependent labels : U sum_i c^+_{i,up} c^+_{i,dn} c_{i,dn} c_{i,up}
E2 = ground_state(
    'spin_dependent',
    lambda CM: CM.general_interaction_operator(
        'Ug', [(i, i + ns, i + ns, i, 1.0) for i in range(1, ns + 1)], spin_dependent=True
    ),
    "Ug_1 = 4\nt = 1\n",
)
pyqcm.reset_model()

assert np.abs(E1 - E_ref) < 1e-9, 'the spin-summed general interaction does not reproduce U'
assert np.abs(E2 - E_ref) < 1e-9, 'the spin-dependent general interaction does not reproduce U'

# --------------------------------------------------------------------------------
# 2. equivalent terms and automatic Hermitian conjugation
#
# Hubbard + nearest-neighbor repulsion + pair hopping between sites 1 and 2.
# The pair hopping term (1,1,2,2) is not Hermitian by itself : its conjugate (2,2,1,1) must be
# supplied, or else added automatically. The density-density term is given here in its two
# equivalent forms (1,2,2,1) and (2,1,1,2), each with half of its value.

full = [(i, i, i, i, 0.5) for i in range(1, ns + 1)]
full += [(1, 2, 2, 1, 0.15), (2, 1, 1, 2, 0.15), (1, 1, 2, 2, 0.2), (2, 2, 1, 1, 0.2)]

short = [(i, i, i, i, 0.5) for i in range(1, ns + 1)]
short += [(1, 2, 2, 1, 0.3), (1, 1, 2, 2, 0.2)]

E3 = ground_state('explicit conjugates', lambda CM: CM.general_interaction_operator('Ug', full), "Ug_1 = 4\nt = 1\n")
pyqcm.reset_model()
E4 = ground_state('implicit conjugates', lambda CM: CM.general_interaction_operator('Ug', short), "Ug_1 = 4\nt = 1\n")
pyqcm.reset_model()

assert np.abs(E3 - E4) < 1e-9, 'the Hermitian conjugates are not correctly added'
assert np.abs(E3 - E_ref) > 1e-6, 'the extra interaction terms have no effect'

# --------------------------------------------------------------------------------
# 3. complex-valued general interaction
#
# A complex pair-hopping amplitude requires a complex Hilbert space. The reference energy below was
# checked against an independent exact diagonalization performed with numpy on the same cluster.
# Conjugating all the matrix elements amounts to conjugating the Hamiltonian, which leaves the
# spectrum unchanged; dropping the imaginary part, on the other hand, does change it.

base = [(i, i, i, i, 0.5) for i in range(1, ns + 1)]

E5 = ground_state('complex', lambda CM: CM.general_interaction_operator(
    'Ug', base + [(1, 1, 2, 2, 0.12 + 0.16j)]), "Ug_1 = 4\nt = 1\n")
pyqcm.reset_model()

E6 = ground_state('conjugate', lambda CM: CM.general_interaction_operator(
    'Ug', base + [(1, 1, 2, 2, 0.12 - 0.16j)]), "Ug_1 = 4\nt = 1\n")
pyqcm.reset_model()

E7 = ground_state('real part only', lambda CM: CM.general_interaction_operator(
    'Ug', base + [(1, 1, 2, 2, 0.12)]), "Ug_1 = 4\nt = 1\n")
pyqcm.reset_model()

assert np.abs(E5 - E6) < 1e-9, 'the spectrum should be invariant under conjugation of the interaction'
assert np.abs(E5 - E7) > 1e-6, 'the imaginary part of the interaction is ignored'
assert np.abs(E5 + 2.073276374388929) < 1e-9, 'the complex interaction does not have the expected value'

# --------------------------------------------------------------------------------
# 4. ill-formed sets of matrix elements

CM = pyqcm.cluster_model(ns, name='errors')
bad = [
    ('non-Hermitian pair', [(1, 1, 2, 2, 0.3), (2, 2, 1, 1, 0.5)], False),
    ('complex self-conjugate term', [(1, 2, 2, 1, 0.3j)], False),
    ('index out of range', [(1, 1, ns + 1, ns + 1, 0.3)], False),
    ('wrong format', [(1, 1, 2, 0.3)], False),
    ('empty operator', [(1, 1, 1, 1, 0.5)], True),
]
for i, (label, elem, spin_dependent) in enumerate(bad):
    try:
        CM.general_interaction_operator('bad{:d}'.format(i), elem, spin_dependent=spin_dependent)
    except ValueError as S:
        print('{:28s} correctly rejected : {:s}'.format(label, str(S)))
    else:
        raise RuntimeError('the matrix elements ({:s}) should have been rejected'.format(label))

print('\nall tests passed')
