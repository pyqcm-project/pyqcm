"""
Tests cluster_model.update_operator(): redefining an operator after the model is closed.

A 4-site cluster tiles a 1D chain with uniform hopping t. The operator 't' is then redefined on the
cluster so that the middle bond is weakened. We check that
  - the cluster hopping matrix and the cluster ground state energy agree with an independent cluster model
    defined directly with the new matrix elements,
  - the lattice dispersion agrees with the Bloch Hamiltonian of the dimerized chain (the inter-cluster bond
    is unchanged),
  - restoring the original matrix elements restores the original results,
  - invalid updates are rejected.
"""
import numpy as np
import pyqcm

L = 4
CM = pyqcm.cluster_model(L)
clus = pyqcm.cluster(CM, [(i, 0, 0) for i in range(L)])
model = pyqcm.lattice_model('1D_4', clus, ((L, 0, 0),), ((L, 0, 0),))
model.interaction_operator('U')
model.hopping_operator('t', (1, 0, 0), -1)
model.set_target_sectors(['R0:N4:S0'])
model.set_parameters("""
t = 1
U = 4
mu = 0
""")

# independent cluster model with a weakened middle bond, for reference
t_new = [(1, 2, -1.0), (2, 3, -0.5), (3, 4, -1.0)]
t_new = t_new + [(i + L, j + L, v) for i, j, v in t_new]
CMref = pyqcm.cluster_model(L, name='ref')
CMref.new_operator('t', 'one-body', t_new)
CMref.new_operator('U', 'interaction', [(i + 1, i + 1 + L, 1.0) for i in range(L)])


def bloch_bands(k, bonds):
    """eigenvalues of the Bloch Hamiltonian of the 4-site unit cell (k in units of 2*pi)"""
    H = np.zeros((L, L), dtype=complex)
    for i in range(L - 1):
        H[i, i + 1] = bonds[i]
    H[L - 1, 0] += bonds[L - 1] * np.exp(-2j * np.pi * k * L)
    H += H.conj().T
    return np.linalg.eigvalsh(H)


def check_dispersion(I, bonds):
    for k in np.linspace(0, 0.5, 7):
        e = np.sort(I.dispersion(np.array([k, 0, 0])))
        assert np.allclose(e, bloch_bands(k, bonds)), (k, e, bloch_bands(k, bonds))


# before the update
I0 = pyqcm.model_instance(model)
E0 = I0.ground_state()[0][0]
check_dispersion(I0, [-1, -1, -1, -1])
elem0 = CM.matrix_elements('t')

# redefining the operator t
CM.update_operator('t', t_new)
I1 = pyqcm.model_instance(model)
E1 = I1.ground_state()[0][0]

tc = I1.cluster_hopping_matrix()
tc_ref = np.zeros((L, L))
for i, j, v in t_new[:L - 1]:
    tc_ref[i - 1, j - 1] = tc_ref[j - 1, i - 1] = v
assert np.allclose(tc, tc_ref), tc

pyqcm.qcm.new_model_instanceC('ref', {'t': 1.0, 'U': 4.0}, 'R0:N4:S0', 100)
E_ref = pyqcm.qcm.ground_state_solve(100)[0]
print('E0 = ', E0, '  E1 = ', E1, '  E_ref = ', E_ref)
assert abs(E1 - E_ref) < 1e-8
assert abs(E1 - E0) > 1e-3

check_dispersion(I1, [-1, -0.5, -1, -1])

# the energy of the old instance is unchanged
assert I0.ground_state()[0][0] == E0

# restoring the original operator
CM.update_operator('t', [(e[0] + 1, e[1] + 1, e[2].real) for e in elem0[1] if e[0] <= e[1]])
I2 = pyqcm.model_instance(model)
assert abs(I2.ground_state()[0][0] - E0) < 1e-8
check_dispersion(I2, [-1, -1, -1, -1])

# invalid updates
for op, elem in (
    ('t', [(1, 6, -1.0)]),   # adds spin flip
    ('tp', [(1, 3, -1.0)]),  # no such operator
    ('U', [(1, 5, 1j)]),     # complex interaction
):
    try:
        CM.update_operator(op, elem)
    except Exception as ex:
        print('rejected as expected: ', ex)
    else:
        raise AssertionError(f'update of {op} with {elem} should have been rejected')

# the rejected updates changed nothing
I3 = pyqcm.model_instance(model)
assert abs(I3.ground_state()[0][0] - E0) < 1e-8
check_dispersion(I3, [-1, -1, -1, -1])
print('test_update_operator: all checks passed')
