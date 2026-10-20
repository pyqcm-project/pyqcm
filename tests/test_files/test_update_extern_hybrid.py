"""
Tests lattice_model.update_extern_hybrid(): replacing the external hybridization of a model in use.

A 1D chain of 2-site clusters is coupled to an external hybridization read from a file. Two files A and B
are written, with different frequency grids and amplitudes. The reference results (CPT Green function on
the grid, lattice averages, CDMFT host) are computed with a model defined directly with file B. The model is
then redefined with file A, an instance is solved, the hybridization is replaced by B, and we check that
  - the existing instance and a new instance both reproduce the reference results,
  - an invalid file (wrong matrix dimension) is rejected and leaves the current hybridization in place,
  - with a file on the real axis (eta != 0), the Green function can be computed on the grid, but lattice averages
    and the CDMFT host are refused,
  - a CDMFT host rejected because of a wrong frequency grid does not leave a stale host behind,
  - unsupported mixing states are rejected (hybridization in the full Nambu mixing, or a normal hybridization
    with a model in the full Nambu mixing).
"""
import numpy as np
import pyqcm
from pyqcm.TRIQS_converter import write_hybrid_file

L = 2
nk = 8
k = np.array([[i / nk, 0, 0] for i in range(nk)])  # superdual basis


def write_file(filename, grid, amplitude, d=L, mixing=0, eta=0.0):
    """hybridization of each site with a noninteracting level at energy 1: amplitude^2/(iw - 1)"""
    w, weight = pyqcm.legendre_frequency_grid(*grid)
    hyb = np.zeros((len(w), nk, d, d), dtype=complex)
    for iw in range(len(w)):
        for ik in range(nk):
            hyb[iw, ik] = np.eye(d) * amplitude**2 / (1j * w[iw] - 1.0) * (1 + 0.2 * np.cos(2 * np.pi * k[ik, 0]))
    write_hybrid_file(filename, w, weight, k, hyb, mixing, eta)
    return w, weight


wA, weightA = write_file('hybrid_A.h5', (1, 10, 5, 5, 5), 0.5)
wB, weightB = write_file('hybrid_B.h5', (1, 8, 4, 6, 4), 0.8)
write_file('hybrid_bad.h5', (1, 8, 4, 6, 4), 0.8, d=3)
write_file('hybrid_real_axis.h5', (1, 8, 4, 6, 4), 0.8, eta=0.1)
write_file('hybrid_mixing3.h5', (1, 8, 4, 6, 4), 0.8, d=4 * L, mixing=3)
assert len(wA) != len(wB)


def define_model(hybrid_file, full_Nambu=False):
    """if full_Nambu is True, pairing and spin-flip terms put the model in the full Nambu mixing"""
    CM = pyqcm.cluster_model(L)
    clus = pyqcm.cluster(CM, [(i, 0, 0) for i in range(L)])
    model = pyqcm.lattice_model('1D_2', clus, ((L, 0, 0),), ((1, 0, 0),), hybrid_file=hybrid_file)
    model.interaction_operator('U')
    model.hopping_operator('t', (1, 0, 0), -1)
    params = """
    t = 1
    U = 4
    mu = 2
    """
    if full_Nambu:
        model.anomalous_operator('S', (0, 0, 0), 1)
        model.hopping_operator('hx', (0, 0, 0), 1, tau=0, sigma=1)
        model.set_target_sectors(['R0'])
        params += """
    S = 0.1
    hx = 0.1
    """
    else:
        model.set_target_sectors(['R0:N2:S0'])
    model.set_parameters(params)
    return model


def results(I, w, weight):
    """CPT Green function on the grid, lattice averages and CDMFT host"""
    G = np.array([[I.CPT_Green_function_grid(iw, ik) for ik in range(nk)] for iw in range(len(w))])
    ave = I.averages()
    pyqcm.qcm.CDMFT_host(w, weight, I.label)
    host = pyqcm.qcm.get_CDMFT_host(0, 0, I.label)
    return G, ave, host


def compare(R1, R2):
    G1, ave1, host1 = R1
    G2, ave2, host2 = R2
    assert G1.shape == G2.shape and np.allclose(G1, G2), 'CPT Green functions differ'
    for x in ave2:
        assert abs(ave1[x] - ave2[x]) < 1e-8, (x, ave1[x], ave2[x])
    assert np.allclose(host1, host2), 'CDMFT hosts differ'


# reference: model defined with file B from the start
model = define_model('hybrid_B.h5')
R_ref = results(pyqcm.model_instance(model), wB, weightB)
pyqcm.reset_model()

# model defined with file A, then updated to B
model = define_model('hybrid_A.h5')
I = pyqcm.model_instance(model)

# a CDMFT host rejected because of a wrong frequency grid must not leave a stale host behind
try:
    pyqcm.qcm.CDMFT_host(wB, weightB, I.label)
except Exception as ex:
    print('rejected as expected: ', ex)
else:
    raise AssertionError('the CDMFT host with the wrong frequency grid should have been rejected')
R_A = results(I, wA, weightA)
assert np.abs(R_A[2]).max() > 1e-3, 'the CDMFT host is empty'
assert abs(R_A[1]['t'] - R_ref[1]['t']) > 1e-4, 'files A and B should give different averages'

model.update_extern_hybrid('hybrid_B.h5')
assert model.hybrid_file == 'hybrid_B.h5'
compare(results(I, wB, weightB), R_ref)                          # existing instance
compare(results(pyqcm.model_instance(model), wB, weightB), R_ref)  # new instance

# the averages stored in the props of the existing instance are those of file B
assert abs(I.props['t_ave'] - R_ref[1]['t']) < 1e-8

# an invalid file is rejected and changes nothing
def expect_error(f, what):
    try:
        f()
    except Exception as ex:
        print('rejected as expected: ', ex)
    else:
        raise AssertionError(what + ' should have been rejected')


expect_error(lambda: model.update_extern_hybrid('hybrid_bad.h5'), 'the file with the wrong dimension')
expect_error(lambda: model.update_extern_hybrid('hybrid_mixing3.h5'), 'the file with mixing = 3')
assert model.hybrid_file == 'hybrid_B.h5'
compare(results(pyqcm.model_instance(model), wB, weightB), R_ref)

# a hybridization on the real axis: only the Green function on the grid is available
model.update_extern_hybrid('hybrid_real_axis.h5')
I = pyqcm.model_instance(model)
G = I.CPT_Green_function_grid(0, 0)
assert np.all(np.isfinite(G))
expect_error(lambda: I.averages(), 'averages with eta != 0')
expect_error(lambda: pyqcm.qcm.CDMFT_host(wB, weightB, I.label), 'the CDMFT host with eta != 0')
model.update_extern_hybrid('hybrid_B.h5')
compare(results(I, wB, weightB), R_ref)
pyqcm.reset_model()

# a normal hybridization cannot be used with a model in the full Nambu mixing
model = define_model('hybrid_A.h5', full_Nambu=True)
expect_error(lambda: pyqcm.model_instance(model), 'a normal hybridization with a full Nambu model')
print('test_update_extern_hybrid: all checks passed')
